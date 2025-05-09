# coding=utf-8
# Copyright 2020-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The Trainer class, to easily train a 🤗 Transformers from scratch or finetune it on a new task.
"""
import copy
import inspect
import itertools

import datasets
import math
import os
import shutil
import sys
import time
from functools import partial
from typing import TYPE_CHECKING, Optional, List, Dict

import numpy as np
import torch
import torch.distributed as dist
from accelerate import skip_first_batches, DistributedType
from accelerate.utils import is_xpu_available, is_mlu_available, is_npu_available, is_torch_version, is_mps_available
from packaging import version
from sklearn.linear_model import LogisticRegressionCV
from torch import nn
from torch.func import functional_call, jvp
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm.auto import tqdm
from transformers import Trainer, is_torch_xla_available, AdamW
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.deepspeed import deepspeed_init
# Integrations must be imported before ML frameworks:
from transformers.integrations import (  # isort: split
    hp_params, deepspeed_load_checkpoint,
)
from transformers.integrations.tpu import tpu_spmd_dataloader
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.trainer import _is_peft_model
from transformers.trainer_callback import (
    DefaultFlowCallback,
    ProgressCallback,
    TrainerState, ExportableState,
)
from transformers.trainer_pt_utils import (
    get_model_param_count, IterableDatasetShard,
)
from transformers.trainer_utils import (
    HPSearchBackend,
    TrainOutput,
    has_length,
    speed_metrics,
    seed_worker,
)
from transformers.training_args import OptimizerNames, ParallelMode
from transformers.utils import (
    is_apex_available,
    is_in_notebook,
    is_sagemaker_mp_enabled,
    is_torch_tpu_available,
    is_datasets_available,
    logging, is_accelerate_available,
)

from datasets import Dataset as HFDataset, Dataset

import wandb
from ..metrics import f1
from ..peft.lora import MultiTaskLoRALinear
from ..utils import BILEVEL_ACTIVE_LEVEL

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    from transformers.utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

if is_apex_available():
    from apex import amp

if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")
    from transformers.trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False

if TYPE_CHECKING:
    pass

logger = logging.get_logger(__name__)

# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"


class BiLevelMinimaxTrainer2(Trainer):

    # ZO-Bench added: new parameters to our trainer
    def __init__(self, evaluate_func, dev_dataset, tasks_eval_samples, training_tasks, test_tasks,
                 *args, num_tasks_per_iteration=5, **kwargs):
        super().__init__(*args, **kwargs)  # Initialize the base class
        self.evaluate_func = evaluate_func
        self.dev_dataset = dev_dataset
        self.eval_samples = tasks_eval_samples
        self.do_grad_scaling = False
        self.training_tasks = training_tasks
        self.test_tasks = test_tasks
        self.num_tasks_per_iteration = min(num_tasks_per_iteration, len(training_tasks))
        self.sampled_tasks = np.sort(np.random.choice(len(training_tasks), self.num_tasks_per_iteration, replace=False))
        logger.info(f"Sampled tasks: {[self.training_tasks[i] for i in self.sampled_tasks]}")

    def _inner_training_loop(
            self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        """
        We overload the original training loop to add linear probing and MeZO. Search key word "MeZO added"
        for those updates.
        """
        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        if self.args.auto_find_batch_size:
            if self.state.train_batch_size != self._train_batch_size:
                from accelerate.utils import release_memory

                (self.model_wrapped,) = release_memory(self.model_wrapped)
                self.model_wrapped = self.model

                # Check for DeepSpeed *after* the intial pass and modify the config
                if self.is_deepspeed_enabled:
                    # Temporarily unset `self.args.train_batch_size`
                    original_bs = self.args.per_device_train_batch_size
                    self.args.per_device_train_batch_size = self._train_batch_size // max(1, self.args.n_gpu)
                    self.propagate_args_to_deepspeed(True)
                    self.args.per_device_train_batch_size = original_bs
            self.state.train_batch_size = self._train_batch_size
        logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")

        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        if self.is_fsdp_xla_v2_enabled:
            train_dataloader = tpu_spmd_dataloader(train_dataloader)

        # MeZO added: Linear probing
        if self.args.linear_probing:

            def _get_token_prediction_layer(model):
                if model.config.model_type in ["opt", "llama", "mistral"]:
                    return model.lm_head
                else:
                    raise NotImplementedError(model.config.model_type)

            def _extract_features(model, *args, **kwargs):
                """some magic for getting features pre last layer"""
                features = {}

                def __hook(model_, input_, output_):
                    features["features"] = input_[0].detach()

                _get_token_prediction_layer(model).register_forward_hook(__hook)
                model.forward(*args, **kwargs)
                return features["features"]

            logger.info("Linear probing")
            logger.info("Starting to get features for training dataset")
            targets = []
            features = []
            with torch.inference_mode():
                for step, inputs in enumerate(tqdm(train_dataloader)):
                    for k, v in inputs.items():
                        if isinstance(v, torch.Tensor):
                            inputs[k] = v.to(self.model.device)

                    feature = _extract_features(self.model, **inputs)
                    target = inputs["labels"]

                    # Shift the target (bc it's autoregressive LM) and add the corresponding part
                    assert not self.args.train_as_classification and self.args.only_train_option
                    feature, target = feature[:, :-1], target[:, 1:]
                    for _i, _len in enumerate(inputs["option_len"]):
                        features.append(feature[_i, -_len:])
                        targets.append(target[_i, -_len:])

            logger.info("Finished getting features for training dataset")

            features = torch.cat(features, dim=0).cpu().numpy()
            targets = torch.cat(targets, dim=0).cpu().numpy()
            # Whether to use bias
            if self.model.config.model_type in ["opt", "gpt2", "llama", "mistral"]:
                use_bias = False
            else:
                raise NotImplementedError
            # Set early stopping
            tol = 0.01 if self.args.lp_early_stopping else 1e-4  # 1e-4 is scipy default
            max_iter = 1000 if self.args.lp_early_stopping else 5000

            logger.info("Fitting logistic regression...")
            reg = LogisticRegressionCV(max_iter=max_iter, fit_intercept=use_bias, multi_class="multinomial",
                                       random_state=0, tol=tol, n_jobs=-1).fit(features, targets)
            logger.info("Done")

            logger.info("Assigning weights to model")
            decoder = _get_token_prediction_layer(self.model)
            coef_torch = torch.tensor(reg.coef_, device=decoder.weight.device, dtype=decoder.weight.dtype)
            if use_bias:
                bias_torch = torch.tensor(reg.intercept_, device=decoder.weight.device, dtype=decoder.weight.dtype)
            if coef_torch.shape[0] == 1:  # The regressor only detects two classes
                assert len(reg.classes_) == 2
                coef_torch = torch.cat([-coef_torch / 2, coef_torch / 2], dim=0)
                if use_bias:
                    bias_torch = torch.cat([-bias_torch / 2, bias_torch / 2], dim=0)

            for _i, token_id in enumerate(reg.classes_):
                decoder.weight.data[token_id] = coef_torch[_i]
                if use_bias:
                    decoder.bias.data[token_id] = bias_torch[_i]

            return None

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        num_train_tokens = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
                if args.include_tokens_per_second:
                    num_train_tokens = (
                            self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
                    )
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
                if args.include_tokens_per_second:
                    num_train_tokens = self.num_tokens(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
            if args.include_tokens_per_second:
                num_train_tokens = self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torchrun or torch.distributed.launch (deprecated))."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled

        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState(
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ]
        )
        self.state.is_hyper_param_search = trial is not None
        self.state.train_batch_size = self._train_batch_size

        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            if args.gradient_checkpointing_kwargs is None:
                gradient_checkpointing_kwargs = {}
            else:
                gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs

            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

        model = self._wrap_model(self.model_wrapped)

        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = True if model is self.model else False

        if delay_optimizer_creation:
            if use_accelerator_prepare:
                self._fsdp_qlora_plugin_updates()
                self.model = self.accelerator.prepare(self.model)
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            self.model.train()
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.)
                model,self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )
        elif self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            # In this case we are in DDP + LOMO, which should be supported
            self.optimizer = self.accelerator.prepare(self.optimizer)

        if self.is_fsdp_enabled:
            self.model = self.model_wrapped = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        # ckpt loading
        if resume_from_checkpoint is not None:
            if self.is_deepspeed_enabled:
                deepspeed_load_checkpoint(
                    self.model_wrapped, resume_from_checkpoint, load_module_strict=not _is_peft_model(self.model)
                )
            elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # overload the optimizer here
        if args.trainer == "zo_adam":
            self.optimizer = AdamW(self.model.parameters(), lr=args.learning_rate)
            assert args.lr_scheduler_type == 'constant', "we did not implement lr_schedule."
        elif args.trainer == "zo_sgd":
            self.optimizer = SGD(self.model.parameters(), lr=args.learning_rate, momentum=args.momentum)
            assert args.lr_scheduler_type == 'constant', "we did not implement lr_schedule."
        elif "bilevel" in args.trainer:
            # load the lower level optimizer
            lower_parameters = [param for name, param in self.model.named_parameters() if "upper_level_model" not in name]
            assert args.lr_scheduler_type == 'constant', "we did not implement lr_schedule."
            if args.optimizer == "adam":
                self.optimizer = Adam(lower_parameters, lr=args.learning_rate)
            elif args.optimizer == "adamw":
                self.optimizer = AdamW(lower_parameters, lr=args.learning_rate)
            elif args.optimizer == "sgd":
                self.optimizer = SGD(lower_parameters, lr=args.learning_rate, momentum=args.momentum)
        else:
            assert args.lr_scheduler_type == 'constant', "we did not implement lr_schedule."
            if args.optimizer == "adam":
                self.optimizer = Adam(self.model.parameters(), lr=args.learning_rate)
            elif args.optimizer == "adamw":
                self.optimizer = AdamW(self.model.parameters(), lr=args.learning_rate)
            elif args.optimizer == "sgd":
                self.optimizer = SGD(self.model.parameters(), lr=args.learning_rate, momentum=args.momentum)

        if args.trainer == "bilevel_minimax2":
            logger.info("loading the upper level optimizer")
            upper_parameters = [param for name, param in self.model.named_parameters() if "lower_level_model" not in name]
            # assert args.upper_lr_scheduler_type == 'constant', "we did not implement lr_schedule."
            if args.upper_optimizer == "adam":
                self.upper_optimizer = Adam(upper_parameters, lr=args.upper_learning_rate)
            elif args.upper_optimizer == "adamw":
                self.upper_optimizer = AdamW(upper_parameters, lr=args.upper_learning_rate)
            elif args.upper_optimizer == "sgd":
                self.upper_optimizer = SGD(upper_parameters, lr=args.upper_learning_rate,
                                           momentum=args.upper_momentum)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
        # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(
                f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
        logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
                os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            self.compare_trainer_and_checkpoint_args(self.args, self.state)
            self._load_callback_state()
            epochs_trained = int(self.state.global_step // num_update_steps_per_epoch)
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()
        grad_norm: Optional[float] = None
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        if args.eval_on_start:
            self._evaluate(trial, ignore_keys_for_eval, skip_scheduler=True)

        # Main training loop
        total_steps = 0
        total_batched_samples = 0
        for epoch in range(epochs_trained, num_train_epochs):
            epoch_iterator = self.get_train_dataloader()
            if hasattr(epoch_iterator, "set_epoch"):
                epoch_iterator.set_epoch(epoch)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                epoch_iterator = skip_first_batches(epoch_iterator, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            step = -1
            train_iterator_p = itertools.cycle(self.get_train_dataloader())
            dev_iterator = itertools.cycle(self.get_dev_dataloader())
            for step, batch in enumerate(epoch_iterator):
                total_batched_samples += 1
                total_steps += 1
                model.set_active_level(BILEVEL_ACTIVE_LEVEL.LOWER)

                if rng_to_sync:
                    self._load_rng_state(resume_from_checkpoint)
                    rng_to_sync = False

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                # torch.cuda.synchronize()
                step_start_time = time.time()

                for task in batch:
                    torch.cuda.empty_cache()
                    inputs = batch[task]
                    if self.args.lora:
                        self.set_active_lora(model, task)

                    if self.args.include_num_input_tokens_seen:
                        main_input_name = getattr(self.model, "main_input_name", "input_ids")
                        if main_input_name not in inputs:
                            logger.warning(
                                "Tried to track the number of tokens seen, however the current model is "
                                "not configured properly to know what item is the input. To fix this, add "
                                "a `main_input_name` attribute to the model class you are using."
                            )
                        else:
                            self.state.num_input_tokens_seen += (
                                torch.sum(
                                    self.accelerator.gather(
                                        torch.tensor(
                                            inputs[main_input_name].numel(), device=self.args.device, dtype=torch.int64
                                        )
                                    )
                                )
                                .cpu()
                                .item()
                            )

                    # MeZO added: estimate gradient
                    if args.trainer in ["zo_sgd", "zo_adam", "zo_sign_opt"]:
                        if args.q == 1:
                            tr_loss_step = self.zo_step(model, inputs)
                        elif args.q > 1:
                            tr_loss_step = self.zo_step_v1(model, inputs)
                        else:
                            raise ValueError(f"q={args.q} is not supported.")
                    elif args.trainer == "zo_conserv":
                        tr_loss_step = self.zo_conserv_step(model, inputs)
                    elif args.trainer == "forward_grad":
                        tr_loss_step = self.forward_grad_step(model, inputs)
                    else:
                        with self.accelerator.accumulate(model):
                            tr_loss_step = self.training_step(model, inputs)
                            torch.cuda.empty_cache()
                            # set all gradients to zero except for peft params
                        if len(self.training_tasks) > 1:
                            self.zero_grad_except_peft(model)

                    if (
                            args.logging_nan_inf_filter
                            and not is_torch_xla_available()
                            and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                    ):
                        # if loss is nan or inf simply add the average of previous logged losses
                        tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                    else:
                        if tr_loss.device != tr_loss_step.device:
                            raise ValueError(
                                f"Calculated loss must be on the original device: {tr_loss.device} but device in use is {tr_loss_step.device}"
                            )
                        tr_loss += tr_loss_step

                    self.current_flos += float(self.floating_point_ops(inputs))

                is_last_step_and_steps_less_than_grad_acc = (
                        steps_in_epoch <= args.gradient_accumulation_steps and (step + 1) == steps_in_epoch
                )

                if (
                        total_batched_samples % args.gradient_accumulation_steps == 0
                        or
                        # last step in epoch but step is always smaller than gradient_accumulation_steps
                        is_last_step_and_steps_less_than_grad_acc
                ):
                    # MeZO added: update model with the estimated gradient
                    if args.trainer in ["zo_sgd", "zo_adam", "zo_sign_opt", "zo_conserv"]:
                        self.zo_update(model)
                    elif args.trainer == "forward_grad":
                        self.forward_grad_update(model)
                    else:
                        if is_last_step_and_steps_less_than_grad_acc:
                            self.accelerator.gradient_state._set_sync_gradients(True)

                        # Gradient clipping
                        if args.max_grad_norm is not None and args.max_grad_norm > 0:
                            # deepspeed does its own clipping

                            if is_sagemaker_mp_enabled() and args.fp16:
                                _grad_norm = self.optimizer.clip_master_grads(args.max_grad_norm)
                            elif self.use_apex:
                                # Revert to normal clipping otherwise, handling Apex or full precision
                                _grad_norm = nn.utils.clip_grad_norm_(
                                    amp.master_params(self.optimizer),
                                    args.max_grad_norm,
                                )
                            else:
                                _grad_norm = self.accelerator.clip_grad_norm_(
                                    model.parameters(),
                                    args.max_grad_norm,
                                )

                            if (
                                    is_accelerate_available()
                                    and self.accelerator.distributed_type == DistributedType.DEEPSPEED
                            ):
                                grad_norm = model.get_global_grad_norm()
                                # In some cases the grad norm may not return a float
                                if hasattr(grad_norm, "item"):
                                    grad_norm = grad_norm.item()
                            else:
                                grad_norm = _grad_norm

                        self.optimizer.step()
                        self.control = self.callback_handler.on_optimizer_step(args, self.state, self.control)

                        optimizer_was_run = not self.accelerator.optimizer_step_was_skipped
                        if optimizer_was_run:
                            # Delay optimizer scheduling until metrics are generated
                            if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                                self.lr_scheduler.step()

                    model.zero_grad()

                    if not self.state.global_step % self.args.lower_level_num_train_steps == 0 or \
                            self.state.global_step == 0:
                        self.state.global_step += 1
                        self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                        self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                        self._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval)

                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                # check whether to do upper level update:
                if self.state.global_step % self.args.lower_level_num_train_steps == 0 and \
                        self.state.global_step != 0:
                    model.set_active_level(BILEVEL_ACTIVE_LEVEL.UPPER)
                    logger.info("\nUpdate upper level model\n")
                    try:
                        tasks_inputs_f = next(dev_iterator)
                    except StopIteration:
                        break

                    try:
                        tasks_inputs_p = next(train_iterator_p)
                    except StopIteration:
                        break

                    for task in tasks_inputs_f:
                        torch.cuda.empty_cache()
                        inputs_f = tasks_inputs_f[task]
                        inputs_p = tasks_inputs_p[task]

                        if self.args.lora:
                            self.set_active_lora(model, task)

                        self.compute_grad_p(model, inputs_f, inputs_p, task, epoch, ignore_keys_for_eval)
                        torch.cuda.empty_cache()
                        self.compute_zo_grad_theta(model, inputs_f, inputs_p)
                        torch.cuda.empty_cache()
                    torch.cuda.empty_cache()
                    self.bilevel_upper_step(model)

                    self.sampled_tasks = np.random.choice(len(self.training_tasks), self.num_tasks_per_iteration,
                                                          replace=False)
                    logger.info(f"Sampled tasks: {[self.training_tasks[i] for i in self.sampled_tasks]}")

                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                    self._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval)

                # torch.cuda.synchronize()
                train_step_duration = time.time() - step_start_time

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    # PyTorch/XLA relies on the data loader to insert the mark_step for
                    # each step. Since we are breaking the loop early, we need to manually
                    # insert the mark_step here.
                    if is_torch_xla_available():
                        xm.mark_step()
                    break

                if self.control.should_log:
                    max_memory_allocated = 0
                    for device_id in range(torch.cuda.device_count()):
                        # this is not accurate since max memory does not happen simultaneously across all devices
                        max_memory_allocated += torch.cuda.max_memory_allocated(device_id)
                    self.log({"peak_mem": max_memory_allocated / 1024 ** 3,
                              "step_consumption": train_step_duration * 1000})
                    wandb.log({"peak_mem": max_memory_allocated / 1024 ** 3,
                               "step_consumption": train_step_duration * 1000})

            if step < 0:
                # Why would this happen? I don't know, but let's be safe.
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_xla_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sure the model has been saved by process 0.
            if is_torch_xla_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        effective_global_step = max(self.state.global_step, 0.001)  # Avoid ZeroDivisionError
        train_loss = self._total_loss_scalar / effective_global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
            num_tokens=num_train_tokens,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        wandb.log(metrics)
        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint, ignore_errors=True)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        # Wait for the checkpoint to be uploaded.
        self._finish_current_push()

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None:
            self._deactivate_neftune(self.model)

        return TrainOutput(self.state.global_step, train_loss, metrics)

    ###########upper level update#######################
    ######################################################
    def compute_upper_loss(self, model, inputs_f, inputs_p, calc_lower_loss=True):
        if calc_lower_loss:
            model.set_active_level(BILEVEL_ACTIVE_LEVEL.LOWER)
            c = self.compute_loss(model, inputs_p)
        else:
            c = 0
        model.set_active_level(BILEVEL_ACTIVE_LEVEL.UPPER)

        a = self.compute_loss(model, inputs_f)
        b = self.compute_loss(model, inputs_p)

        upper_loss = a + self.args.Lambda * (b - c)
        # upper_loss = self.compute_loss(model_p, inputs_f) + self.args.Lambda * (
        #             self.compute_loss(model_p, inputs_p) - self.compute_loss(model, inputs_p))

        return upper_loss

    def compute_grad_p(self, model, inputs_f, inputs_p, trial, epoch, ignore_keys_for_eval):
        model.train()

        inputs_p = self._prepare_inputs(inputs_p)
        inputs_f = self._prepare_inputs(inputs_f)

        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs_p, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss = self.compute_upper_loss(model, inputs_f, inputs_p, calc_lower_loss=False)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach()

    # @torch.no_grad()
    def compute_zo_grad_theta(self, model, inputs_f, inputs_p):
        """
        Estimate gradient of the outer loss and update the parameters. Return the loss from f(theta + z)
        """

        args = self.args

        # What parameters to optimize
        if self.args.prompt_tuning:
            addtional_parameter_name = "prompt_encoder"
        elif self.args.lora:
            addtional_parameter_name = "lora"
        elif self.args.prefix_tuning:
            addtional_parameter_name = "prefix"

        self.named_parameters_for_zo_step = []
        for name, param in model.named_parameters():
            if addtional_parameter_name not in name:
                param.requires_grad = True
                param.grad = None
                self.named_parameters_for_zo_step.append((name, param))

        # Sample the random seed for sampling z
        self.zo_random_seed = np.random.randint(1000000000)

        # First function evaluation
        self.zo_perturb_parameters(scaling_factor=1)

        # calculate perturbed upper level loss
        loss1 = self.zo_forward_upper(model, inputs_f, inputs_p)

        # Second function evaluation
        assert args.q == 1, ("only support q=1 for the memory efficiency. If you want to implement q>1,"
                             " need to store random seeds to save memory. "
                             "In addition, we need to set different random seed for different z in the q-loop.")
        for _ in range(args.q):  # TODO shall we change the seed?
            self.zo_perturb_parameters(scaling_factor=-1)
            loss2 = self.zo_forward_upper(model, inputs_f, inputs_p)
            self.projected_grad = ((loss1 - loss2) / self.args.zo_eps).item()

            # Set the random seed to ensure that we sample the same z for perturbation/update
            torch.manual_seed(self.zo_random_seed)
            # update theta
            for name, param in self.named_parameters_for_zo_step:
                # Resample z
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device,
                                 dtype=param.data.dtype)

                graddiff_times_z = self.projected_grad * z
                if param.grad is None:
                    param.grad = graddiff_times_z / args.q
                else:
                    param.grad += graddiff_times_z / args.q  # NOTE this q division does not work for q>1.
        assert self.args.gradient_accumulation_steps == 1

    def bilevel_upper_step(self, model):
        self.upper_optimizer.step()  # will only update grad that is not None.

        # set the grad for the finetuned parameters theta false and update the theta in the lower level model
        if self.args.prompt_tuning:
            addtional_parameter_name = "prompt_encoder"
        elif self.args.lora:
            addtional_parameter_name = "lora"
        elif self.args.prefix_tuning:
            addtional_parameter_name = "prefix"

        for name, param in model.named_parameters():
            if addtional_parameter_name not in name:
                param.requires_grad = False
                param.grad = None

        model.zero_grad()

    #
    # ############## MeZO ##############

    def zo_perturb_parameters(self, random_seed=None, scaling_factor=1):
        """
        Perturb the parameters with random vector z.
        Input:
        - random_seed: random seed for MeZO in-place perturbation (if it's None, we will use self.zo_random_seed)
        - scaling_factor: theta = theta + scaling_factor * z * eps
        """

        # Set the random seed to ensure that we sample the same z for perturbation/update
        torch.manual_seed(random_seed if random_seed is not None else self.zo_random_seed)

        if "bilevel_minimax" in self.args.trainer:
            for _, param in self.named_parameters_for_zo_step:
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device,
                                 dtype=param.data.dtype)
                param.data = param.data + scaling_factor * z * self.args.zo_eps
        else:
            for _, param in self.named_parameters_to_optim:
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device,
                                 dtype=param.data.dtype)
                param.data = param.data + scaling_factor * z * self.args.zo_eps

    def zo_forward(self, model, inputs):
        """
        Get (no gradient) loss from the model. Dropout is turned off too.
        """
        model.eval()
        if self.args.non_diff:
            # Non-differentiable objective (may require autoregressive generation)
            return self.zo_forward_nondiff(model, inputs)

        with torch.inference_mode():
            inputs = self._prepare_inputs(inputs)
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1:
                # Warning: this is copied from the original Huggingface Trainer. Untested.
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
        return loss.detach()

    def zo_forward_upper(self, model, inputs_f, inputs_p):
        """
        Get (no gradient) loss from the model. Dropout is turned off too.
        """
        model.eval()

        with torch.inference_mode():
            inputs_p = self._prepare_inputs(inputs_p)
            inputs_f = self._prepare_inputs(inputs_f)
            with self.compute_loss_context_manager():
                loss = self.compute_upper_loss(model, inputs_f, inputs_p)
            if self.args.n_gpu > 1:
                # Warning: this is copied from the original Huggingface Trainer. Untested.
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
        return loss.detach()

    def zo_forward_nondiff(self, model, inputs):
        """
        Get (no gradient) non-diffiable loss from the model.
        """
        model.eval()
        assert self.args.task_name == "SQuAD", "Non differentiable objective only supports SQuAD for now."

        with torch.inference_mode():
            inputs = self._prepare_inputs(inputs)
            args = self.args
            outputs = self.model.generate(
                inputs["input_ids"], do_sample=args.sampling, temperature=args.temperature,
                num_beams=args.num_beams, top_p=args.top_p, top_k=args.top_k,
                max_new_tokens=min(args.max_new_tokens, args.max_length - inputs["input_ids"].size(1)),
                num_return_sequences=1,
                eos_token_id=[self.tokenizer.encode(args.eos_token, add_special_tokens=False)[-1],
                              self.tokenizer.eos_token_id],
            )
            output_text = []
            for i in range(len(outputs)):
                output_text.append(
                    self.tokenizer.decode(outputs[i][inputs["input_ids"].size(1):], skip_special_tokens=True).strip())
            f1s = [f1(output_text[i], inputs['gold'][i]) for i in range(len(output_text))]

        return -torch.tensor(np.mean(f1s), dtype=torch.float32)

    @torch.no_grad()
    def zo_step(self, model, inputs):
        """
        Estimate gradient by MeZO. Return the loss from f(theta + z)
        """
        args = self.args

        # What parameters to optimize
        self.named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))
                # # TODO avoid init the memory for grad.
                # param.grad = torch.zeros_like(param.data)
                param.grad = None  # Make sure the grad is empty and will not be updated.

        # Sample the random seed for sampling z
        self.zo_random_seed = np.random.randint(1000000000)

        # First function evaluation
        self.zo_perturb_parameters(scaling_factor=1)
        loss1 = self.zo_forward(model, inputs)

        # Second function evaluation
        assert args.q == 1, "only support q=1 for the memory efficiency. If you want to implement q>1, need to store random seeds to save memory. In addition, we need to set different random seed for different z in the q-loop."
        for _ in range(args.q):  # TODO shall we change the seed?
            if self.args.perturbation_mode == "one_side":
                self.zo_perturb_parameters(scaling_factor=-1)
                loss2 = self.zo_forward(model, inputs)
                self.projected_grad = ((loss1 - loss2) / self.args.zo_eps).item()
            else:  # two side perturbation
                self.zo_perturb_parameters(scaling_factor=-2)
                loss2 = self.zo_forward(model, inputs)
                self.projected_grad = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()

                # Reset model back to its parameters at start of step
                self.zo_perturb_parameters(scaling_factor=1)

            # Set the random seed to ensure that we sample the same z for perturbation/update
            torch.manual_seed(self.zo_random_seed)
            for name, param in self.named_parameters_to_optim:
                # Resample z
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device,
                                 dtype=param.data.dtype)

                if args.trainer == "zo_sign_opt":
                    # ----signOpt_orig
                    # TODo why do we multiply lr here? We will multiply lr twice?
                    graddiff_times_z = np.sign(self.projected_grad) * z
                    # ----signOpt_mul_sign
                    # graddiff_times_z = self._get_learning_rate() * torch.sign(self.projected_grad * z)
                else:
                    # ----mezo original
                    graddiff_times_z = self.projected_grad * z

                # # previous implementation
                # # no param.grad involved
                # param.data -= self._get_learning_rate() * self.projected_grad * z

                # param.grad += graddiff_times_z.detach()
                # more mem-efficient:
                # run optimizer.step here to avoid caching all grad.
                param.grad = graddiff_times_z / args.q  # NOTE this q division does not work for q>1.
                self.optimizer.step()  # will only update grad that is not None.
                # param.data = param.data - graddiff_times_z / args.q  # NOTE this q division does not work for q>1.
                param.grad = None  # avoid further update.

        # for name, param in self.named_parameters_to_optim:
        #     param.grad = param.grad / args.q

        # No gradient accumulation support
        assert self.args.gradient_accumulation_steps == 1

        return loss1

    @torch.no_grad()
    def zo_step_v1(self, model, inputs):
        """
        Estimate gradient by MeZO. Return the loss from f(theta + z)
        Works with q > 1. But for q > 1, it is not memory efficient.
        """
        args = self.args

        # What parameters to optimize
        self.named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))
                # # TODO avoid init the memory for grad.
                # param.grad = torch.zeros_like(param.data)

        for i_q in range(args.q):  # TODO shall we change the seed?
            # Sample the random seed for sampling z
            self.zo_random_seed = np.random.randint(1000000000)

            # First function evaluation
            self.zo_perturb_parameters(scaling_factor=1)
            loss1 = self.zo_forward(model, inputs)

            # Second function evaluation
            if self.args.perturbation_mode == "one_side":
                self.zo_perturb_parameters(scaling_factor=-1)
                loss2 = self.zo_forward(model, inputs)
                self.projected_grad = ((loss1 - loss2) / self.args.zo_eps).item()
            else:  # two side perturbation
                self.zo_perturb_parameters(scaling_factor=-2)
                loss2 = self.zo_forward(model, inputs)
                self.projected_grad = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()

                # Reset model back to its parameters at start of step
                self.zo_perturb_parameters(scaling_factor=1)

            # Set the random seed to ensure that we sample the same z for perturbation/update
            torch.manual_seed(self.zo_random_seed)
            for name, param in self.named_parameters_to_optim:
                # Resample z
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device,
                                 dtype=param.data.dtype)

                if args.trainer == "zo_sign_opt":
                    # ----signOpt_orig
                    graddiff_times_z = np.sign(self.projected_grad) * z
                    # ----signOpt_mul_sign
                    # graddiff_times_z = self._get_learning_rate() * torch.sign(self.projected_grad * z)
                else:
                    # ----mezo original
                    graddiff_times_z = self.projected_grad * z

                # # previous implementation
                # # no param.grad involved
                # param.data -= self._get_learning_rate() * self.projected_grad * z

                # param.grad += graddiff_times_z.detach()
                # more mem-efficient:
                # run optimizer.step here to avoid caching all grad.
                if i_q == 0:
                    param.grad = graddiff_times_z / args.q
                else:
                    param.grad += graddiff_times_z / args.q
                # if i_q == args.q - 1:
                #     self.optimizer.step()  # TODO If q > 1, We cannot use this trick anymore. This will cause repeated update.
                #     # param.data = param.data - graddiff_times_z / args.q  # NOTE this q division does not work for q>1.
                #     param.grad = None

        # for name, param in self.named_parameters_to_optim:
        #     param.grad = param.grad / args.q
        self.optimizer.step()
        self.optimizer.zero_grad()

        # No gradient accumulatio n support
        assert self.args.gradient_accumulation_steps == 1

        return loss1

    @torch.no_grad()
    def zo_step_v2(self, model, inputs):
        """
        Estimate gradient by MeZO. Return the loss from f(theta + z)
        Works with q > 1. But for q > 1, it is not memory efficient.
        """
        args = self.args

        # What parameters to optimize
        self.named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))
                # # TODO avoid init the memory for grad.
                # param.grad = torch.zeros_like(param.data)

        seed_list = []
        projected_grad_list = []
        for i_q in range(args.q):  # TODO shall we change the seed?
            # Sample the random seed for sampling z
            self.zo_random_seed = np.random.randint(1000000000)
            seed_list.append(self.zo_random_seed)

            # First function evaluation
            self.zo_perturb_parameters(scaling_factor=1)
            loss1 = self.zo_forward(model, inputs)

            # Second function evaluation
            if self.args.perturbation_mode == "one_side":
                self.zo_perturb_parameters(scaling_factor=-1)
                loss2 = self.zo_forward(model, inputs)
                self.projected_grad = ((loss1 - loss2) / self.args.zo_eps).item()
            else:  # two side perturbation
                self.zo_perturb_parameters(scaling_factor=-2)
                loss2 = self.zo_forward(model, inputs)
                self.projected_grad = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()

                # Reset model back to its parameters at start of step
                self.zo_perturb_parameters(scaling_factor=1)

            projected_grad_list.append(self.projected_grad)

        # difference from v1: switch the order of for loop
        # to save memory
        for name, param in self.named_parameters_to_optim:
            for i_q in range(args.q):
                # Set the random seed to ensure that we sample the same z for perturbation/update
                torch.manual_seed(seed_list[i_q])

                graddiff_times_z = torch.zeros_like(param.data, device=param.data.device,
                                                    dtype=param.data.dtype)

                # Resample z
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device,
                                 dtype=param.data.dtype)

                if args.trainer == "zo_sign_opt":
                    # ----signOpt_orig
                    graddiff_times_z += np.sign(projected_grad_list[i_q]) * z
                    # ----signOpt_mul_sign
                    # graddiff_times_z = torch.sign(projected_grad_list[i_q] * z)
                else:
                    # ----mezo original
                    graddiff_times_z += projected_grad_list[i_q] * z

                # # previous implementation
                # # no param.grad involved
                # param.data -= self._get_learning_rate() * self.projected_grad * z

                # param.grad += graddiff_times_z.detach()
                # more mem-efficient:
                # run optimizer.step here to avoid caching all grad.
                if i_q == args.q - 1:
                    param.grad = graddiff_times_z.detach()
                    self.optimizer[name].step()
                    # param.data = param.data - graddiff_times_z / args.q  # NOTE this q division does not work for q>1.
                    param.grad = None

        # for name, param in self.named_parameters_to_optim:
        #     param.grad = param.grad / args.q

        # No gradient accumulation support
        assert self.args.gradient_accumulation_steps == 1

        return loss1

    @torch.no_grad()
    def zo_conserv_step(self, model, inputs):
        """
        Estimate gradient by MeZO. Return the loss from f(theta + z)
        update in the conservative way, i.e. 
        reject the update if it's not decreasing
        """
        args = self.args

        # What parameters to optimize
        self.named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))
                param.grad = None

        loss0 = self.zo_forward(model, inputs)

        # Sample the random seed for sampling z
        self.zo_random_seed = np.random.randint(1000000000)

        # First function evaluation
        self.zo_perturb_parameters(scaling_factor=1)
        loss1 = self.zo_forward(model, inputs)

        # Second function evaluation
        if self.args.perturbation_mode == "one_side":
            self.zo_perturb_parameters(scaling_factor=-1)
            loss2 = self.zo_forward(model, inputs)
            self.projected_grad = ((loss1 - loss2) / self.args.zo_eps).item()
        else:  # two side perturbation
            self.zo_perturb_parameters(scaling_factor=-2)
            loss2 = self.zo_forward(model, inputs)
            self.projected_grad = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()

            # Reset model back to its parameters at start of step
            self.zo_perturb_parameters(scaling_factor=1)

        def update_params(sign=1.0):
            # Set the random seed to ensure that we sample the same z for perturbation/update
            torch.manual_seed(self.zo_random_seed)
            for name, param in self.named_parameters_to_optim:
                # Resample z
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device,
                                 dtype=param.data.dtype)

                if args.trainer == "zo_sign_opt":
                    # ----signOpt_orig
                    # TODo why do we multiply lr here? We will multiply lr twice?
                    graddiff_times_z = np.sign(self.projected_grad) * z
                    # ----signOpt_mul_sign
                    # graddiff_times_z = self._get_learning_rate() * torch.sign(self.projected_grad * z)
                else:
                    # ----mezo original
                    graddiff_times_z = self.projected_grad * z

                # # previous implementation
                # # no param.grad involved
                # param.data -= self._get_learning_rate() * self.projected_grad * z

                # param.grad += graddiff_times_z.detach()
                # more mem-efficient:
                # run optimizer.step here to avoid caching all grad.
                param.grad = sign * graddiff_times_z
                # self.optimizer[name].step()
                self.optimizer.step()
                # param.data = param.data - graddiff_times_z / args.q
                param.grad = None

        update_params()
        loss1 = self.zo_forward(model, inputs)

        update_params(sign=-2.0)
        loss2 = self.zo_forward(model, inputs)

        # conduct the update in the conservative way
        # choose from the three and take the minimum loss one
        if loss1 > loss0:
            if loss0 < loss2:
                update_params()
        else:
            if loss1 < loss2:
                update_params(2.0)

        # No gradient accumulation support
        assert self.args.gradient_accumulation_steps == 1

        return loss1

    def zo_update(self, model):
        """
        Update the parameters with the estimated gradients.
        """
        # # Optimizer step
        # self.optimizer.step()
        # print(type(self.optimizer), self.optimizer)
        self.lr_scheduler.step()  # NOTE When we use own optimizer, this will no longer update the lr anymore.
        # self.optimizer.zero_grad()
        # model.zero_grad()

    @staticmethod
    @torch.no_grad()
    def functional_call_loss(params, names, buffers, model, batch):
        params = {k: v for k, v in zip(names, params)}
        outputs = functional_call(model, (params, buffers), tuple(), kwargs=batch)
        return outputs

    def forward_grad_step(self, model, inputs):
        """
        Forward Gradient Method

        Paper: Gradients without Backpropagation
        https://arxiv.org/pdf/2202.08587.pdf
        """
        args = self.args
        # print(model.__dict__)
        # What parameters to optimize
        self.named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))
                param.grad = None
                # this is not memory efficient.
                # param.grad = torch.zeros_like(param.data)

        # Sample the random seed for sampling vs
        self.zo_random_seed = np.random.randint(1000000000)
        torch.manual_seed(self.zo_random_seed)

        loss = 0
        vs = [torch.randn_like(p) for _, p in self.named_parameters_to_optim]

        assert args.q == 1, "q > 1"

        # fixme: this is a workaround for device map error when using jvp
        inputs = {
            k: v.to(device=model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()
        }
        f = partial(
            self.functional_call_loss,
            names=[n for n, _ in self.named_parameters_to_optim], buffers=dict(model.named_buffers()),
            model=model, batch=inputs
        )

        # jvp profiling
        # torch.cuda.reset_peak_memory_stats()
        loss_, jvp_ = jvp(f, (list([p for _, p in self.named_parameters_to_optim]),), (vs,))
        # print(f"JVP peak memory usage: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB")

        # print(len(jvp_), jvp_[0], jvp_[1].shape, len(jvp_[2][0][0]))
        # for v, p in zip(vs, [p for _, p in self.named_parameters_to_optim]):
        #     p.grad += v * jvp_[0].to(p.device)
        jvp_ = jvp_[0]
        with torch.no_grad():
            for v, (n, p) in zip(vs, [(n, p) for n, p in self.named_parameters_to_optim]):
                # p.grad += v * jvp_[0].to(p.device)
                # p.grad.add_(v * jvp_[0].to(p.device))
                # grad = v * jvp_[0].to(p.device) / args.q

                if "bias" not in n and "layer_norm" not in n and "layernorm" not in n:
                    p.data.sub_(self._get_learning_rate() * (v * jvp_.to(p.device) + args.weight_decay * p.data))
                else:
                    p.data.sub_(self._get_learning_rate() * (v * jvp_.to(p.device)))
        loss += loss_[0].item()

        # for name, param in self.named_parameters_to_optim:
        #     param.grad = param.grad / args.q

        # for name, param in self.named_parameters_to_optim:
        #     if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
        #         param.data = param.data - self._get_learning_rate() * (param.grad + args.weight_decay * param.data)
        #     else:
        #         param.data = param.data - self._get_learning_rate() * (param.grad)

        return torch.tensor(loss)

    def forward_grad_update(self, model):
        """
        Update the parameters with the estimated gradients from forward_grad_step.
        """
        args = self.args
        # for name, param in self.named_parameters_to_optim:
        #     if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
        #         param.data = param.data - self._get_learning_rate() * (param.grad + args.weight_decay * param.data)
        #     else:
        #         param.data = param.data - self._get_learning_rate() * (param.grad)

        self.lr_scheduler.step()

    def _set_signature_columns_if_needed(self):
        """
        We overload this function for non-differentiable objective training to pass "gold" -- the gold text for the task
        """
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            # Labels may be named label or label_ids, the default data collator handles that.
            self._signature_columns += list(set(["label", "label_ids"] + self.label_names))
            self._signature_columns += ["gold"]

    # Custom function to interleave batches (as a list of mini-batches, one from each dataset)
    def interleave_batches(self, loaders):
        iter_loaders = [(str(self.training_tasks[i]), iter(loader)) for i, loader in enumerate(loaders) if
                        i in self.sampled_tasks]
        while True:
            batch = {}
            for task, loader_iter in iter_loaders:
                try:
                    batch[task] = (next(loader_iter))  # Append mini-batch (dict) from each dataset
                except StopIteration:
                    return  # Stop when one loader is exhausted
            yield batch  # Yield a list of mini-batch

    def get_dev_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.dev_dataset is None:
            raise ValueError("Trainer: upper level train requires a dev_dataset.")

        dev_dataset = self.dev_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(dev_dataset, HFDataset):
            dev_dataset = self._remove_unused_columns(dev_dataset, description="upper level training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="upper level training")

        # Define DataLoader params (common to all individual dataset loaders)
        dataloader_params = {
            "batch_size": self._train_batch_size,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(dev_dataset, torch.utils.data.IterableDataset):
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        # For ConcatDataset, create separate loaders for each dataset
        if isinstance(dev_dataset, torch.utils.data.ConcatDataset):
            datasets = dev_dataset.datasets  # Access the individual datasets from the ConcatDataset
            loaders = []

            for dataset in datasets:
                dataset_sampler = SubsetRandomSampler(range(len(dataset)))  # Random sampler for each dataset
                loader = DataLoader(
                    dataset,
                    sampler=dataset_sampler,
                    collate_fn=data_collator,
                    **dataloader_params
                )
                loaders.append(loader)

            # Return a DataLoader that yields lists of mini-batches
            return self.accelerator.prepare(self.interleave_batches(loaders))

        # Default dataloader for non-ConcatDataset (fallback)
        else:
            dataloader_params["collate_fn"] = data_collator
            dataloader_params["sampler"] = self._get_eval_sampler() if not isinstance(dev_dataset,
                                                                                      torch.utils.data.IterableDataset) else None
            return self.accelerator.prepare(DataLoader(dev_dataset, **dataloader_params))

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`] that yields batches as lists of mini-batches,
        where each mini-batch is taken from one of the datasets in the ConcatDataset.

        Each batch will be a list of dictionaries, with one dictionary per dataset.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator

        # Handle dataset-specific column removal (if applicable)
        if is_datasets_available() and isinstance(train_dataset, HFDataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        # Define DataLoader params (common to all individual dataset loaders)
        dataloader_params = {
            "batch_size": self._train_batch_size,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            # dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        # For ConcatDataset, create separate loaders for each dataset
        if isinstance(train_dataset, torch.utils.data.ConcatDataset):
            datasets = train_dataset.datasets  # Access the individual datasets from the ConcatDataset
            loaders = []

            for dataset in datasets:
                dataset_sampler = SubsetRandomSampler(range(len(dataset)))  # Random sampler for each dataset
                loader = DataLoader(
                    dataset,
                    sampler=dataset_sampler,
                    collate_fn=data_collator,
                    **dataloader_params
                )
                loaders.append(loader)

            # Return a DataLoader that yields lists of mini-batches
            return self.accelerator.prepare(self.interleave_batches(loaders))

        # Default dataloader for non-ConcatDataset (fallback)
        else:
            dataloader_params["collate_fn"] = data_collator
            dataloader_params["sampler"] = self._get_train_sampler() if not isinstance(train_dataset,
                                                                                       torch.utils.data.IterableDataset) else None
            return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    @staticmethod
    def set_active_lora(model, task):
        for module in model.modules():
            if isinstance(module, MultiTaskLoRALinear):
                module.set_active_lora(task)

    @staticmethod
    def zero_grad_except_peft(model):
        for name, param in model.named_parameters():
            if "lora" not in name or "prefix_" not in name or "prompt_encoder" not in name:
                param.grad = None

    @staticmethod
    def disable_lora(model):
        for module in model.modules():
            if isinstance(module, MultiTaskLoRALinear) and len(module.tasks) > 1:
                module.original_merged = module.merged
                module.merged = True

    @staticmethod
    def enable_lora(model):
        for module in model.modules():
            if isinstance(module, MultiTaskLoRALinear) and len(module.tasks) > 1:
                module.merged = module.original_merged

    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval):
        original_bilevel_active_level = model.active_level
        model.set_active_level(BILEVEL_ACTIVE_LEVEL.UPPER)
        if self.args.mode == "lora":
            self.disable_lora(model)
        super()._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval)

        if self.args.mode == "lora":
            self.enable_lora(model)
        model.set_active_level(original_bilevel_active_level)

    def evaluate(
            self,
            eval_dataset: Optional[Dataset] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (`Dataset`, *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns
                not accepted by the `model.forward()` method are automatically removed. It must implement the `__len__`
                method.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        test_logs = {}
        for i, task_eval_samples in enumerate(self.eval_samples):
            task = self.test_tasks[i]
            if len(task_eval_samples) == 0:
                continue
            test_metrics = self.evaluate_func(task, [], task_eval_samples)
            if "accuracy" in test_metrics:
                test_logs[f'task{str(task)}-test_accuracy'] = test_metrics["accuracy"]
            else:
                keys = list(test_metrics.keys())
                log_dict = {}
                for k in keys:
                    log_dict[f'task{str(task)}-test_' + k] = test_metrics[k]
                    # log_dict['val_' + k] = val_metrics[k]
                test_logs.update(log_dict)

        output.metrics.update(test_logs)
        self.log(output.metrics)

        if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics


    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        token_type_ids = getattr(inputs, "token_type_ids", None)
        if token_type_ids is not None:
            input_ids = inputs.input_ids
            attention_mask = inputs.attention_mask
            labels = None
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[..., :-1, :].contiguous()

            if labels is None:
                labels = input_ids
            labels = labels[..., 1:].contiguous()
            label_mask = token_type_ids[..., 1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            losses = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))  # [batch_size, length]

            losses = losses.view(logits.size(0), logits.size(1)) * label_mask
            loss = torch.sum(losses, axis=1) / torch.sum(label_mask, axis=1)
            loss = loss.mean()
            return (loss, outputs) if return_outputs else loss
        else:
            if self.label_smoother is not None and "labels" in inputs:
                labels = inputs.pop("labels")
            else:
                labels = None
            outputs = model(**inputs)
            # Save past state if it exists
            # TODO: this needs to be fixed and made cleaner later.
            if self.args.past_index >= 0:
                self._past = outputs[self.args.past_index]

            if labels is not None:
                unwrapped_model = self.accelerator.unwrap_model(model)
                if _is_peft_model(unwrapped_model):
                    model_name = unwrapped_model.base_model.model._get_name()
                else:
                    model_name = unwrapped_model._get_name()
                if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                    loss = self.label_smoother(outputs, labels, shift_labels=True)
                else:
                    loss = self.label_smoother(outputs, labels)
            else:
                if isinstance(outputs, dict) and "loss" not in outputs:
                    raise ValueError(
                        "The model did not return a loss from the inputs, only the following keys: "
                        f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                    )
                # We don't use .loss here since the model may return tuples instead of ModelOutput.
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

            return (loss, outputs) if return_outputs else loss