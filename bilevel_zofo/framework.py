import os
import copy

from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForTokenClassification
)

from .metrics import calculate_metric
from .trainers.bilevel_minimax_trainer2 import BiLevelMinimaxTrainer2
from .trainers.zo_llm_trainer import ZOLLMTrainer
from .utils import *
from .peft.lora import MultiTaskLoRALinear


def set_active_level(self, active_level):
    """
    Set the active level for the model
    """
    self.active_level = active_level
    for module in self.modules():
        if hasattr(module, "set_bilevel_active_level"):
            module.set_bilevel_active_level(active_level)


class Framework:

    def __init__(self, args, training_tasks, test_tasks, num_tasks_per_iteration=1):

        self.args = args
        self.training_tasks = training_tasks
        self.test_tasks = test_tasks
        self.model, self.tokenizer = self.load_model()
        self.num_tasks_per_iteration = num_tasks_per_iteration

    def load_model(self):
        """
        Load HuggingFace models
        """
        with count_time("Loading model with FP%d" % (16 if self.args.load_float16 else 32)):
            # free_in_GB = int(torch.cuda.mem_get_info()[0] / 1024 ** 3)
            #            print(free_in_GB)
            config = AutoConfig.from_pretrained(self.args.model_name)
            if self.args.untie_emb:
                # Untie embeddings/LM head
                logger.warn("Untie embeddings and LM head")
                config.tie_word_embeddings = False
            if self.args.head_tuning:
                torch_dtype = torch.float32
                if self.args.load_float16:
                    torch_dtype = torch.float16
                elif self.args.load_bfloat16:
                    torch_dtype = torch.bfloat16
                # Head tuning
                if "opt" in self.args.model_name.lower():
                    from .models.modeling_opt import OPTForCausalLM
                    model = OPTForCausalLM.from_pretrained(
                        self.args.model_name,
                        config=config,
                        device_map='auto',
                        torch_dtype=torch_dtype,
                        # max_memory={i: f'{free_in_GB - 5}GB' for i in
                        #             range(torch.cuda.device_count())},
                    )
                elif "llama" in self.args.model_name.lower():
                    from .models.modeling_llama import LlamaForCausalLMWithHeadTuning
                    model = LlamaForCausalLMWithHeadTuning.from_pretrained(
                        self.args.model_name,
                        config=config,
                        device_map='auto',
                        torch_dtype=torch_dtype,
                        # max_memory={i: f'{free_in_GB - 5}GB' for i in
                        #             range(torch.cuda.device_count())},
                    )
                elif "mistral" in self.args.model_name.lower():
                    from .models.modeling_mistral import MistralForCausalLMWithHeadTuning
                    model = MistralForCausalLMWithHeadTuning.from_pretrained(
                        self.args.model_name,
                        config=config,
                        device_map='auto',
                        torch_dtype=torch_dtype,
                        # max_memory={i: f'{free_in_GB - 5}GB' for i in
                        #             range(torch.cuda.device_count())},
                    )
                else:
                    raise NotImplementedError(f"Head tuning is not supported for {self.args.model_name}")
            elif self.args.no_auto_device:
                # No auto device (use for FSDP)
                model = AutoModelForCausalLM.from_pretrained(self.args.model_name, config=config, )
            else:
                # Auto device loading
                torch_dtype = torch.float32
                if self.args.load_float16:
                    torch_dtype = torch.float16
                elif self.args.load_bfloat16:
                    torch_dtype = torch.bfloat16
                model = AutoModelForCausalLM.from_pretrained(self.args.model_name, config=config, device_map='auto',
                                                             torch_dtype=torch_dtype,
                                                             load_in_8bit=self.args.load_int8, )
                # max_memory={i: f'{free_in_GB - 5}GB' for i in
                #             range(torch.cuda.device_count())},
                # load_in_8bit=self.args.load_int8, )
            model.eval()

        # Load tokenizer
        #  In mezo, use_fast is set to False. But TypeError will occur when running SQuaD. Setting to be True can fix.
        tokenizer = AutoTokenizer.from_pretrained(self.args.model_name, use_fast=True)

        # HF tokenizer bug fix
        if "opt" in self.args.model_name:
            tokenizer.bos_token_id = 0

        if ("llama" in self.args.model_name) or ("mistral" in self.args.model_name.lower()):
            # LLaMA padding token
            tokenizer.pad_token_id = 0  # technically <unk>

        # Prefix tuning/LoRA
        if self.args.prefix_tuning:
            if "bilevel_minimax" in self.args.trainer and len(self.training_tasks) > 1:
                raise NotImplementedError("Prefix tuning is not supported for multi-task bilevel minimax")
            if "bilevel_minimax" in self.args.trainer:
                from .peft.prefix_tuning import BilevelPrefixTuning
                BilevelPrefixTuning(model, num_prefix=self.args.num_prefix, reparam=not self.args.no_reparam,
                                    float16=self.args.load_float16, init_by_real_act=self.args.prefix_init_by_real_act)
            else:
                from .peft.prefix_tuning import PrefixTuning
                PrefixTuning(model, num_prefix=self.args.num_prefix, reparam=not self.args.no_reparam,
                             float16=self.args.load_float16, init_by_real_act=self.args.prefix_init_by_real_act)

        if self.args.lora:
            if "bilevel_minimax" in self.args.trainer:
                from .peft.lora import BilevelLoRA
                self.lora = BilevelLoRA(model, r=self.args.lora_r, alpha=self.args.lora_alpha,
                                        float16=self.args.load_float16,
                                        tasks=self.training_tasks)
            else:
                from .peft.lora import LoRA
                self.lora = LoRA(model, r=self.args.lora_r, alpha=self.args.lora_alpha, float16=self.args.load_float16,
                                 tasks=self.training_tasks)

        if self.args.prompt_tuning:
            print("Adding Prompt Tuning to model...")
            if "bilevel_minimax" in self.args.trainer:

                from .peft.prompt_tuning import BilevelPromptTuning
                BilevelPromptTuning(
                    model,
                    num_virtual_tokens=self.args.num_virtual_tokens,
                    init_by_real_tokens=self.args.prompt_init_by_real_tokens,
                    hide_virtual_token_logits=True,  # a workaround for the other loss/prediction functions
                )
                print("Total/Trainable number of parameters in model: {}/{}".format(
                    sum(p.numel() for p in model.parameters()),
                    sum(p.numel() for p in model.parameters() if p.requires_grad),
                ))
            else:
                from .peft.prompt_tuning import PromptTuning
                PromptTuning(
                    model,
                    num_virtual_tokens=self.args.num_virtual_tokens,
                    init_by_real_tokens=self.args.prompt_init_by_real_tokens,
                    hide_virtual_token_logits=True,  # a workaround for the other loss/prediction functions
                )
                print("Total/Trainable number of parameters: {}/{}".format(
                    sum(p.numel() for p in model.parameters()),
                    sum(p.numel() for p in model.parameters() if p.requires_grad),
                ))

        if self.args.head_tuning:
            if model.config.model_type in ["opt", "llama", "mistral"]:
                head_name = "lm_head" if self.args.untie_emb else "embed_tokens"
            else:
                raise NotImplementedError
            for n, p in model.named_parameters():
                if head_name not in n:
                    p.requires_grad = False
                else:
                    logger.info(f"Only tuning {n}")

        if "bilevel_minimax" in self.args.trainer:
            model.set_active_level = set_active_level.__get__(model, type(model))
            model.set_active_level(BILEVEL_ACTIVE_LEVEL.LOWER)

        return model, tokenizer

    def forward(self, input_ids, option_len=None, generation=False):
        """
        Given input_ids and the length of the option, return the log-likelihood of each token in the option.
        For generation tasks, return the generated text.
        This function is only for inference
        """
        input_ids = torch.tensor([input_ids]).to(self.model.device)

        if self.args.mode == 'lora' and len(self.test_tasks) > 1 and "bilevel_minimax" in self.args.trainer:
            # set the merged attribute of all lora layers to True so only the base model is used
            for module in self.model.modules():
                if isinstance(module, MultiTaskLoRALinear):
                    module.original_merged = module.merged
                    module.merged = True

        if generation:
            args = self.args
            # Autoregressive generation
            outputs = self.model.generate(input_ids, do_sample=args.sampling, temperature=args.temperature,
                                          num_beams=args.num_beams, top_p=args.top_p, top_k=args.top_k,
                                          max_new_tokens=min(args.max_new_tokens, args.max_length - input_ids.size(1)),
                                          num_return_sequences=1,
                                          eos_token_id=[
                                              self.tokenizer.encode(args.eos_token, add_special_tokens=False)[-1],
                                              self.tokenizer.eos_token_id], )
            # For generation, directly return the text output
            output_text = self.tokenizer.decode(outputs[0][input_ids.size(1):], skip_special_tokens=True).strip()

            if self.args.mode == 'lora' and len(self.test_tasks) > 1 and "bilevel_minimax" in self.args.trainer:
                # reset the merged attribute
                for module in self.model.modules():
                    if isinstance(module, MultiTaskLoRALinear):
                        module.merged = module.original_merged
            return output_text
        else:
            # with torch.inference_mode():
            #     self.model.eval()
            logits = self.model(input_ids=input_ids).logits
            labels = input_ids[0, 1:]
            logits = logits[0, :-1]
            log_probs = F.log_softmax(logits, dim=-1)

            selected_log_probs = log_probs[torch.arange(len(labels)).to(labels.device), labels]
            selected_log_probs = selected_log_probs.cpu().detach()

            if self.args.mode == 'lora' and len(self.test_tasks) > 1 and "bilevel_minimax" in self.args.trainer:
                # reset the merged attribute
                for module in self.model.modules():
                    if isinstance(module, MultiTaskLoRALinear):
                        module.merged = module.original_merged

            # Only return the option (candidate) part
            return selected_log_probs[-option_len:]

    def one_step_pred(self, task, train_samples, eval_sample, verbose=False):
        """
        Return the prediction on the eval sample. In ICL, use train_samples as demonstrations
        """
        verbose = verbose or self.args.verbose
        # if verbose:
        #     logger.info("========= Example =========")
        #     logger.info(f"Candidate: {eval_sample.candidates}")
        #     logger.info(f"Correct candidate: {eval_sample.correct_candidate}")

        # Encode (add prompt and tokenize) the sample; if multiple-choice/classification, encode all candidates (options)
        encoded_candidates, option_lens = encode_prompt(task,
                                                        task.get_template(template_version=self.args.template_ver),
                                                        train_samples, eval_sample,
                                                        self.tokenizer, max_length=self.args.max_length,
                                                        generation=task.generation,
                                                        max_new_tokens=self.args.max_new_tokens)

        # Calibration
        if self.args.sfc or self.args.icl_sfc:
            sfc_encoded_candidates, sfc_option_lens = encode_prompt(task, task.get_template(
                template_version=self.args.template_ver), train_samples,
                                                                    eval_sample, self.tokenizer,
                                                                    max_length=self.args.max_length, sfc=self.args.sfc,
                                                                    icl_sfc=self.args.icl_sfc,
                                                                    generation=task.generation,
                                                                    max_new_tokens=self.args.max_new_tokens)

        outputs = []
        if task.generation:
            # For generation tasks, return the autoregressively-generated text
            output_text = self.forward(encoded_candidates[0], generation=True)
            # if verbose:
            #     logger.info("=== Prompt ===")
            #     logger.info(self.tokenizer.decode(encoded_candidates[0]))
            #     logger.info(f"Output: {output_text}")
            return Prediction(correct_candidate=eval_sample.correct_candidate, predicted_candidate=output_text)
        else:
            # For classification/multiple-choice, calculate the probabilities of all candidates
            for candidate_id, encoded_candidate in enumerate(encoded_candidates):
                selected_log_probs = self.forward(encoded_candidate, option_len=option_lens[candidate_id])
                if verbose:
                    # if candidate_id == 0:
                    #     logger.info("=== Candidate %d ===" % candidate_id)
                    #     logger.info(self.tokenizer.decode(encoded_candidate))
                    # else:
                    #     logger.info("=== Candidate %d (without context)===" % candidate_id)
                    #     logger.info(self.tokenizer.decode(encoded_candidate).split(self.task.train_sep)[-1])
                    logger.info(f"Log probabilities of the option tokens: {selected_log_probs}")

                if self.args.sfc or self.args.icl_sfc:
                    sfc_selected_log_probs = self.forward(sfc_encoded_candidates[candidate_id],
                                                          option_len=sfc_option_lens[
                                                              candidate_id])
                    # if verbose:
                    #   logger.info("=== Candidate %d (without context) SFC ===" % candidate_id)
                    #   logger.info(
                    #       self.tokenizer.decode(sfc_encoded_candidates[candidate_id]).split(self.task.train_sep)[-1])
                    #   logger.info(f"Log probabilities of the option tokens: {sfc_selected_log_probs}")

                outputs.append({"log_probs": selected_log_probs,
                                "sfc_log_probs": sfc_selected_log_probs if self.args.sfc or self.args.icl_sfc else None})

            if self.args.sfc or self.args.icl_sfc:
                # Calibrated probabilities (surface form competition; https://arxiv.org/pdf/2104.08315.pdf)
                # log p(candidate | input) = log p_lm(candidate | input) - log p_lm(candidate | sfc prompt)
                scores = [x['log_probs'].sum().item() - x['sfc_log_probs'].sum().item() for x in outputs]
            else:
                # (Default) length-normalized log probabilities
                # log p(candidate | input) = log p_lm(candidate | input) / |candidate #tokens|
                scores = [x['log_probs'].mean().item() for x in outputs]

            if verbose:
                logger.info(f"Prediction scores: {scores}")

            if isinstance(eval_sample.correct_candidate, list):
                # For some datasets there are multiple correct answers
                correct_candidate_id = [eval_sample.candidates.index(c) for c in eval_sample.correct_candidate]
            else:
                correct_candidate_id = eval_sample.candidates.index(eval_sample.correct_candidate)

            return Prediction(correct_candidate=correct_candidate_id, predicted_candidate=int(np.argmax(scores)))

    def evaluate(self, task, train_samples, eval_samples, one_train_set_per_eval_sample=False, description=None):
        """
        Evaluate function.
        Here, train_samples are used for demonstrations for ICL.
        If one_train_set_per_eval_sample is True, then each eval sample has its own training (demonstration) set.
        Otherwise, the same training set is used for all eval samples.
        """
        if task is None:
            task = self.test_tasks[0]
        if one_train_set_per_eval_sample:
            logger.info(f"There are {len(eval_samples)} validation samples and one train set per eval sample")
        else:
            logger.info(f"There are {len(train_samples)} training samples and {len(eval_samples)} validation samples")

        # Prediction loop
        predictions = []
        for eval_id, eval_sample in enumerate(tqdm(eval_samples, desc=description)):
            predictions.append(
                self.one_step_pred(task, train_samples[eval_id] if one_train_set_per_eval_sample else train_samples,
                                   eval_sample, verbose=False))

        # Calculate metrics
        metric_name = getattr(task, "metric_name", "accuracy")
        metrics = {metric_name: calculate_metric(predictions, metric_name)}
        return metrics

    def train(self, tasks_train_samples, tasks_dev_samples, tasks_eval_samples):
        """
        Training function
        if self.num_dev is not None, eval_samples are dev_samples
        """

        # Set tokenizer to left padding (so that all the options are right aligned)
        self.tokenizer.padding_side = "left"

        class HFDataset(Dataset):

            def __init__(self, data):
                self.data = data

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]

        def _convert(samples, task):
            """
            Convert samples to HF-compatible dataset
            """
            data = []
            for sample in samples:
                encoded_candidates, option_lens = encode_prompt(task,
                                                                task.get_template(
                                                                    template_version=self.args.template_ver), [],
                                                                sample,
                                                                self.tokenizer, max_length=self.args.max_length,
                                                                generation=task.generation,
                                                                generation_with_gold=True,
                                                                max_new_tokens=self.args.max_new_tokens)
                if task.generation:
                    correct_candidate_id = 0
                elif isinstance(sample.correct_candidate, list):
                    correct_candidate_id = sample.candidates.index(sample.correct_candidate[0])
                else:
                    correct_candidate_id = sample.candidates.index(sample.correct_candidate)

                if self.args.non_diff:
                    # For non-differentiable objective, there is no teacher forcing thus the
                    # current answer part is removed
                    encoded_candidates[correct_candidate_id] = encoded_candidates[correct_candidate_id][
                                                               :-option_lens[correct_candidate_id]]

                if self.args.train_as_classification:
                    # For classification, we provide the label as the correct candidate id
                    data.append([{"input_ids": encoded_candidates[_i], "labels": correct_candidate_id,
                                  "option_len": option_lens[_i], "num_options": len(sample.candidates)} for _i in
                                 range(len(encoded_candidates))])
                elif self.args.only_train_option:
                    # Otherwise, it is just LM-style teacher forcing
                    if self.args.non_diff:
                        # For non-differentiable objective, we need to provide the gold answer to calculate F1/acc
                        data.append({"input_ids": encoded_candidates[correct_candidate_id],
                                     "labels": encoded_candidates[correct_candidate_id],
                                     "option_len": option_lens[correct_candidate_id], "gold": sample.correct_candidate})
                    else:
                        data.append({"input_ids": encoded_candidates[correct_candidate_id],
                                     "labels": encoded_candidates[correct_candidate_id],
                                     "option_len": option_lens[correct_candidate_id]})
                else:
                    data.append({"input_ids": encoded_candidates[correct_candidate_id],
                                 "labels": encoded_candidates[correct_candidate_id]})
            return data

        with count_time("Tokenizing training samples"):
            train_datasets = [HFDataset(_convert(train_samples, self.training_tasks[i])) for i, train_samples in
                              enumerate(tasks_train_samples)]
            eval_datasets = [HFDataset(_convert(eval_samples, self.test_tasks[i])) for i, eval_samples in
                             enumerate(tasks_eval_samples)]
            dev_datasets = [HFDataset(_convert(dev_samples, self.training_tasks[i])) for i, dev_samples in
                            enumerate(tasks_dev_samples)]

            # concatenate all the datasets
            train_dataset = torch.utils.data.ConcatDataset(train_datasets)
            eval_dataset = torch.utils.data.ConcatDataset(eval_datasets)
            dev_dataset = torch.utils.data.ConcatDataset(dev_datasets)

        if self.args.only_train_option and not self.args.non_diff:
            # If --only_train_option and not with a non-differentiable objective, we wrap the forward function
            self.model.original_forward = self.model.forward
            self.model.forward = forward_wrap_with_option_len.__get__(self.model, type(self.model))

        if self.args.non_diff:
            collator = NondiffCollator
        else:
            collator = DataCollatorForTokenClassification

        print(self.args)
        if self.args.trainer == "bilevel_minimax":
            raise NotImplementedError("bilevel_minimax is not supported")
            # self.lower_level_training_args = TrainingArguments(
            #     output_dir="lower_level_output/model",
            #     learning_rate=self.args.lower_level_learning_rate,  # 1e-3
            #     per_device_train_batch_size=self.args.lower_level_per_device_train_batch_size,
            #     per_device_eval_batch_size=self.args.lower_level_per_device_eval_batch_size,
            #     num_train_epochs=self.args.lower_level_num_train_epochs,
            #     max_steps=self.args.lower_level_num_train_steps,
            #     evaluation_strategy="no",
            #     save_strategy="no",
            #     load_best_model_at_end=True,
            #     lr_scheduler_type="constant",
            #     seed=self.args.train_set_seed
            # )
            # trainer = OurBilevelMinimaxTrainer(model=self.model,
            #                                    model_s=self.model_s,
            #                                    args=self.args,
            #                                    lower_level_training_args=self.lower_level_training_args,
            #                                    lower_train_dataset=dev_dataset,
            #                                    train_dataset=train_dataset,
            #                                    eval_dataset=eval_dataset,
            #                                    tokenizer=self.tokenizer,
            #                                    data_collator=DataCollatorWithPaddingAndNesting(self.tokenizer,
            #                                                                                    pad_to_multiple_of=8) if self.args.train_as_classification else collator(
            #                                        self.tokenizer, pad_to_multiple_of=8),
            #                                    eval_samples=eval_samples,
            #                                    dev_samples=dev_samples,
            #                                    evaluate_func=self.evaluate,
            #                                    )  # the upper level uses the dev_dataset for ZO method. the train_dataset in the OurBilevelTrainer is used for upper level updates. Therefore we set train_dataset=dev_dataset
        elif self.args.trainer == "bilevel_minimax2":
            trainer = BiLevelMinimaxTrainer2(model=self.model,
                                             args=self.args,
                                             train_dataset=train_dataset,
                                             eval_dataset=dev_dataset,
                                             dev_dataset=dev_dataset,
                                             tokenizer=self.tokenizer,
                                             training_tasks=self.training_tasks,
                                             test_tasks=self.test_tasks,
                                             data_collator=DataCollatorWithPaddingAndNesting(self.tokenizer,
                                                                                             pad_to_multiple_of=8) if self.args.train_as_classification else collator(
                                                 self.tokenizer, pad_to_multiple_of=8),
                                             tasks_eval_samples=tasks_eval_samples,
                                             evaluate_func=self.evaluate,
                                             num_tasks_per_iteration=self.num_tasks_per_iteration
                                             )
        elif self.args.trainer == "bilevel_minimax_hyper_p":
            raise NotImplementedError("bilevel_minimax_hyper_p is not supported")
            # trainer = OurBilevelMinimaxTrainer(model=self.model,
            #                                    model_s=self.model_s,
            #                                    args=self.args,
            #                                    lower_level_training_args=self.lower_level_training_args,
            #                                    lower_train_dataset=dev_dataset,
            #                                    train_dataset=train_dataset,
            #                                    eval_dataset=eval_dataset,
            #                                    tokenizer=self.tokenizer,
            #                                    data_collator=DataCollatorWithPaddingAndNesting(self.tokenizer,
            #                                                                                    pad_to_multiple_of=8) if self.args.train_as_classification else collator(
            #                                        self.tokenizer, pad_to_multiple_of=8),
            #                                    eval_samples=eval_samples,
            #                                    dev_samples=dev_samples,
            #                                    evaluate_func=self.evaluate,
            #                                    )
        else:
            trainer = ZOLLMTrainer(model=self.model,
                                   args=self.args,
                                   train_dataset=train_dataset,
                                   eval_dataset=dev_dataset,
                                   tokenizer=self.tokenizer,
                                   data_collator=DataCollatorWithPaddingAndNesting(self.tokenizer,
                                                                                   pad_to_multiple_of=8) if self.args.train_as_classification else collator(
                                       self.tokenizer, pad_to_multiple_of=8),
                                   eval_samples=tasks_eval_samples[0],
                                   dev_samples=tasks_dev_samples[0],
                                   evaluate_func=self.evaluate,
                                   )

        if self.args.save_on_interrupt:
            trainer.add_callback(SIGUSR1Callback())

        # Resume training from a last checkpoint
        last_checkpoint = None
        from transformers.trainer_utils import get_last_checkpoint
        if os.path.isdir(self.args.output_dir) and not self.args.overwrite_output_dir:
            last_checkpoint = get_last_checkpoint(self.args.output_dir)
        if last_checkpoint is not None and self.args.resume_from_checkpoint is None:
            logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                        "the `--output_dir` or add `--overwrite_output_dir` to train from scratch.")
        if self.args.resume_from_checkpoint is not None:
            last_checkpoint = self.args.resume_from_checkpoint

        # This calls the trainer._inner_training_loop()
        trainer.train(resume_from_checkpoint=last_checkpoint)

        # Explicitly save the model
        if self.args.save_model:
            logger.info("Save model..")
            trainer.save_model()

        # FSDP compatibility
        self.model = trainer.model

        # Reset the forward function for evaluation
        if self.args.only_train_option and not self.args.non_diff:
            if type(self.model) == FSDP:
                logger.info("This is an FSDP model now. Be careful when assigning back the original forward function")
                self.model._fsdp_wrapped_module.forward = self.model._fsdp_wrapped_module.original_forward
            else:
                self.model.forward = self.model.original_forward

    def delete_checkpoints(self):
        import shutil
        print(f"\nWARNING: Removing everything at end: {self.args.output_dir}")
        deleted_folders = [folder for folder in os.listdir(self.args.output_dir)
                           if os.path.isdir(os.path.join(self.args.output_dir, folder))
                           and folder.startswith("checkpoint-")]
        for f in deleted_folders:
            shutil.rmtree(os.path.join(self.args.output_dir, f))
        print(f"deleted folders: ", deleted_folders)
