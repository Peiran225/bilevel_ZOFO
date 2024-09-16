import os
import random
import wandb
from transformers import (
    HfArgumentParser,
    TrainingArguments,
)
from bilevel_zofo.data.tasks import get_tasks
from bilevel_zofo.utils import *
from bilevel_zofo.framework import Framework

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@dataclass
class OurArguments(TrainingArguments):
    output_dir: str = "./output"
    # dataset and sampling strategy

    # task name should match the string before Dataset in the Dataset.
    # We support the following task_name: SST2, RTE, CB, BoolQ, WSC, WIC, MultiRC, Copa, ReCoRD, SQuAD, DROP
    training_tasks: Union[str, list[str]] = "SST2"
    test_tasks: Union[str, list[str]] = "SST2"
    num_tasks_per_iteration: int = 1  # number of tasks to sample in each iteration

    lr_scheduler_type: str = 'constant'
    # Number of examples
    num_train: list[int] = 0  # ICL mode: number of demonstrations; training mode: number of training samples
    num_dev: list[int] = None  # (only enabled with training) number of development samples
    num_eval: list[int] = None  # number of evaluation samples
    num_train_sets: int = None  # how many sets of training samples/demos to sample; if None and train_set_seed is None,
    # then we will sample one set for each evaluation sample
    train_set_seed: int = 0  # designated seed to sample training samples/demos
    result_file: str = None  # file name for saving performance; if None, then use the task name, model name, and config

    # Model loading
    model_name: str = "facebook/opt-125m"  # HuggingFace model name
    load_float16: bool = False  # load model parameters as float16
    load_bfloat16: bool = False  # load model parameters as bfloat16
    load_int8: bool = False  # load model parameters as int8
    max_length: int = 2048  # max length the model can take
    no_auto_device: bool = False  # do not load model by auto device; should turn this on when using FSDP

    # Calibration
    sfc: bool = False  # whether to use SFC calibration
    icl_sfc: bool = False  # whether to use SFC calibration for ICL samples

    template_ver: int = 0  # template. For some tasks (SST2, RTE, Copa), we add template ver=1 as the empty template.

    # Training
    trainer: str = "none"
    """
    # options
    # - none: no training -- for zero-shot or in-context learning (ICL)
    # - regular: regular huggingface trainer -- for fine-tuning
    # - zo_sgd: zeroth-order SGD (MeZO) training
    # - zo_conserv: zeroth-order SGD conservative training
    # - zo_adam: zeroth-order Adam training
    # - zo_sign_opt: zeroth-order sign sgd training
    # - forward_grad: forward gradient
    # - bilevel_minimax2
    """
    optimizer: str = "adamw"
    """
    # options
    # - sgd
    # - adam
    # - adamw
    """
    only_train_option: bool = True  # whether to only train the option part of the input
    train_as_classification: bool = False  # take the log likelihood of all options and train as classification
    momentum: float = 0.0  # only work for SGD optimizer

    # MeZO
    zo_eps: float = 1e-3  # eps in MeZO
    perturbation_mode: str = "two_side"
    q: int = 1  # number of Gaussian samples for zeroth-order trainers

    # Prefix tuning
    prefix_tuning: bool = False  # whether to use prefix tuning
    num_prefix: int = 5  # number of prefixes to use
    no_reparam: bool = True  # do not use re-parameterization trick
    prefix_init_by_real_act: bool = True  # initialize prefix by real activations of random words

    # prompt tuning hyperparameters
    prompt_tuning: bool = False  # whether to use prompt tuning
    num_virtual_tokens: int = 10  # number of prompt tokens to use
    prompt_init_by_real_tokens: bool = False  # whether to sample random tokens from Embedding layer

    # LoRA
    lora: bool = False  # whether to use LoRA
    lora_alpha: int = 16  # alpha in LoRA
    lora_r: int = 8  # r in LoRA

    # Generation
    sampling: bool = False  # whether to use sampling
    temperature: float = 1.0  # temperature for generation
    num_beams: int = 1  # number of beams for generation
    top_k: int = None  # top-k for generation
    top_p: float = 0.95  # top-p for generation
    max_new_tokens: int = 50  # max number of new tokens to generate
    eos_token: str = "\n"  # end of sentence token

    # Saving
    save_model: bool = False  # whether to save the model
    no_eval: bool = False  # whether to skip evaluation
    tag: str = ""  # saving tag

    # Linear probing
    linear_probing: bool = False  # whether to do linear probing
    lp_early_stopping: bool = False  # whether to do early stopping in linear probing
    head_tuning: bool = False  # head tuning: only tune the LM head

    # Untie emb/lm_head weights
    untie_emb: bool = False  # untie the embeddings and LM head

    # Display
    verbose: bool = False  # verbose output

    # Non-diff objective
    non_diff: bool = False  # use non-differentiable objective (only support F1 for SQuAD for now)

    # Auto saving when interrupted
    save_on_interrupt: bool = False  # save model when interrupted (useful for long training)

    clean_model_at_end: bool = True  # remove everthing at the end.

    # sparse gradient pruning
    gradient_sparsity: float = None
    sparse_gradient_resample_steps: int = 1
    sparse_gradient_group: str = "layer"
    """
    Options
    ## - global: global sparsity will assign different sparsity to each layer, based on the pretrained weight magnitude
    ## - layer: each layer has the same sparsity
    """

    # module-wise perturbation
    module_wise_perturbation: bool = False
    perturbed_module_level: str = "transformer-block"
    coordinate_perturbation: bool = True  # If True, will update weight right after the gradient is computed
    """
    Options
    ## - transformer-block: perturb one transformer block at a time
    ## - mlp-attn: perturb one mlp/attention layer at a time
    ## - linear: perturb one linear layer at a time
    """

    # evaluation every eval_step
    # eval_steps: int = 10

    # training arguments for the bilevel problem
    Lambda: float = 0
    lower_level_num_train_steps: int = 1
    lower_level_learning_rate: float = 1e-3
    lower_level_per_device_train_batch_size: int = 1
    lower_level_per_device_eval_batch_size: int = 1
    lower_level_num_train_epochs: int = 1
    upper_optimizer: str = "sgd"  # sgd/adam
    upper_learning_rate: float = 1e-9
    upper_momentum: float = 0.0


def parse_args():
    # parser = argparse.ArgumentParser()
    parser = HfArgumentParser(OurArguments)
    args = parser.parse_args_into_dataclasses()[0]
    return args


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def result_file_tag(args):
    """
    Get the result file tag
    """
    save_model_name = args.model_name.split("/")[-1]
    sfc_tag = "-sfc" if args.sfc else ""
    icl_sfc_tag = "-icl_sfc" if args.icl_sfc else ""
    sample_eval_tag = "-sampleeval%d" % args.num_eval if args.num_eval is not None else ""
    sample_train_tag = "-ntrain%d" % args.num_train if args.num_train > 0 else ""
    sample_dev_tag = "-ndev%d" % args.num_dev if args.num_dev is not None else ""
    customized_tag = f"-{args.tag}" if len(args.tag) > 0 else ""
    return (f"train:{args.training_tasks}-test:{args.test_tasks}-{save_model_name}" + sfc_tag + icl_sfc_tag
            + sample_eval_tag + sample_train_tag + sample_dev_tag + customized_tag)


def main():
    args = parse_args()

    if args.prefix_tuning:
        args.mode = "prefix"
    elif args.lora:
        args.mode = "lora"
    elif args.prompt_tuning:
        args.mode = "prompt"
    else:
        args.mode = "ft"
    if "bilevel_minimax" in args.trainer:
        args.tag = (f"{args.trainer}-train:{args.training_tasks}-test:{args.test_tasks}-{args.template_ver}"
                    f"-{args.model_name.split('/')[-1]}"
                    f"-OPTIM_{args.mode}-STEP{args.max_steps}-{args.optimizer}"
                    f"-LR{args.learning_rate}-{args.lr_scheduler_type}"
                    f"-ZOEPS{args.zo_eps}-Q{args.q}"
                    f"-LowerSTEPS{args.lower_level_num_train_steps}"
                    f"-UpperLr{args.upper_learning_rate}"
                    f"-Lambda{args.Lambda}")
    else:
        args.tag = (f"{args.trainer}-train:{args.training_tasks}-test:{args.test_tasks}-{args.template_ver}"
                    f"-{args.model_name.split('/')[-1]}-OPTIM_{args.mode}-STEP{args.max_steps}-{args.optimizer}"
                    f"-LR{args.learning_rate}-{args.lr_scheduler_type}-ZOEPS{args.zo_eps}-Q{args.q}")

    # we only support mylti-task for lora
    if len(args.training_tasks) > 1 and not args.mode == "lora":
        raise NotImplementedError("We currently only support multitask for LoRA")
    args.tag = "momen" + args.tag if args.momentum > 0 else args.tag
    args.run_name = args.tag
    args.output_dir = os.path.join(args.output_dir, args.tag)
    args.result_file = os.path.join(args.output_dir, "result.json")
    os.makedirs(args.output_dir, exist_ok=True)
    args.logging_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(args.logging_dir, exist_ok=True)

    wandb.init(project="zo_bench", name=args.tag, config=vars(args), dir=args.output_dir)

    set_seed(args.seed)
    training_tasks = get_tasks(args.training_tasks)
    test_tasks = get_tasks(args.test_tasks)

    if len(args.num_train) < len(training_tasks):
        assert len(args.num_train) == 1, "If you want to use the same number of training samples for all tasks, " \
                                         "please provide only one number"
        args.num_train = [args.num_train[0]] * len(training_tasks)
    if args.num_dev is None or len(args.num_dev) < len(training_tasks):
        assert args.num_dev is None or len(
            args.num_dev) == 1, "If you want to use the same number of dev samples for all tasks, " \
                                "please provide only one number"
        if args.num_dev is None:
            args.num_dev = [None] * len(training_tasks)
        else:
            args.num_dev = [args.num_dev[0]] * len(training_tasks)
    if args.num_eval is None or len(args.num_eval) < len(test_tasks):
        assert args.num_eval is None or len(
            args.num_eval) == 1, "If you want to use the same number of eval samples for all tasks, " \
                                 "please provide only one number"
        if args.num_eval is None:
            args.num_eval = [None] * len(test_tasks)
        else:
            args.num_eval = [args.num_eval[0]] * len(test_tasks)

    # This function samples both training and validation samples.
    # The validation (dev) samples are also stored in "train_sets"
    # Later the train_samples and dev_samples are separated
    train_sets = [
        training_tasks[i].sample_train_sets(num_train=args.num_train[i], num_dev=args.num_dev[i],
                                            num_eval=None, num_train_sets=args.num_train_sets, seed=args.train_set_seed)
        for i in range(len(training_tasks))]

    train_sets = list(map(list, zip(*train_sets)))

    # Initialize trainer and load model
    framework = Framework(args, training_tasks, test_tasks, args.num_tasks_per_iteration)

    # ZO-Bench Added
    # We add these parameters to evaluate the model during the training.
    # These two parameters will be used in the training loop
    # args.task = task
    # args.framework = framework

    if args.train_set_seed is not None or args.num_train_sets is not None:

        # Training goes to this way

        # Eval samples share one (or multiple) training set(s)
        for train_set_id, tasks_train_samples in enumerate(train_sets):
            train_set_seed = train_set_id if args.train_set_seed is None else args.train_set_seed

            tasks_eval_samples = []
            for i, task in enumerate(test_tasks):
                num_eval = args.num_eval[i]
                if num_eval is not None:
                    eval_samples = task.sample_subset(data_split="valid", seed=train_set_seed, num=num_eval)
                else:
                    eval_samples = task.valid_samples
                tasks_eval_samples.append(eval_samples)

            tasks_dev_samples = []
            tasks_validate_samples = []
            if args.trainer != "none":
                for i, task in enumerate(args.training_tasks):
                    num_dev = args.num_dev[i]
                    if num_dev is not None:
                        dev_samples = tasks_train_samples[i][-num_dev:]
                        tasks_train_samples[i] = tasks_train_samples[i][:-num_dev]
                        logger.info(f"Task {task} has {len(tasks_train_samples[i])} training samples "
                                    f"and {len(dev_samples)} dev samples")
                    else:
                        dev_samples = None
                        logger.info(f"Task {task} has {len(tasks_train_samples[i])} training samples "
                                    f"and no dev samples")
                    tasks_dev_samples.append(dev_samples)

                args.tasks_dev_samples = tasks_dev_samples
                args.tasks_eval_samples = tasks_eval_samples

                # Training
                framework.train(tasks_train_samples,
                                tasks_dev_samples if len(tasks_dev_samples) > 0 else tasks_eval_samples,
                                tasks_eval_samples)
                tasks_metrics = []
                if not args.no_eval:  # This is True
                    tasks_metrics = [framework.evaluate(training_tasks[i], [], tasks_eval_samples[i],
                                                        description="Evaluating on the Test Set")
                                     for i in range(len(training_tasks))]
                    for i, task in enumerate(args.training_tasks):
                        metrics = tasks_metrics[i]
                        metrics["task"] = task
                        _keys = list(metrics.keys())
                        for m in _keys:
                            metrics["test_" + m] = metrics[m]
                        if tasks_dev_samples is not None:
                            dev_metrics = framework.evaluate(framework.training_tasks[i],
                                                             [], tasks_dev_samples[i],
                                                             description="Evaluating on the Validation Set"
                                                             )
                            _keys = list(dev_metrics.keys())
                            for m in _keys:
                                metrics["val_" + m] = dev_metrics[m]
            else:
                assert args.num_dev is None
                # Zero-shot / in-context learning
                tasks_metrics = []
                for i, task in enumerate(framework.training_tasks):
                    tasks_metrics.append(framework.evaluate(task, tasks_train_samples[i], tasks_eval_samples[i]))
            for task_metrics in tasks_metrics:
                logger.info(task_metrics)
                wandb.log(task_metrics)

            if not args.no_eval:
                logger.info("===== Train set %d =====" % train_set_seed)
                for i, task_metrics in enumerate(tasks_metrics):
                    task = args.training_tasks[i]
                    logger.info(task_metrics)
                    wandb.log(task_metrics)
                    if args.local_rank <= 0:
                        write_metrics_to_file(task_metrics, "result/" + result_file_tag(
                            args) + f"-trainset{train_set_id}-task{task}.json" if args.result_file is None else args.result_file)
            if args.trainer != "none" and args.clean_model_at_end:
                framework.delete_checkpoints()

    else:
        # For each eval sample, there is a training set. no training is allowed
        # This is for in-context learning (ICL)
        assert args.trainer == "none"

        tasks_eval_samples = []
        for i, task in enumerate(training_tasks):
            num_eval = args.num_eval[i]
            if num_eval is not None:
                eval_samples = task.sample_subset(data_split="valid", seed=0, num=num_eval)
            else:
                eval_samples = task.valid_samples
            tasks_eval_samples.append(eval_samples)

        tasks_metrics = [framework.evaluate(framework.training_tasks[i], train_sets[i], tasks_eval_samples[i],
                                            one_train_set_per_eval_sample=True) for i in range(len(training_tasks))]
        for i, task_metrics in enumerate(tasks_metrics):
            logger.info(task_metrics)
            wandb.log(task_metrics)
            if args.local_rank <= 0:
                write_metrics_to_file(task_metrics, "result/" + result_file_tag(
                    args) + f"task-{training_tasks[i]}-onetrainpereval.json" if args.result_file is None else args.result_file)


if __name__ == "__main__":
    main()
