import json
import logging
import sys
from dataclasses import dataclass
from typing import List, Union

import numpy as np
from datasets import load_dataset
from torch.utils.data import ConcatDataset

from .templates import *
from ..utils import temp_seed
import os


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_tasks(task_names, data_dir=None, meta_k=None, meta_test_k=None, meta_icl_seed=None):
    instances = []
    if "meta_icl" in task_names:
        assert type(task_names) == str
        meta_icl_task = task_names.split("__")[1]
        split = task_names.split("__")[2]
        tasks_py_dir = os.path.dirname(os.path.abspath(__file__))
        with open(f"{tasks_py_dir}/MetaICL/config/{meta_icl_task}.json") as f:
            meta_icl_tasks = json.load(f)
        task_names = meta_icl_tasks[split]
        instances = []
        for task_name in task_names:
            instance = MetaICLDataset(task_name, data_dir=data_dir, meta_k=meta_k, meta_test_k=meta_test_k,
                                      meta_icl_seed=meta_icl_seed)
            instances.append(instance)
        return instances
    if isinstance(task_names, str):
        task_names = [task_names]
    for task_name in task_names:
        aa = task_name.split("__")
        if len(aa) == 2:
            task_group, subtask = aa
        else:
            task_group = aa[0]
            subtask = None
        class_ = getattr(sys.modules[__name__], f"{task_group}Dataset")
        instance = class_(subtask)
        instances.append(instance)
    return instances


@dataclass
class Sample:
    id: Union[int, None] = None
    data: dict = None
    correct_candidate: Union[str, int, List[str], List[int]] = None
    candidates: Union[str, int, List[str], List[int], None] = None


class Dataset:
    mixed_set = False
    train_sep = "\n\n"
    generation = False  # whether this is a generation task
    subtask = None

    def __init__(self, subtask=None, **kwargs) -> None:
        self.samples = None
        self.subtask = subtask

    def get_task_name(self):
        if self.subtask is None:
            return f"{self.__class__.__name__}"
        return self.subtask

    def load_dataset(self, path, **kwargs):
        raise NotImplementedError

    def get_template(self, template_version=0):
        templates = {0: Template}
        return templates[template_version]

    def build_sample(self, example):
        return

    def sample_train_sets(self, num_train=32, num_dev=None, num_eval=None, num_train_sets=None, seed=None):
        if seed is not None:
            # one train/demo set using the designated seed
            seeds = [seed]
        elif num_train_sets is not None:
            # num_train_sets train/demo sets
            seeds = list(range(num_train_sets))
        else:
            # one train/demo set per evaluation sample
            assert num_dev is None  # not supported
            len_valid_samples = len(self.samples["valid"]) if num_eval is None else num_eval
            with temp_seed(0):
                seeds = np.random.randint(0, 10000, len_valid_samples)

        train_samples = []
        for i, set_seed in enumerate(seeds):
            if self.mixed_set:  # This is always False for now
                raise NotImplementedError
            else:
                if num_dev is not None:
                    train_samples.append(self.sample_subset(data_split="train", seed=set_seed,
                                                            num=num_train + num_dev))  # dev set is included at the end of train set
                    if num_train + num_dev > len(self.samples["train"]):
                        logger.warn("num_train + num_dev > available training examples")
                else:
                    train_samples.append(self.sample_subset(data_split="train", seed=set_seed, num=num_train))
                if num_dev is not None:
                    logger.info(f"Sample train set {len(train_samples[-1])}/{len(self.samples['train'])}")
                    logger.info(f"... including dev set {num_dev} samples")
        return train_samples

    def sample_subset(self, data_split="train", seed=0, num=100, exclude=None):
        with temp_seed(seed):
            samples = self.samples[data_split]
            lens = len(samples)
            index = np.random.permutation(lens).tolist()[:num if exclude is None else num + 1]
            if exclude is not None and exclude in index:
                index.remove(exclude)
            else:
                index = index[:num]
            return [samples[i] for i in index]

    @property
    def valid_samples(self):
        return self.samples["valid"]

    def __str__(self):
        return self.get_task_name()


class SST2Dataset(Dataset):
    train_sep = "\n\n"

    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        d = load_dataset('glue', 'sst2')
        train_d = d["train"]
        validation_d = d["validation"]

        train_samples = [self.build_sample(example) for example in train_d]
        valid_samples = [self.build_sample(example) for example in validation_d]

        self.samples = {"train": train_samples, "valid": valid_samples}

    # for generative tasks, candidates are []
    def build_sample(self, example):
        label = int(example["label"])
        return Sample(id=example["idx"], data=example, correct_candidate=label, candidates=[0, 1])

    def get_template(self, template_version=0):
        return {0: SST2Template, 1: SST2TemplateEmpty}[template_version]()


class CopaDataset(Dataset):
    train_sep = "\n\n"
    mixed_set = False

    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        train_examples = load_dataset('super_glue', "copa")["train"]
        valid_examples = load_dataset('super_glue', "copa")["validation"]

        train_samples = [self.build_sample(example) for example in train_examples]
        valid_samples = [self.build_sample(example) for example in valid_examples]
        self.samples = {"train": train_samples, "valid": valid_samples}

    # for generative tasks, candidates are []
    def build_sample(self, example):
        sample = \
            Sample(
                id=example["idx"],
                data=example,
                candidates=[example["choice1"], example["choice2"]],
                correct_candidate=example[f"choice{example['label'] + 1}"],
            )

        return sample

    def get_template(self, template_version=0):
        return {0: CopaTemplate, 1: CopaTemplateEmpty}[template_version]()


class BoolQDataset(Dataset):
    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        d = load_dataset("boolq")
        train_set = d["train"]
        valid_set = d["validation"]

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example):
        sample = \
            Sample(
                data=example,
                candidates=["Yes", "No"],
                correct_candidate="Yes" if example["answer"] else "No",
            )

        return sample

    def get_template(self, template_version=2):
        return {0: BoolQTemplate, 1: BoolQTemplateV2, 2: BoolQTemplateV3}[template_version]()


class MultiRCDataset(Dataset):

    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        d = load_dataset("super_glue", "multirc")
        train_set = d["train"]
        valid_set = d["validation"]

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example):
        sample = \
            Sample(
                data=example,
                candidates=[0, 1],
                correct_candidate=example['label']
            )

        return sample

    def get_template(self, template_version=0):
        return {0: MultiRCTemplate}[template_version]()


class CBDataset(Dataset):

    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        d = load_dataset("super_glue", "cb")
        train_set = d["train"]
        valid_set = d["validation"]

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example):
        sample = \
            Sample(
                data=example,
                candidates=[0, 1, 2],
                correct_candidate=example['label']
            )

        return sample

    def get_template(self, template_version=0):
        return {0: CBTemplate}[template_version]()


class WICDataset(Dataset):

    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        d = load_dataset("super_glue", "wic")
        train_set = d["train"]
        valid_set = d["validation"]

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example):
        sample = \
            Sample(
                data=example,
                candidates=[0, 1],
                correct_candidate=example['label']
            )

        return sample

    def get_template(self, template_version=0):
        return {0: WICTemplate}[template_version]()


class WSCDataset(Dataset):

    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        d = load_dataset("super_glue", "wsc.fixed")
        train_set = d["train"]
        valid_set = d["validation"]

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example):
        sample = \
            Sample(
                data=example,
                candidates=[0, 1],
                correct_candidate=example['label']
            )

        return sample

    def get_template(self, template_version=0):
        return {0: WSCTemplate}[template_version]()


class ReCoRDDataset(Dataset):

    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        d = load_dataset("super_glue", "record")
        train_set = d["train"]
        valid_set = d["validation"]

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example):
        sample = \
            Sample(
                data=example,
                candidates=example['entities'],
                correct_candidate=example['answers']
            )

        return sample

    def get_template(self, template_version=0):
        return {0: ReCoRDTemplateGPT3}[template_version]()


class RTEDataset(Dataset):

    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        d = load_dataset("super_glue", "rte")
        train_set = d["train"]
        valid_set = d["validation"]

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example):
        sample = \
            Sample(
                data=example,
                candidates=[0, 1],
                correct_candidate=example['label']
            )

        return sample

    def get_template(self, template_version=0):
        return {0: RTETemplate, 1: RTETemplateEmpty}[template_version]()


class SQuADDataset(Dataset):
    generation = True
    metric_name = "f1"

    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset()

    def load_dataset(self, **kwargs):
        dataset = load_dataset("squad", **kwargs)
        train_examples = dataset["train"]
        valid_examples = dataset["validation"]

        train_samples = [self.build_sample(example, idx) for idx, example in enumerate(train_examples)]
        valid_samples = [self.build_sample(example, idx) for idx, example in enumerate(valid_examples)]
        self.samples = {"train": train_samples, "valid": valid_samples}

    # for generative tasks, candidates are []
    def build_sample(self, example, idx):
        answers = example['answers']['text']
        assert len(answers) > 0
        return Sample(
            id=idx,
            data={
                "title": example['title'],
                "context": example['context'],
                "question": example['question'],
                "answers": answers
            },
            candidates=None,
            correct_candidate=answers
        )

    def get_template(self, template_version=0):
        return {0: SQuADv2Template}[template_version]()


class DROPDataset(Dataset):
    metric_name = "f1"
    generation = True

    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset()

    def load_dataset(self, **kwargs):
        dataset = load_dataset("drop")
        train_examples = dataset["train"]
        valid_examples = dataset["validation"]

        train_samples = [self.build_sample(example, idx) for idx, example in enumerate(train_examples)]
        valid_samples = [self.build_sample(example, idx) for idx, example in enumerate(valid_examples)]
        self.samples = {"train": train_samples, "valid": valid_samples}

    # for generative tasks, candidates are []
    def build_sample(self, example, idx):
        answers = example['answers_spans']['spans']
        assert len(answers) > 0
        return Sample(
            id=idx,
            data={
                "context": example['passage'],
                "question": example['question'],
                "answers": answers
            },
            candidates=None,
            correct_candidate=answers
        )

    def get_template(self, template_version=0):
        return {0: DROPTemplate}[template_version]()


class WinoGrandeDataset(Dataset):
    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        train_set = load_dataset('winogrande', 'winogrande_m', split='train', **kwargs)
        valid_set = load_dataset('winogrande', 'winogrande_m', split='validation', **kwargs)

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example):
        """
        Prompt adapted from https://arxiv.org/pdf/2110.08207.pdf
        """
        sentence = example["sentence"]
        context, target = sentence.split("_")
        sample = Sample(
            data=example,
            candidates=[example['option1'] + target, example['option2'] + target],
            correct_candidate=example[f'option{example["answer"]}'] + target,
        )
        return sample

    def get_template(self, template_version=0):
        if template_version == 0:
            return WinoGrandeTemplate()
        else:
            raise NotImplementedError(f"Template version {template_version} not implemented for WinoGrande")


class TweetEvalSentimentDataset(Dataset):
    train_sep = "\n\n"
    metric_name = "clf_f1"

    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        d = load_dataset('tweet_eval', 'sentiment')
        train_d = d["train"]
        validation_d = d["validation"]

        train_samples = [self.build_sample(example) for example in train_d]
        valid_samples = [self.build_sample(example) for example in validation_d]

        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example):
        label = example["label"]
        return Sample(id=None, data=example, correct_candidate=label, candidates=[0, 1, 2])

    def get_template(self, template_version=0):
        return {0: TweetEvalSentimentTemplate, 1: TweetEvalSentimentTemplateEmpty}[template_version]()


class IMDBDataset(Dataset):
    train_sep = "\n\n"

    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        d = load_dataset('imdb')
        train_d = d["train"]
        validation_d = d["test"]

        train_samples = [self.build_sample(example) for example in train_d]
        valid_samples = [self.build_sample(example) for example in validation_d]

        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example):
        label = example["label"]
        return Sample(id=None, data=example, correct_candidate=label, candidates=[0, 1])

    def get_template(self, template_version=0):
        return {0: IMDBTemplate, 1: IMDBTemplateEmpty}[template_version]()


class RottenTomatoesDataset(Dataset):
    train_sep = "\n\n"

    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        d = load_dataset('rotten_tomatoes')
        train_d = d["train"]
        validation_d = d["test"]

        train_samples = [self.build_sample(example) for example in train_d]
        valid_samples = [self.build_sample(example) for example in validation_d]

        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example):
        label = example["label"]
        return Sample(id=None, data=example, correct_candidate=label, candidates=[0, 1])

    def get_template(self, template_version=0):
        return {0: RottenTomatoesTemplate, 1: RottenTomatoesTemplateEmpty}[template_version]()


class EmotionDataset(Dataset):
    train_sep = "\n\n"
    metric_name = "clf_f1"

    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        d = load_dataset('emotion', 'split')
        train_d = d["train"]
        validation_d = d["validation"]

        train_samples = [self.build_sample(example) for example in train_d]
        valid_samples = [self.build_sample(example) for example in validation_d]

        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example):
        label = example["label"]
        return Sample(id=None, data=example, correct_candidate=label, candidates=[0, 1, 2, 3, 4, 5])

    def get_template(self, template_version=0):
        return {0: EmotionTemplate}[template_version]()


class TweetEvalEmotionDataset(Dataset):
    train_sep = "\n\n"
    metric_name = "clf_f1"

    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        d = load_dataset('tweet_eval', 'emotion')
        train_d = d["train"]
        validation_d = d["validation"]

        train_samples = [self.build_sample(example) for example in train_d]
        valid_samples = [self.build_sample(example) for example in validation_d]

        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example):
        label = example["label"]
        return Sample(id=None, data=example, correct_candidate=label, candidates=[0, 1, 2, 3])

    def get_template(self, template_version=0):
        return {0: TweetEvalEmotionTemplate}[template_version]()


class TweetEvalIronyDataset(Dataset):
    train_sep = "\n\n"

    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        d = load_dataset('tweet_eval', 'irony')
        train_d = d["train"]
        validation_d = d["validation"]

        train_samples = [self.build_sample(example) for example in train_d]
        valid_samples = [self.build_sample(example) for example in validation_d]

        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example):
        label = example["label"]
        return Sample(id=None, data=example, correct_candidate=label, candidates=[0, 1])

    def get_template(self, template_version=0):
        return {0: TweetEvalIronyTemplate}[template_version]()


class AmazonPolarityDataset(Dataset):
    train_sep = "\n\n"

    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        d = load_dataset('mteb/amazon_polarity')
        train_d = d["train"]
        validation_d = d["test"]

        train_samples = [self.build_sample(example) for example in train_d]
        valid_samples = [self.build_sample(example) for example in validation_d]

        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example):
        label = example["label"]
        return Sample(id=None, data=example, correct_candidate=label, candidates=[0, 1])

    def get_template(self, template_version=0):
        return {0: AmazonPolarityTemplate}[template_version]()


class PoemSentimentDataset(Dataset):
    train_sep = "\n\n"
    metric_name = "clf_f1"

    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        d = load_dataset('google-research-datasets/poem_sentiment')
        train_d = d["train"]
        validation_d = d["test"]

        train_samples = [self.build_sample(example) for example in train_d]
        valid_samples = [self.build_sample(example) for example in validation_d]

        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example):
        label = example["label"]
        return Sample(id=example["id"], data=example, correct_candidate=label, candidates=[0, 1, 2, 3])

    def get_template(self, template_version=0):
        return {0: PoemSentimentTemplate}[template_version]()


class YelpReviewPolarityDataset(Dataset):
    train_sep = "\n\n"

    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        d = load_dataset('fancyzhx/yelp_polarity')
        train_d = d["train"]
        validation_d = d["test"]

        train_samples = [self.build_sample(example) for example in train_d]
        valid_samples = [self.build_sample(example) for example in validation_d]

        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example):
        label = example["label"]
        return Sample(id=None, data=example, correct_candidate=label, candidates=[0, 1])

    def get_template(self, template_version=0):
        return {0: YelpReviewPolarityTemplate}[template_version]()


class FinancialPhrasebankDataset(Dataset):
    train_sep = "\n\n"
    metric_name = "clf_f1"

    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        d = load_dataset('yzhuang/financial_phrasebank')
        train_d = d["train"]
        validation_d = d["validation"]

        train_samples = [self.build_sample(example) for example in train_d]
        valid_samples = [self.build_sample(example) for example in validation_d]

        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example):
        label = example["label"][:-1].strip()
        return Sample(id=None, data=example, correct_candidate=label, candidates=["positive", "negative", "neutral"])

    def get_template(self, template_version=0):
        return {0: FinancialPhrasebankTemplate}[template_version]()


class EmoDataset(Dataset):
    train_sep = "\n\n"
    metric_name = "clf_f1"

    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        d = load_dataset('aladar/emo')
        train_d = d["train"]
        validation_d = d["test"]

        train_samples = [self.build_sample(example) for example in train_d]
        valid_samples = [self.build_sample(example) for example in validation_d]

        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example):
        label = example["label"]
        return Sample(id=None, data=example, correct_candidate=label, candidates=[0, 1, 2, 3])

    def get_template(self, template_version=0):
        return {0: EmoTemplate}[template_version]()


class GLUERTEDataset(Dataset):

    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        d = load_dataset('glue', 'rte')
        train_d = d["train"]
        validation_d = d["validation"]

        train_samples = [self.build_sample(example) for example in train_d]
        valid_samples = [self.build_sample(example) for example in validation_d]

        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example):
        label = example["label"]
        return Sample(data=example, correct_candidate=label, candidates=[0, 1])

    def get_template(self, template_version=0):
        return {0: GLUERTETemplate}[template_version]()


class GLUEMRPCDataset(Dataset):
    metric_name = "clf_f1"

    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        d = load_dataset('glue', 'mrpc')
        train_d = d["train"]
        validation_d = d["validation"]

        train_samples = [self.build_sample(example) for example in train_d]
        valid_samples = [self.build_sample(example) for example in validation_d]

        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example):
        label = example["label"]
        return Sample(id=example["idx"], data=example, correct_candidate=label, candidates=[0, 1])

    def get_template(self, template_version=0):
        return {0: GLUEMRPCTemplate}[template_version]()


class GLUEQQPDataset(Dataset):
    metric_name = "clf_f1"

    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        d = load_dataset('glue', 'qqp')
        train_d = d["train"]
        validation_d = d["validation"]

        train_samples = [self.build_sample(example) for example in train_d]
        valid_samples = [self.build_sample(example) for example in validation_d]

        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example):
        label = example["label"]
        return Sample(id=example["idx"], data=example, correct_candidate=label, candidates=[0, 1])

    def get_template(self, template_version=0):
        return {0: GLUEQQPTemplate}[template_version]()


class GLUEMNLIDataset(Dataset):
    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        d = load_dataset('glue', 'mnli')
        train_d = d["train"]
        validation_d = d["validation_matched"]

        train_samples = [self.build_sample(example) for example in train_d]
        valid_samples = [self.build_sample(example) for example in validation_d]

        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example):
        label = example["label"]
        return Sample(id=example["idx"], data=example, correct_candidate=label, candidates=[0, 1, 2])

    def get_template(self, template_version=0):
        return {0: GLUEMNLITemplate}[template_version]()


class SciTailDataset(Dataset):
    metric_name = "clf_f1"

    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        d = load_dataset('allenai/scitail', 'dgem_format')
        train_d = d["train"]
        validation_d = d["validation"]

        train_samples = [self.build_sample(example) for example in train_d]
        valid_samples = [self.build_sample(example) for example in validation_d]

        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example):
        label = example["label"]
        return Sample(id=None, data=example, correct_candidate=label, candidates=["neutral", "entails"])

    def get_template(self, template_version=0):
        return {0: SciTailTemplate}[template_version]()


class GLUEWNLIDataset(Dataset):
    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        d = load_dataset('glue', 'wnli')
        train_d = d["train"]
        validation_d = d["validation"]

        train_samples = [self.build_sample(example) for example in train_d]
        valid_samples = [self.build_sample(example) for example in validation_d]

        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example):
        label = example["label"]
        return Sample(id=example["idx"], data=example, correct_candidate=label, candidates=[0, 1])

    def get_template(self, template_version=0):
        return {0: GLUEWNLITemplate}[template_version]()


class SICKDataset(Dataset):
    metric_name = "clf_f1"

    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        d = load_dataset('sick')
        train_d = d["train"]
        validation_d = d["validation"]

        train_samples = [self.build_sample(example) for example in train_d]
        valid_samples = [self.build_sample(example) for example in validation_d]

        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example):
        label = example["label"]
        return Sample(id=example["id"], data=example, correct_candidate=label, candidates=[0, 1, 2])

    def get_template(self, template_version=0):
        return {0: SICKTemplate}[template_version]()


class ANLIDataset(Dataset):
    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        d = load_dataset('anli')
        train_d = d["train_r1"]
        validation_d = d["dev_r1"]

        train_samples = [self.build_sample(example) for example in train_d]
        valid_samples = [self.build_sample(example) for example in validation_d]

        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example):
        label = example["label"]
        return Sample(id=None, data=example, correct_candidate=label, candidates=[0, 1, 2])

    def get_template(self, template_version=0):
        return {0: ANLITemplate}[template_version]()


class MedicalQuestionsPairsDataset(Dataset):
    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        d = load_dataset('curaihealth/medical_questions_pairs')

        train_samples = [self.build_sample(example) for i, example in enumerate(d["train"]) if i < 2000]
        valid_samples = [self.build_sample(example) for i, example in enumerate(d["train"]) if i > 2000]

        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example):
        label = example["label"]
        return Sample(id=None, data=example, correct_candidate=label, candidates=[0, 1])

    def get_template(self, template_version=0):
        return {0: MedicalQuestionsPairsTemplate}[template_version]()


class HatexplainDataset(Dataset):
    metric_name = "clf_f1"

    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        d = load_dataset('NochnoyRitzar/hatexplain_cleaned')

        train_samples = [self.build_sample(example) for i, example in enumerate(d["train"]) if i < 8000]
        valid_samples = [self.build_sample(example) for i, example in enumerate(d["train"]) if i > 5000]

        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example):
        label = example["major_label"]
        label = 1 if "Hate" in label else 0
        return Sample(id=None, data=example, correct_candidate=label, candidates=[0, 1])

    def get_template(self, template_version=0):
        return {0: HatexplainTemplate}[template_version]()


class HateSpeechOffensiveDataset(Dataset):
    metric_name = "clf_f1"

    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        d = load_dataset('tdavidson/hate_speech_offensive')

        train_samples = [self.build_sample(example) for i, example in enumerate(d["train"]) if i < 10000]
        valid_samples = [self.build_sample(example) for i, example in enumerate(d["train"]) if i > 10000]

        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example):
        label = example["class"]
        return Sample(id=None, data=example, correct_candidate=label, candidates=[0, 1, 2])

    def get_template(self, template_version=0):
        return {0: HateSpeechOffensiveTemplate}[template_version]()


class TweetEvalOffensiveDataset(Dataset):
    metric_name = "clf_f1"

    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        d = load_dataset('tweet_eval', 'offensive')
        train_d = d["train"]
        validation_d = d["validation"]

        train_samples = [self.build_sample(example) for example in train_d]
        valid_samples = [self.build_sample(example) for example in validation_d]

        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example):
        label = example["label"]
        return Sample(id=None, data=example, correct_candidate=label, candidates=[0, 1])

    def get_template(self, template_version=0):
        return {0: TweetEvalOffensiveTemplate}[template_version]()


class EthosDataset(Dataset):
    metric_name = "clf_f1"

    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        d = load_dataset('ethos', 'binary')

        train_samples = [self.build_sample(example) for i, example in enumerate(d["train"]) if i < 800]
        valid_samples = [self.build_sample(example) for i, example in enumerate(d["train"]) if i > 800]

        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example):
        label = example["label"]
        return Sample(id=None, data=example, correct_candidate=label, candidates=[0, 1])

    def get_template(self, template_version=0):
        return {0: EthosTemplate}[template_version]()


class TweetEvalHateDataset(Dataset):
    metric_name = "clf_f1"

    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        d = load_dataset('tweet_eval', 'hate')
        train_d = d["train"]
        validation_d = d["validation"]

        train_samples = [self.build_sample(example) for example in train_d]
        valid_samples = [self.build_sample(example) for example in validation_d]

        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example):
        label = example["label"]
        return Sample(id=None, data=example, correct_candidate=label, candidates=[0, 1])

    def get_template(self, template_version=0):
        return {0: TweetEvalHateTemplate}[template_version]()


class HateSpeech18Dataset(Dataset):
    metric_name = "clf_f1"

    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        d = load_dataset('SetFit/hate_speech18')

        train_samples = [self.build_sample(example) for i, example in enumerate(d["train"]) if i < 9000]
        valid_samples = [self.build_sample(example) for i, example in enumerate(d["train"]) if i > 9000]

        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example):
        label = example["label"]
        return Sample(id=None, data=example, correct_candidate=label, candidates=[0, 1, 2, 3])

    def get_template(self, template_version=0):
        return {0: HateSpeech18Template}[template_version]()


class KiltFeverDataset(Dataset):
    metric_name = "clf_f1"

    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        d = load_dataset('kilt_tasks', 'fever')
        train_d = d["train"]
        validation_d = d["validation"]

        train_samples = [self.build_sample(example) for example in train_d]
        valid_samples = [self.build_sample(example) for example in validation_d]

        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example):
        label = 1 if example["output"][0]["answer"] == "SUPPORTS" else 0
        return Sample(id=None, data=example, correct_candidate=label, candidates=[0, 1])

    def get_template(self, template_version=0):
        return {0: KiltFeverTemplate}[template_version]()


class LiarDataset(Dataset):
    metric_name = "clf_f1"

    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        d = load_dataset('chengxuphd/liar2')
        train_d = d["train"]
        validation_d = d["validation"]

        train_samples = [self.build_sample(example) for example in train_d]
        valid_samples = [self.build_sample(example) for example in validation_d]

        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example):
        label = example["label"]
        return Sample(id=None, data=example, correct_candidate=label, candidates=[0, 1, 2, 3, 4, 5])

    def get_template(self, template_version=0):
        return {0: LiarTemplate, 1: LiarTemplateJustified}[template_version]()


class HealthFactsDataset(Dataset):
    metric_name = "clf_f1"

    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        d = load_dataset('ImperialCollegeLondon/health_fact')
        train_d = d["train"]
        validation_d = d["validation"]

        train_samples = [self.build_sample(example) for example in train_d]
        valid_samples = [self.build_sample(example) for example in validation_d]

        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example):
        # true, false, unproven, mixed
        label = 0 if example["label"] == "false" else 1 if example["label"] == "true" else 2 if example[
                                                                                                    "label"] == "unproven" else 3
        return Sample(id=None, data=example, correct_candidate=label, candidates=[0, 1, 2, 3])

    def get_template(self, template_version=0):
        return {0: HealthFactsTemplate, 1: HealthFactsEvidenceTemplate}[template_version]()


class ClimateFeverDataset(Dataset):
    metric_name = "clf_f1"

    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        d = load_dataset('amandakonet/climate_fever_adopted')
        train_d = d["train"]
        validation_d = d["valid"]

        train_samples = [self.build_sample(example) for example in train_d]
        valid_samples = [self.build_sample(example) for example in validation_d]

        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example):
        label = 0 if example["label"] == "entailment" else 1 if example["label"] == "neutral" else 2
        return Sample(id=None, data=example, correct_candidate=label, candidates=[0, 1, 2])

    def get_template(self, template_version=0):
        return {0: ClimateFeverTemplate, 1: ClimateFeverEvidenceTemplate}[template_version]()


class TabFactDataset(Dataset):
    metric_name = "clf_f1"

    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        d = load_dataset('tab_fact', 'tab_fact')
        train_d = d["train"]
        validation_d = d["validation"]

        train_samples = [self.build_sample(example) for example in train_d]
        valid_samples = [self.build_sample(example) for example in validation_d]

        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example):
        label = example["label"]
        return Sample(id=None, data=example, correct_candidate=label, candidates=[0, 1])

    def get_template(self, template_version=0):
        return {0: TabFactTemplate}[template_version]()


class TweetEvalStanceFeminismDataset(Dataset):
    metric_name = "clf_f1"

    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        d = load_dataset('tweet_eval', 'stance_feminist')
        train_d = ConcatDataset([d["train"], d["validation"]])
        validation_d = d["test"]

        train_samples = [self.build_sample(example) for example in train_d]
        valid_samples = [self.build_sample(example) for example in validation_d]

        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example):
        label = example["label"]
        return Sample(id=None, data=example, correct_candidate=label, candidates=[0, 1, 2])

    def get_template(self, template_version=0):
        return {0: TweetEvalStanceFeminismTemplate}[template_version]()


class TweetEvalStanceAtheismDataset(Dataset):
    metric_name = "clf_f1"

    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        d = load_dataset('tweet_eval', 'stance_atheism')
        train_d = ConcatDataset([d["train"], d["validation"]])
        validation_d = d["test"]

        train_samples = [self.build_sample(example) for example in train_d]
        valid_samples = [self.build_sample(example) for example in validation_d]

        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example):
        label = example["label"]
        return Sample(id=None, data=example, correct_candidate=label, candidates=[0, 1, 2])

    def get_template(self, template_version=0):
        return {0: TweetEvalStanceAtheismTemplate}[template_version]()


class WikiQADataset(Dataset):
    metric_name = "clf_f1"

    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        d = load_dataset('wiki_qa')
        train_d = d["train"]
        validation_d = d["validation"]

        train_samples = [self.build_sample(example) for example in train_d]
        valid_samples = [self.build_sample(example) for example in validation_d]

        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example):
        label = example["label"]
        return Sample(id=None, data=example, correct_candidate=label, candidates=[0, 1])

    def get_template(self, template_version=0):
        return {0: WikiQATemplate}[template_version]()


class MetaICLDataset(Dataset):

    train_sep = "\n\n\n"
    metric_name = "clf_f1"

    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.data_dir = kwargs.get("data_dir", None)
        assert self.data_dir is not None, "data_dir must be provided for MetaICLDataset"
        self.meta_k = kwargs.get("meta_k", None)
        assert self.meta_k is not None, "meta_k must be provided for MetaICLDataset"
        self.meta_test_k = kwargs.get("meta_test_k", 4)
        self.meta_icl_seed = kwargs.get("meta_icl_seed", 100)
        self.method = kwargs.get("method", "direct")

        tasks_py_dir = os.path.dirname(os.path.abspath(__file__))
        with open(f"{tasks_py_dir}/MetaICL/config/tasks/{subtask}.json") as f:
            d = json.load(f)

        self.generation = d["task_type"] == "free-form"

        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        import json
        dataset_path = os.path.join(self.data_dir, path)
        train_file_path = os.path.join(dataset_path, f"{path}_{self.meta_k}_{self.meta_icl_seed}_train.jsonl")
        with open(train_file_path, "r") as f:
            train_d = [json.loads(line) for line in f.readlines()]

        valid_file_path = os.path.join(dataset_path, f"{path}_{self.meta_k}_{self.meta_icl_seed}_test.jsonl")
        if os.path.exists(valid_file_path):
            with open(valid_file_path, "r") as f:
                validation_d = [json.loads(line) for line in f.readlines()]
        else:
            validation_d = []
        # train_samples = [self.build_sample(example) for example in train_d]
        # valid_samples = [self.build_sample(example) for example in validation_d]

        self.samples = {"train": train_d, "valid": validation_d}

    def build_sample(self, example):
        label = example["output"]
        if "options" in example:
            candidates = example["options"]
            if len(candidates) == 0:
                candidates = None
        else:
            candidates = None

        return Sample(id=None, data=example, correct_candidate=label, candidates=candidates)

    def get_template(self, template_version=0):
        return {0: DirectMetaICLTemplate, 1: ChannelMetaICLTemplate}[template_version]()