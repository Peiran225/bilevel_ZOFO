import logging
from typing import Union, List

from transformers.pytorch_utils import Conv1D
from transformers.models.gpt2 import GPT2Model

from torch.nn.modules.module import T

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import torch
from torch import nn
from torch.nn import functional as F
import math

from ..utils import BILEVEL_ACTIVE_LEVEL


def find_module(root_module: nn.Module, key: str):
    """
    Find a module with a specific name in a Transformer model
    From OpenDelta https://github.com/thunlp/OpenDelta
    """
    sub_keys = key.split(".")
    parent_module = root_module
    for sub_key in sub_keys[:-1]:
        parent_module = getattr(parent_module, sub_key)
    module = getattr(parent_module, sub_keys[-1])
    return parent_module, sub_keys[-1], module


class LoRALinear(nn.Linear):
    """
    LoRA implemented in a dense layer
    From https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
    """

    def __init__(
            self,
            in_features: int,
            out_features: int,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.,
            fan_in_fan_out: bool = False,
            # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
            merge_weights: bool = False,
            # Not sure if this will affect saving/loading models so just set it to be False
            **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)

        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights
        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                result += (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0,
                                                                                                      1)) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)


class LoRAConv1D(Conv1D):
    """
    LoRA implemented in a Conv1D layer for GPT-style models.
    """

    def __init__(
            self,
            nf,  # number of output features
            nx,  # number of input features
            r: int = 0,  # Rank for LoRA
            lora_alpha: int = 1,
            lora_dropout: float = 0.,
            merge_weights: bool = False,
    ):
        super().__init__(nf, nx)

        self.r = r
        self.lora_alpha = lora_alpha
        self.merged = False
        self.merge_weights = merge_weights

        if r > 0:
            # LoRA parameters
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, nx)))  # A: (r, nx)
            self.lora_B = nn.Parameter(self.weight.new_zeros((nf, r)))  # B: (nf, r)

            self.scaling = self.lora_alpha / self.r
            self.weight.requires_grad = False

            if lora_dropout > 0.:
                self.lora_dropout = nn.Dropout(p=lora_dropout)
            else:
                self.lora_dropout = lambda x: x

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight, std=0.02)
        if hasattr(self, 'lora_A'):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        Conv1D.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                if self.r > 0:
                    self.weight.data -= (self.lora_B @ self.lora_A).view_as(self.weight) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                if self.r > 0:
                    self.weight.data += (self.lora_B @ self.lora_A).view_as(self.weight) * self.scaling
                self.merged = True

    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            result = super().forward(x)
            result += (self.lora_dropout(x) @ self.lora_A.t() @ self.lora_B.t()) * self.scaling
            return result
        else:
            return super().forward(x)


class BilevelLoRALinear(nn.Linear):
    """
    Bilevel LoRA implemented in a dense layer
    From https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
    """

    def __init__(
            self,
            in_features: int,
            out_features: int,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.,
            fan_in_fan_out: bool = False,
            # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
            merge_weights: bool = False,
            # Not sure if this will affect saving/loading models so just set it to be False
            **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)

        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights
        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lower_level_model_lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lower_level_model_lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))

            self.upper_level_model_lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.upper_level_model_lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))

            self.active_level = BILEVEL_ACTIVE_LEVEL.LOWER

            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                if self.active_level == BILEVEL_ACTIVE_LEVEL.LOWER:
                    result += (self.lora_dropout(x) @ self.lower_level_model_lora_A.transpose(0, 1) @
                               self.lower_level_model_lora_B.transpose(0, 1)) * self.scaling
                else:
                    result += (self.lora_dropout(x) @ self.upper_level_model_lora_A.transpose(0, 1) @
                               self.upper_level_model_lora_B.transpose(0, 1)) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)

    def set_bilevel_active_level(self, active_level):
        self.active_level = active_level
        if self.active_level == BILEVEL_ACTIVE_LEVEL.LOWER:
            # set the all parameters of the upper level to be not requires_grad
            for n, p in self.named_parameters():
                if "upper_level_model" in n:
                    p.requires_grad = False
                if "lower_level_model" in n:
                    p.requires_grad = True
        else:
            # set the all parameters of the lower level to be not requires_grad
            for n, p in self.named_parameters():
                if "lower_level_model" in n:
                    p.requires_grad = False
                if "upper_level_model" in n:
                    p.requires_grad = True


class BilevelLoRAConv1D(Conv1D):
    """
    Bilevel LoRA implemented in a Conv1D layer for GPT-style models.
    """

    def __init__(
            self,
            nf,  # number of output features
            nx,  # number of input features
            r: int = 0,  # Rank for LoRA
            lora_alpha: int = 1,
            lora_dropout: float = 0.,
            merge_weights: bool = False,
    ):
        super().__init__(nf, nx)

        self.r = r
        self.lora_alpha = lora_alpha
        self.merged = False
        self.merge_weights = merge_weights

        if r > 0:
            # LoRA parameters for lower level model
            self.lower_level_model_lora_A = nn.Parameter(self.weight.new_zeros((r, nx)))  # A: (r, nx)
            self.lower_level_model_lora_B = nn.Parameter(self.weight.new_zeros((nf, r)))  # B: (nf, r)

            # LoRA parameters for upper level model
            self.upper_level_model_lora_A = nn.Parameter(self.weight.new_zeros((r, nx)))  # A: (r, nx)
            self.upper_level_model_lora_B = nn.Parameter(self.weight.new_zeros((nf, r)))  # B: (nf, r)

            self.active_level = BILEVEL_ACTIVE_LEVEL.LOWER

            self.scaling = self.lora_alpha / self.r
            self.weight.requires_grad = False

            if lora_dropout > 0.:
                self.lora_dropout = nn.Dropout(p=lora_dropout)
            else:
                self.lora_dropout = lambda x: x

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight, std=0.02)
        if hasattr(self, 'lower_level_model_lora_A'):
            nn.init.kaiming_uniform_(self.lower_level_model_lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lower_level_model_lora_B)

            nn.init.kaiming_uniform_(self.upper_level_model_lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.upper_level_model_lora_B)

    def train(self, mode: bool = True):
        Conv1D.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                if self.r > 0:
                    self.weight.data -= (self.lower_level_model_lora_B @ self.lower_level_model_lora_A).view_as(
                        self.weight) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                if self.r > 0:
                    self.weight.data += (self.lower_level_model_lora_B @ self.lower_level_model_lora_A).view_as(
                        self.weight) * self.scaling
                self.merged = True

    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            result = super().forward(x)
            if self.active_level == BILEVEL_ACTIVE_LEVEL.LOWER:
                result += (self.lora_dropout(
                    x) @ self.lower_level_model_lora_A.t() @ self.lower_level_model_lora_B.t()) * self.scaling
            else:
                result += (self.lora_dropout(
                    x) @ self.upper_level_model_lora_A.t() @ self.upper_level_model_lora_B.t()) * self.scaling
            return result
        else:
            return super().forward(x)

    def set_bilevel_active_level(self, active_level):
        self.active_level = active_level
        if self.active_level == BILEVEL_ACTIVE_LEVEL.LOWER:
            for n, p in self.named_parameters():
                if "upper_level_model" in n:
                    p.requires_grad = False
                if "lower_level_model" in n:
                    p.requires_grad = True
        else:
            for n, p in self.named_parameters():
                if "lower_level_model" in n:
                    p.requires_grad = False
                if "upper_level_model" in n:
                    p.requires_grad = True


class MultiTaskLoRALinear(nn.Linear):
    """
    LoRA implemented in a dense layer
    From https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
    """

    def __init__(
            self,
            tasks,
            in_features: int,
            out_features: int,
            lora_ranks: Union[int, List[int]] = 0,
            lora_alphas: Union[int, List[int]] = 1,
            lora_dropouts: Union[float, List[float]] = 0.,
            fan_in_fan_out: bool = False,
            # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
            merge_weights: bool = False,
            # Not sure if this will affect saving/loading models so just set it to be False
            **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        tasks = [str(task) for task in tasks]
        self.tasks = tasks

        if isinstance(lora_ranks, int):
            lora_ranks = [lora_ranks] * len(tasks)
        if isinstance(lora_alphas, int):
            lora_alphas = [lora_alphas] * len(tasks)
        if isinstance(lora_dropouts, float):
            lora_dropouts = [lora_dropouts] * len(tasks)

        self.lora_ranks = {task: r for task, r in zip(tasks, lora_ranks)}
        # self.lora_alphas = lora_alphas
        # self.lora_dropouts = lora_dropouts

        self.lora_dropouts = nn.ModuleDict()

        # Optional dropout
        for task, lora_dropout in zip(tasks, lora_dropouts):
            if lora_dropout > 0.:
                self.lora_dropouts[task] = nn.Dropout(p=lora_dropout)
            else:
                self.lora_dropouts[task] = nn.Identity()

        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights
        self.fan_in_fan_out = fan_in_fan_out

        # Actual trainable parameters
        self.lora_A = nn.ParameterDict()
        self.lora_B = nn.ParameterDict()
        self.scaling = dict()
        for i, (r, lora_alpha) in enumerate(zip(lora_ranks, lora_alphas)):
            task = self.tasks[i]
            if r > 0:
                self.lora_A[task] = nn.Parameter(self.weight.new_zeros((r, in_features)))
                self.lora_B[task] = nn.Parameter(self.weight.new_zeros((out_features, r)))
                self.scaling[task] = lora_alpha / r
                # Freezing the pre-trained weight matrix
                self.weight.requires_grad = False

        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

        self.active_lora = self.tasks[0]
        self.set_active_lora(self.tasks[0])

    def set_active_lora(self, task):
        self.active_lora = task
        # set the active-lora parameters to be trainable
        if self.lora_ranks[task] > 0:
            self.lora_A[task].requires_grad = True
            self.lora_B[task].requires_grad = True
        # set the other lora parameters to be non-trainable
        for t in self.tasks:
            if t != task:
                if self.lora_ranks[t] > 0:
                    self.lora_A[t].requires_grad = False
                    self.lora_B[t].requires_grad = False

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            for task in self.tasks:
                self.lora_A[str(task)].data = nn.init.kaiming_uniform_(self.lora_A[str(task)], a=math.sqrt(5))
                self.lora_B[str(task)].data = nn.init.zeros_(self.lora_B[str(task)])

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.lora_ranks[self.active_lora] > 0:
                    self.weight.data -= T(self.lora_B[self.active_lora] @ self.lora_A[self.active_lora]) * self.scaling[
                        self.active_lora]
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.lora_ranks[self.active_lora] > 0:
                    self.weight.data += T(self.lora_B[self.active_lora] @ self.lora_A[self.active_lora]) * self.scaling[
                        self.active_lora]
                self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        if self.lora_ranks[self.active_lora] > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.lora_ranks[self.active_lora]:
                result += (self.lora_dropouts[self.active_lora](x) @ self.lora_A[self.active_lora].transpose(0, 1) @
                           self.lora_B[self.active_lora].transpose(0,
                                                                   1)) * self.scaling[self.active_lora]
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)


class MultiTaskLoRAConv1D(Conv1D):
    """
    Multi-task LoRA implemented in a Conv1D layer for GPT-style models.
    """

    def __init__(
            self,
            tasks,
            nf,  # number of output features
            nx,  # number of input features
            lora_ranks: Union[int, List[int]] = 0,
            lora_alphas: Union[int, List[int]] = 1,
            lora_dropouts: Union[float, List[float]] = 0.,
            merge_weights: bool = False,
            **kwargs
    ):
        super().__init__(nf, nx)  # Match Conv1D initialization

        tasks = [str(task) for task in tasks]
        self.tasks = tasks

        if isinstance(lora_ranks, int):
            lora_ranks = [lora_ranks] * len(tasks)
        if isinstance(lora_alphas, int):
            lora_alphas = [lora_alphas] * len(tasks)
        if isinstance(lora_dropouts, float):
            lora_dropouts = [lora_dropouts] * len(tasks)

        self.lora_ranks = {task: r for task, r in zip(tasks, lora_ranks)}
        self.lora_dropouts = nn.ModuleDict()

        # Optional dropout
        for task, lora_dropout in zip(tasks, lora_dropouts):
            if lora_dropout > 0.:
                self.lora_dropouts[task] = nn.Dropout(p=lora_dropout)
            else:
                self.lora_dropouts[task] = nn.Identity()

        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights

        # Trainable LoRA parameters
        self.lora_A = nn.ParameterDict()
        self.lora_B = nn.ParameterDict()
        self.scaling = dict()

        for i, (r, lora_alpha) in enumerate(zip(lora_ranks, lora_alphas)):
            task = self.tasks[i]
            if r > 0:
                self.lora_A[task] = nn.Parameter(self.weight.new_zeros((r, nx)))  # A: (r, in_features)
                self.lora_B[task] = nn.Parameter(self.weight.new_zeros((nf, r)))  # B: (out_features, r)
                self.scaling[task] = lora_alpha / r
                # Freeze the pre-trained weight matrix
                self.weight.requires_grad = False

        self.reset_parameters()

        self.active_lora = self.tasks[0]
        self.set_active_lora(self.tasks[0])

    def set_active_lora(self, task):
        self.active_lora = task
        if self.lora_ranks[task] > 0:
            self.lora_A[task].requires_grad = True
            self.lora_B[task].requires_grad = True
        # Set other LoRA parameters to non-trainable
        for t in self.tasks:
            if t != task:
                if self.lora_ranks[t] > 0:
                    self.lora_A[t].requires_grad = False
                    self.lora_B[t].requires_grad = False

    def reset_parameters(self):
        super().reset_parameters()
        if hasattr(self, 'lora_A'):
            # Initialize A and B matrices
            for task in self.tasks:
                self.lora_A[str(task)].data = nn.init.kaiming_uniform_(self.lora_A[str(task)], a=math.sqrt(5))
                self.lora_B[str(task)].data = nn.init.zeros_(self.lora_B[str(task)])

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            if self.merge_weights and self.merged:
                if self.lora_ranks[self.active_lora] > 0:
                    self.weight.data -= (self.lora_B[self.active_lora] @ self.lora_A[self.active_lora]).view_as(
                        self.weight) * self.scaling[self.active_lora]
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                if self.lora_ranks[self.active_lora] > 0:
                    self.weight.data += (self.lora_B[self.active_lora] @ self.lora_A[self.active_lora]).view_as(
                        self.weight) * self.scaling[self.active_lora]
                self.merged = True

    def forward(self, x: torch.Tensor):
        if self.lora_ranks[self.active_lora] > 0 and not self.merged:
            result = super().forward(x)
            if self.lora_ranks[self.active_lora]:
                result += (self.lora_dropouts[self.active_lora](x) @ self.lora_A[self.active_lora].t() @
                           self.lora_B[self.active_lora].t()) * self.scaling[self.active_lora]
            return result
        else:
            return super().forward(x)


class MultiTaskBilevelLoRALinear(nn.Linear):
    """
    LoRA implemented in a dense layer
    From https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
    """

    def __init__(
            self,
            tasks,
            in_features: int,
            out_features: int,
            lora_ranks: Union[int, List[int]] = 0,
            lora_alphas: Union[int, List[int]] = 1,
            lora_dropouts: Union[float, List[float]] = 0.,
            fan_in_fan_out: bool = False,
            # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
            merge_weights: bool = False,
            # Not sure if this will affect saving/loading models so just set it to be False
            **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        tasks = [str(task) for task in tasks]
        self.tasks = tasks

        if isinstance(lora_ranks, int):
            lora_ranks = [lora_ranks] * len(tasks)
        if isinstance(lora_alphas, int):
            lora_alphas = [lora_alphas] * len(tasks)
        if isinstance(lora_dropouts, float):
            lora_dropouts = [lora_dropouts] * len(tasks)

        self.lora_ranks = {task: r for task, r in zip(tasks, lora_ranks)}
        # self.lora_alphas = lora_alphas
        # self.lora_dropouts = lora_dropouts

        self.lora_dropouts = nn.ModuleDict()

        # Optional dropout
        for task, lora_dropout in zip(tasks, lora_dropouts):
            if lora_dropout > 0.:
                self.lora_dropouts[task] = nn.Dropout(p=lora_dropout)
            else:
                self.lora_dropouts[task] = nn.Identity()

        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights
        self.fan_in_fan_out = fan_in_fan_out

        # Actual trainable parameters
        self.lower_level_model_lora_A = nn.ParameterDict()
        self.lower_level_model_lora_B = nn.ParameterDict()

        self.upper_level_model_lora_A = nn.ParameterDict()
        self.upper_level_model_lora_B = nn.ParameterDict()

        self.scaling = dict()
        for i, (r, lora_alpha) in enumerate(zip(lora_ranks, lora_alphas)):
            task = self.tasks[i]
            if r > 0:
                self.lower_level_model_lora_A[task] = nn.Parameter(self.weight.new_zeros((r, in_features)))
                self.lower_level_model_lora_B[task] = nn.Parameter(self.weight.new_zeros((out_features, r)))

                self.upper_level_model_lora_A[task] = nn.Parameter(self.weight.new_zeros((r, in_features)))
                self.upper_level_model_lora_B[task] = nn.Parameter(self.weight.new_zeros((out_features, r)))

                self.scaling[task] = lora_alpha / r
                # Freezing the pre-trained weight matrix
                self.weight.requires_grad = False

                self.active_level = BILEVEL_ACTIVE_LEVEL.LOWER

        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

        self.active_lora = self.tasks[0]
        self.set_active_lora(self.tasks[0])

    def set_active_lora(self, task):
        self.active_lora = task
        # set the active-lora parameters to be trainable
        if self.lora_ranks[task] > 0:
            if self.active_level == BILEVEL_ACTIVE_LEVEL.LOWER:
                self.lower_level_model_lora_A[task].requires_grad = True
                self.lower_level_model_lora_B[task].requires_grad = True

                self.upper_level_model_lora_A[task].requires_grad = False
                self.upper_level_model_lora_B[task].requires_grad = False
            else:
                self.upper_level_model_lora_A[task].requires_grad = True
                self.upper_level_model_lora_B[task].requires_grad = True

                self.lower_level_model_lora_A[task].requires_grad = False
                self.lower_level_model_lora_B[task].requires_grad = False

        # set the other lora parameters to be non-trainable
        for t in self.tasks:
            if t != task:
                if self.lora_ranks[t] > 0:
                    self.lower_level_model_lora_A[t].requires_grad = False
                    self.lower_level_model_lora_B[t].requires_grad = False
                    self.upper_level_model_lora_A[t].requires_grad = False
                    self.upper_level_model_lora_B[t].requires_grad = False

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lower_level_model_lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            for task in self.tasks:
                self.lower_level_model_lora_A[str(task)].data = nn.init.kaiming_uniform_(
                    self.lower_level_model_lora_A[str(task)], a=math.sqrt(5))
                self.lower_level_model_lora_B[str(task)].data = nn.init.zeros_(self.lower_level_model_lora_B[str(task)])

                self.upper_level_model_lora_A[str(task)].data = nn.init.kaiming_uniform_(
                    self.upper_level_model_lora_A[str(task)], a=math.sqrt(5))
                self.upper_level_model_lora_B[str(task)].data = nn.init.zeros_(self.upper_level_model_lora_B[str(task)])

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.lora_ranks[self.active_lora] > 0:
                    if self.active_level == BILEVEL_ACTIVE_LEVEL.LOWER:
                        self.weight.data -= T(self.lower_level_model_lora_B[self.active_lora] @
                                              self.lower_level_model_lora_A[self.active_lora]) * self.scaling[
                                                self.active_lora]
                    else:
                        self.weight.data -= T(self.upper_level_model_lora_B[self.active_lora] @
                                              self.upper_level_model_lora_A[self.active_lora]) * self.scaling[
                                                self.active_lora]
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.lora_ranks[self.active_lora] > 0:
                    if self.active_level == BILEVEL_ACTIVE_LEVEL.LOWER:
                        self.weight.data += T(self.lower_level_model_lora_B[self.active_lora] @
                                              self.lower_level_model_lora_A[self.active_lora]) * self.scaling[
                                                self.active_lora]
                    else:
                        self.weight.data += T(self.upper_level_model_lora_B[self.active_lora] @
                                              self.upper_level_model_lora_A[self.active_lora]) * self.scaling[
                                                self.active_lora]
                self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        if self.lora_ranks[self.active_lora] > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.lora_ranks[self.active_lora]:
                if self.active_level == BILEVEL_ACTIVE_LEVEL.LOWER:
                    result += (self.lora_dropouts[self.active_lora](x) @
                               self.lower_level_model_lora_A[self.active_lora].transpose(0, 1) @
                               self.lower_level_model_lora_B[self.active_lora].transpose(0, 1)) * self.scaling[
                                  self.active_lora]
                else:
                    result += (self.lora_dropouts[self.active_lora](x) @
                               self.upper_level_model_lora_A[self.active_lora].transpose(0, 1) @
                               self.upper_level_model_lora_B[self.active_lora].transpose(0, 1)) * self.scaling[
                                  self.active_lora]
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)

    def set_bilevel_active_level(self, active_level):
        self.active_level = active_level
        for task in self.tasks:
            if self.active_level == BILEVEL_ACTIVE_LEVEL.LOWER:
                self.lower_level_model_lora_A[task].requires_grad = True
                self.lower_level_model_lora_B[task].requires_grad = True

                self.upper_level_model_lora_A[task].requires_grad = False
                self.upper_level_model_lora_B[task].requires_grad = False
            else:
                self.upper_level_model_lora_A[task].requires_grad = True
                self.upper_level_model_lora_B[task].requires_grad = True

                self.lower_level_model_lora_A[task].requires_grad = False
                self.lower_level_model_lora_B[task].requires_grad = False


class MultiTaskBilevelLoRAConv1D(Conv1D):
    """
    LoRA implemented in a Conv1D layer with bilevel support.
    """

    def __init__(
            self,
            tasks,
            nf,  # number of output features
            nx,  # number of input features
            lora_ranks: Union[int, List[int]] = 0,
            lora_alphas: Union[int, List[int]] = 1,
            lora_dropouts: Union[float, List[float]] = 0.,
            merge_weights: bool = False,
            **kwargs
    ):
        super().__init__(nf, nx)  # Match Conv1D initialization
        tasks = [str(task) for task in tasks]
        self.tasks = tasks

        if isinstance(lora_ranks, int):
            lora_ranks = [lora_ranks] * len(tasks)
        if isinstance(lora_alphas, int):
            lora_alphas = [lora_alphas] * len(tasks)
        if isinstance(lora_dropouts, float):
            lora_dropouts = [lora_dropouts] * len(tasks)

        self.lora_ranks = {task: r for task, r in zip(tasks, lora_ranks)}
        self.lora_dropouts = nn.ModuleDict()

        # Optional dropout
        for task, lora_dropout in zip(tasks, lora_dropouts):
            if lora_dropout > 0.:
                self.lora_dropouts[task] = nn.Dropout(p=lora_dropout)
            else:
                self.lora_dropouts[task] = nn.Identity()

        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights

        # Actual trainable parameters
        self.lower_level_model_lora_A = nn.ParameterDict()
        self.lower_level_model_lora_B = nn.ParameterDict()

        self.upper_level_model_lora_A = nn.ParameterDict()
        self.upper_level_model_lora_B = nn.ParameterDict()

        self.scaling = dict()
        for i, (r, lora_alpha) in enumerate(zip(lora_ranks, lora_alphas)):
            task = self.tasks[i]
            if r > 0:
                self.lower_level_model_lora_A[task] = nn.Parameter(self.weight.new_zeros((r, nx)))
                self.lower_level_model_lora_B[task] = nn.Parameter(self.weight.new_zeros((nf, r)))

                self.upper_level_model_lora_A[task] = nn.Parameter(self.weight.new_zeros((r, nx)))
                self.upper_level_model_lora_B[task] = nn.Parameter(self.weight.new_zeros((nf, r)))

                self.scaling[task] = lora_alpha / r
                self.weight.requires_grad = False

                self.active_level = BILEVEL_ACTIVE_LEVEL.LOWER

        self.reset_parameters()
        self.active_lora = self.tasks[0]
        self.set_active_lora(self.tasks[0])

    def set_active_lora(self, task):
        self.active_lora = task
        if self.lora_ranks[task] > 0:
            if self.active_level == BILEVEL_ACTIVE_LEVEL.LOWER:
                self.lower_level_model_lora_A[task].requires_grad = True
                self.lower_level_model_lora_B[task].requires_grad = True

                self.upper_level_model_lora_A[task].requires_grad = False
                self.upper_level_model_lora_B[task].requires_grad = False
            else:
                self.upper_level_model_lora_A[task].requires_grad = True
                self.upper_level_model_lora_B[task].requires_grad = True

                self.lower_level_model_lora_A[task].requires_grad = False
                self.lower_level_model_lora_B[task].requires_grad = False

        for t in self.tasks:
            if t != task:
                if self.lora_ranks[t] > 0:
                    self.lower_level_model_lora_A[t].requires_grad = False
                    self.lower_level_model_lora_B[t].requires_grad = False
                    self.upper_level_model_lora_A[t].requires_grad = False
                    self.upper_level_model_lora_B[t].requires_grad = False

    def reset_parameters(self):
        nn.init.normal_(self.weight, std=0.02)
        if hasattr(self, 'lower_level_model_lora_A'):
            for task in self.tasks:
                self.lower_level_model_lora_A[str(task)].data = nn.init.kaiming_uniform_(
                    self.lower_level_model_lora_A[str(task)], a=math.sqrt(5))
                self.lower_level_model_lora_B[str(task)].data = nn.init.zeros_(self.lower_level_model_lora_B[str(task)])

                self.upper_level_model_lora_A[str(task)].data = nn.init.kaiming_uniform_(
                    self.upper_level_model_lora_A[str(task)], a=math.sqrt(5))
                self.upper_level_model_lora_B[str(task)].data = nn.init.zeros_(self.upper_level_model_lora_B[str(task)])

    def train(self, mode: bool = True):
        Conv1D.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                if self.lora_ranks[self.active_lora] > 0:
                    if self.active_level == BILEVEL_ACTIVE_LEVEL.LOWER:
                        self.weight.data -= (self.lower_level_model_lora_B[self.active_lora] @
                                             self.lower_level_model_lora_A[self.active_lora]) * self.scaling[
                                                self.active_lora]
                    else:
                        self.weight.data -= (self.upper_level_model_lora_B[self.active_lora] @
                                             self.upper_level_model_lora_A[self.active_lora]) * self.scaling[
                                                self.active_lora]
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                if self.lora_ranks[self.active_lora] > 0:
                    if self.active_level == BILEVEL_ACTIVE_LEVEL.LOWER:
                        self.weight.data += (self.lower_level_model_lora_B[self.active_lora] @
                                             self.lower_level_model_lora_A[self.active_lora]) * self.scaling[
                                                self.active_lora]
                    else:
                        self.weight.data += (self.upper_level_model_lora_B[self.active_lora] @
                                             self.upper_level_model_lora_A[self.active_lora]) * self.scaling[
                                                self.active_lora]
                self.merged = True

    def forward(self, x: torch.Tensor):
        if self.lora_ranks[self.active_lora] > 0 and not self.merged:
            result = super().forward(x)
            if self.lora_ranks[self.active_lora]:
                if self.active_level == BILEVEL_ACTIVE_LEVEL.LOWER:
                    result += (self.lora_dropouts[self.active_lora](x) @
                               self.lower_level_model_lora_A[self.active_lora].t() @
                               self.lower_level_model_lora_B[self.active_lora].t()) * self.scaling[self.active_lora]
                else:
                    result += (self.lora_dropouts[self.active_lora](x) @
                               self.upper_level_model_lora_A[self.active_lora].t() @
                               self.upper_level_model_lora_B[self.active_lora].t()) * self.scaling[self.active_lora]
            return result
        else:
            return super().forward(x)

    def set_bilevel_active_level(self, active_level):
        self.active_level = active_level
        for task in self.tasks:
            if self.active_level == BILEVEL_ACTIVE_LEVEL.LOWER:
                self.lower_level_model_lora_A[task].requires_grad = True
                self.lower_level_model_lora_B[task].requires_grad = True

                self.upper_level_model_lora_A[task].requires_grad = False
                self.upper_level_model_lora_B[task].requires_grad = False
            else:
                self.upper_level_model_lora_A[task].requires_grad = True
                self.upper_level_model_lora_B[task].requires_grad = True

                self.lower_level_model_lora_A[task].requires_grad = False
                self.lower_level_model_lora_B[task].requires_grad = False


class LoRA:

    def __init__(self, model, r, alpha, float16, tasks=None):
        """
        Input:
        r, alpha: LoRA hyperparameters
        float16: Whether the model parameters are float16 or not
        """
        self.model = model
        self.hidden_dim = model.config.hidden_size
        self.float16 = float16

        if model.config.model_type == "opt":
            attention_name = "attn"
        elif model.config.model_type == "roberta":
            attention_name = "attention"
        elif model.config.model_type in ["llama", "mistral"]:
            attention_name = "self_attn"
        elif model.config.model_type == "gpt2":
            attention_name = "attn"
        else:
            raise NotImplementedError

        self.attention_name = attention_name

        # Insert LoRA
        for key, _ in model.named_modules():
            if key[-len(attention_name):] == attention_name:
                logger.info(f"Inject lora to: {key}")
                _, _, attn = find_module(model, key)

                if model.config.model_type == "opt":
                    original_q_weight = attn.q_proj.weight.data
                    original_q_bias = attn.q_proj.bias.data
                    original_v_weight = attn.v_proj.weight.data
                    original_v_bias = attn.v_proj.bias.data
                    if tasks is None:
                        attn.q_proj = LoRALinear(model.config.hidden_size, model.config.hidden_size, r=r,
                                                 lora_alpha=alpha,
                                                 bias=model.config.enable_bias).to(original_q_weight.device)
                        attn.v_proj = LoRALinear(model.config.hidden_size, model.config.hidden_size, r=r,
                                                 lora_alpha=alpha,
                                                 bias=model.config.enable_bias).to(original_v_weight.device)
                    else:
                        attn.q_proj = MultiTaskLoRALinear(tasks, model.config.hidden_size, model.config.hidden_size,
                                                          lora_ranks=r, lora_alphas=alpha,
                                                          bias=model.config.enable_bias).to(
                            original_q_weight.device)
                        attn.v_proj = MultiTaskLoRALinear(tasks, model.config.hidden_size, model.config.hidden_size,
                                                          lora_ranks=r, lora_alphas=alpha,
                                                          bias=model.config.enable_bias).to(
                            original_v_weight.device)
                    if float16:
                        attn.q_proj.half()
                        attn.v_proj.half()
                    attn.q_proj.weight.data = original_q_weight
                    attn.q_proj.bias.data = original_q_bias
                    attn.v_proj.weight.data = original_v_weight
                    attn.v_proj.bias.data = original_v_bias
                elif model.config.model_type == "llama":
                    # in early version of transformers, llama attention bias is hard coded to False
                    attention_bias = False if not hasattr(model.config,
                                                          "attention_bias") else model.config.attention_bias
                    original_q_weight = attn.q_proj.weight.data
                    original_v_weight = attn.v_proj.weight.data
                    original_q_bias = attn.q_proj.bias.data if attention_bias else None
                    original_v_bias = attn.v_proj.bias.data if attention_bias else None
                    if tasks is None:
                        attn.q_proj = LoRALinear(
                            model.config.hidden_size,
                            model.config.hidden_size,
                            r=r, lora_alpha=alpha, bias=attention_bias
                        ).to(original_q_weight.device)
                        attn.v_proj = LoRALinear(
                            model.config.hidden_size,
                            model.config.hidden_size,
                            r=r, lora_alpha=alpha, bias=attention_bias
                        ).to(original_v_weight.device)
                    else:
                        attn.q_proj = MultiTaskLoRALinear(
                            tasks,
                            model.config.hidden_size,
                            model.config.hidden_size,
                            r=r, lora_alphas=alpha, bias=attention_bias
                        ).to(original_q_weight.device)
                        attn.v_proj = MultiTaskLoRALinear(
                            tasks,
                            model.config.hidden_size,
                            model.config.hidden_size,
                            r=r, lora_alphas=alpha, bias=attention_bias
                        ).to(original_v_weight.device)
                    if float16:
                        attn.q_proj.half()
                        attn.v_proj.half()
                    attn.q_proj.weight.data = original_q_weight
                    attn.v_proj.weight.data = original_v_weight
                    if attention_bias:
                        attn.q_proj.bias.data = original_q_bias
                        attn.v_proj.bias.data = original_v_bias
                elif model.config.model_type == "mistral":
                    # in early version of transformers, llama attention bias is hard coded to False
                    config = model.config
                    original_q_weight = attn.q_proj.weight.data
                    original_v_weight = attn.v_proj.weight.data
                    head_dim = config.hidden_size // config.num_attention_heads
                    if tasks is None:
                        attn.q_proj = LoRALinear(
                            config.hidden_size,
                            config.hidden_size,
                            r=r, lora_alpha=alpha
                        ).to(original_q_weight.device)
                        attn.v_proj = LoRALinear(
                            config.hidden_size,
                            config.num_key_value_heads * head_dim,
                            r=r, lora_alpha=alpha
                        ).to(original_v_weight.device)
                    else:
                        attn.q_proj = MultiTaskLoRALinear(
                            tasks,
                            config.hidden_size,
                            config.hidden_size,
                            lora_ranks=r, lora_alphas=alpha
                        ).to(original_q_weight.device)
                        attn.v_proj = MultiTaskLoRALinear(
                            tasks,
                            config.hidden_size,
                            config.num_key_value_heads * head_dim,
                            lora_ranks=r, lora_alphas=alpha
                        ).to(original_v_weight.device)
                    if float16:
                        attn.q_proj.half()
                        attn.v_proj.half()
                    attn.q_proj.weight.data = original_q_weight
                    attn.v_proj.weight.data = original_v_weight
                elif model.config.model_type == "gpt2":
                    original_c_attn_weight = attn.c_attn.weight.data
                    original_c_attn_bias = attn.c_attn.bias.data
                    original_c_proj_weight = attn.c_proj.weight.data
                    original_c_prof_bias = attn.c_proj.bias.data
                    if tasks is None:
                        attn.c_attn = LoRAConv1D(model.config.hidden_size * 3, model.config.hidden_size, r=r,
                                                        lora_alpha=alpha, ).to(original_c_attn_weight.device)
                        attn.c_proj = LoRAConv1D(model.config.hidden_size, model.config.hidden_size, r=r,
                                                        lora_alpha=alpha).to(original_c_proj_weight.device)
                    else:
                        attn.c_attn = MultiTaskLoRAConv1D(tasks, model.config.hidden_size * 3,
                                                                 model.config.hidden_size, lora_ranks=r,
                                                                 lora_alphas=alpha).to(original_c_attn_weight.device)
                        attn.c_proj = MultiTaskLoRAConv1D(tasks, model.config.hidden_size,
                                                                 model.config.hidden_size, lora_ranks=r,
                                                                 lora_alphas=alpha).to(original_c_proj_weight.device)
                    if float16:
                        attn.c_attn.half()
                        attn.c_proj.half()
                    attn.c_attn.weight.data = original_c_attn_weight
                    attn.c_attn.bias.data = original_c_attn_bias
                    attn.c_proj.weight.data = original_c_proj_weight
                    attn.c_proj.bias.data = original_c_prof_bias
                else:
                    raise NotImplementedError

        # Freeze non-LoRA parameters
        for n, p in model.named_parameters():
            if "lora" not in n:
                p.requires_grad = False

    def remove_loras(self, model):

        for key, _ in model.named_modules():
            if key[-len(self.attention_name):] == self.attention_name and "c_attn" not in key:
                logger.info(f"Removing lora from: {key}")
                _, _, attn = find_module(model, key)

                if model.config.model_type == "opt":
                    original_q_weight = attn.q_proj.weight.data
                    original_q_bias = attn.q_proj.bias.data
                    original_v_weight = attn.v_proj.weight.data
                    original_v_bias = attn.v_proj.bias.data
                    attn.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=model.config.enable_bias).to(
                        original_q_weight.device)
                    attn.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=model.config.enable_bias).to(
                        original_v_weight.device)
                    if self.float16:
                        attn.q_proj.half()
                        attn.v_proj.half()
                    attn.q_proj.weight.data = original_q_weight
                    attn.q_proj.bias.data = original_q_bias
                    attn.v_proj.weight.data = original_v_weight
                    attn.v_proj.bias.data = original_v_bias
                elif model.config.model_type == "llama":
                    attention_bias = False if not hasattr(model.config,
                                                          "attention_bias") else model.config.attention_bias
                    original_q_weight = attn.q_proj.weight.data
                    original_v_weight = attn.v_proj.weight.data
                    original_q_bias = attn.q_proj.bias.data if attention_bias else None
                    original_v_bias = attn.v_proj.bias.data if attention_bias else None
                    attn.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=attention_bias).to(
                        original_q_weight.device)
                    attn.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=attention_bias).to(
                        original_v_weight.device)
                    if self.float16:
                        attn.q_proj.half()
                        attn.v_proj.half()
                    attn.q_proj.weight.data = original_q_weight
                    attn.v_proj.weight.data = original_v_weight
                    if attention_bias:
                        attn.q_proj.bias.data = original_q_bias
                        attn.v_proj.bias.data = original_v_bias
                elif model.config.model_type == "mistral":
                    config = model.config
                    original_q_weight = attn.q_proj.weight.data
                    original_v_weight = attn.v_proj.weight.data
                    head_dim = config.hidden_size // config.num_attention_heads
                    attn.q_proj = nn.Linear(config.hidden_size, config.hidden_size).to(original_q_weight.device)
                    attn.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * head_dim).to(
                        original_v_weight.device)
                    if self.float16:
                        attn.q_proj.half()
                        attn.v_proj.half()
                    attn.q_proj.weight.data = original_q_weight
                    attn.v_proj.weight.data = original_v_weight
                elif model.config.model_type == "gpt2":
                    original_c_attn_weight = attn.c_attn.weight.data
                    original_c_attn_bias = attn.c_attn.bias.data
                    original_c_proj_weight = attn.c_proj.weight.data
                    original_c_prof_bias = attn.c_proj.bias.data
                    attn.c_attn = Conv1D(model.config.hidden_size * 3, model.config.hidden_size).to(original_c_attn_weight.device)
                    attn.c_proj = Conv1D(model.config.hidden_size, model.config.hidden_size).to(original_c_proj_weight.device)
                    if self.float16:
                        attn.c_attn.half()
                        attn.c_proj.half()
                    attn.c_attn.weight.data = original_c_attn_weight
                    attn.c_attn.bias.data = original_c_attn_bias
                    attn.c_proj.weight.data = original_c_proj_weight
                    attn.c_proj.bias.data = original_c_prof_bias
                else:
                    raise NotImplementedError

        return model


class BilevelLoRA:
    def __init__(self, model, r, alpha, float16, tasks=None):
        """
        Input:
        r, alpha: LoRA hyperparameters
        float16: Whether the model parameters are float16 or not
        """
        self.model = model
        self.hidden_dim = model.config.hidden_size
        self.float16 = float16

        if model.config.model_type == "opt":
            attention_name = "attn"
        elif model.config.model_type == "roberta":
            attention_name = "attention"
        elif model.config.model_type in ["llama", "mistral"]:
            attention_name = "self_attn"
        elif model.config.model_type == "gpt2":
            attention_name = "attn"
        else:
            raise NotImplementedError

        self.attention_name = attention_name

        # Insert LoRA
        for key, _ in model.named_modules():
            if key[-len(attention_name):] == attention_name and "c_attn" not in key:
                logger.info(f"Inject lora to: {key}")
                _, _, attn = find_module(model, key)

                if model.config.model_type == "opt":
                    original_q_weight = attn.q_proj.weight.data
                    original_q_bias = attn.q_proj.bias.data
                    original_v_weight = attn.v_proj.weight.data
                    original_v_bias = attn.v_proj.bias.data
                    if tasks is None:
                        attn.q_proj = BilevelLoRALinear(model.config.hidden_size, model.config.hidden_size, r=r,
                                                        lora_alpha=alpha,
                                                        bias=model.config.enable_bias).to(original_q_weight.device)
                        attn.v_proj = BilevelLoRALinear(model.config.hidden_size, model.config.hidden_size, r=r,
                                                        lora_alpha=alpha,
                                                        bias=model.config.enable_bias).to(original_v_weight.device)
                    else:
                        attn.q_proj = MultiTaskBilevelLoRALinear(tasks, model.config.hidden_size,
                                                                 model.config.hidden_size,
                                                                 lora_ranks=r, lora_alphas=alpha,
                                                                 bias=model.config.enable_bias).to(
                            original_q_weight.device)
                        attn.v_proj = MultiTaskBilevelLoRALinear(tasks, model.config.hidden_size,
                                                                 model.config.hidden_size,
                                                                 lora_ranks=r, lora_alphas=alpha,
                                                                 bias=model.config.enable_bias).to(
                            original_v_weight.device)
                    if float16:
                        attn.q_proj.half()
                        attn.v_proj.half()
                    attn.q_proj.weight.data = original_q_weight
                    attn.q_proj.bias.data = original_q_bias
                    attn.v_proj.weight.data = original_v_weight
                    attn.v_proj.bias.data = original_v_bias
                elif model.config.model_type == "llama":
                    # in early version of transformers, llama attention bias is hard coded to False
                    attention_bias = False if not hasattr(model.config,
                                                          "attention_bias") else model.config.attention_bias
                    original_q_weight = attn.q_proj.weight.data
                    original_v_weight = attn.v_proj.weight.data
                    original_q_bias = attn.q_proj.bias.data if attention_bias else None
                    original_v_bias = attn.v_proj.bias.data if attention_bias else None
                    if tasks is None:
                        attn.q_proj = BilevelLoRALinear(
                            model.config.hidden_size,
                            model.config.hidden_size,
                            r=r, lora_alpha=alpha, bias=attention_bias
                        ).to(original_q_weight.device)
                        attn.v_proj = BilevelLoRALinear(
                            model.config.hidden_size,
                            model.config.hidden_size,
                            r=r, lora_alpha=alpha, bias=attention_bias
                        ).to(original_v_weight.device)
                    else:
                        attn.q_proj = MultiTaskBilevelLoRALinear(
                            tasks,
                            model.config.hidden_size,
                            model.config.hidden_size,
                            r=r, lora_alphas=alpha, bias=attention_bias
                        ).to(original_q_weight.device)
                        attn.v_proj = MultiTaskBilevelLoRALinear(
                            tasks,
                            model.config.hidden_size,
                            model.config.hidden_size,
                            r=r, lora_alphas=alpha, bias=attention_bias
                        ).to(original_v_weight.device)
                    if float16:
                        attn.q_proj.half()
                        attn.v_proj.half()
                    attn.q_proj.weight.data = original_q_weight
                    attn.v_proj.weight.data = original_v_weight
                    if attention_bias:
                        attn.q_proj.bias.data = original_q_bias
                        attn.v_proj.bias.data = original_v_bias
                elif model.config.model_type == "mistral":
                    # in early version of transformers, llama attention bias is hard coded to False
                    config = model.config
                    original_q_weight = attn.q_proj.weight.data
                    original_v_weight = attn.v_proj.weight.data
                    head_dim = config.hidden_size // config.num_attention_heads
                    if tasks is None:
                        attn.q_proj = BilevelLoRALinear(
                            config.hidden_size,
                            config.hidden_size,
                            r=r, lora_alpha=alpha
                        ).to(original_q_weight.device)
                        attn.v_proj = BilevelLoRALinear(
                            config.hidden_size,
                            config.num_key_value_heads * head_dim,
                            r=r, lora_alpha=alpha
                        ).to(original_v_weight.device)
                    else:
                        attn.q_proj = MultiTaskBilevelLoRALinear(
                            tasks,
                            config.hidden_size,
                            config.hidden_size,
                            lora_ranks=r, lora_alphas=alpha
                        ).to(original_q_weight.device)
                        attn.v_proj = MultiTaskBilevelLoRALinear(
                            tasks,
                            config.hidden_size,
                            config.num_key_value_heads * head_dim,
                            lora_ranks=r, lora_alphas=alpha
                        ).to(original_v_weight.device)
                    if float16:
                        attn.q_proj.half()
                        attn.v_proj.half()
                    attn.q_proj.weight.data = original_q_weight
                    attn.v_proj.weight.data = original_v_weight
                elif model.config.model_type == "gpt2":
                    original_c_attn_weight = attn.c_attn.weight.data
                    original_c_attn_bias = attn.c_attn.bias.data
                    original_c_proj_weight = attn.c_proj.weight.data
                    original_c_prof_bias = attn.c_proj.bias.data
                    if tasks is None:
                        attn.c_attn = BilevelLoRAConv1D(model.config.hidden_size * 3, model.config.hidden_size, r=r,
                                                        lora_alpha=alpha, ).to(original_c_attn_weight.device)
                        attn.c_proj = BilevelLoRAConv1D(model.config.hidden_size, model.config.hidden_size, r=r,
                                                        lora_alpha=alpha).to(original_c_proj_weight.device)
                    else:
                        attn.c_attn = MultiTaskBilevelLoRAConv1D(tasks, model.config.hidden_size * 3,
                                                                 model.config.hidden_size, lora_ranks=r,
                                                                 lora_alphas=alpha).to(original_c_attn_weight.device)
                        attn.c_proj = MultiTaskBilevelLoRAConv1D(tasks, model.config.hidden_size,
                                                                 model.config.hidden_size, lora_ranks=r,
                                                                 lora_alphas=alpha).to(original_c_proj_weight.device)
                    if float16:
                        attn.c_attn.half()
                        attn.c_proj.half()
                    attn.c_attn.weight.data = original_c_attn_weight
                    attn.c_attn.bias.data = original_c_attn_bias
                    attn.c_proj.weight.data = original_c_proj_weight
                    attn.c_proj.bias.data = original_c_prof_bias
                else:
                    raise NotImplementedError

        # Freeze non-LoRA parameters
        for n, p in model.named_parameters():
            if "lora" not in n:
                p.requires_grad = False

    def remove_loras(self, model):
        for key, _ in model.named_modules():
            if key[-len(self.attention_name):] == self.attention_name:
                logger.info(f"Removing lora from: {key}")
                _, _, attn = find_module(model, key)

                if model.config.model_type == "opt":
                    original_q_weight = attn.q_proj.weight.data
                    original_q_bias = attn.q_proj.bias.data
                    original_v_weight = attn.v_proj.weight.data
                    original_v_bias = attn.v_proj.bias.data
                    attn.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=model.config.enable_bias).to(
                        original_q_weight.device)
                    attn.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=model.config.enable_bias).to(
                        original_v_weight.device)
                    if self.float16:
                        attn.q_proj.half()
                        attn.v_proj.half()
                    attn.q_proj.weight.data = original_q_weight
                    attn.q_proj.bias.data = original_q_bias
                    attn.v_proj.weight.data = original_v_weight
                    attn.v_proj.bias.data = original_v_bias
                elif model.config.model_type == "llama":
                    attention_bias = False if not hasattr(model.config,
                                                          "attention_bias") else model.config.attention_bias
                    original_q_weight = attn.q_proj.weight.data
                    original_v_weight = attn.v_proj.weight.data
                    original_q_bias = attn.q_proj.bias.data if attention_bias else None
                    original_v_bias = attn.v_proj.bias.data if attention_bias else None
                    attn.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=attention_bias).to(
                        original_q_weight.device)
                    attn.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=attention_bias).to(
                        original_v_weight.device)
                    if self.float16:
                        attn.q_proj.half()
                        attn.v_proj.half()
                    attn.q_proj.weight.data = original_q_weight
                    attn.v_proj.weight.data = original_v_weight
                    if attention_bias:
                        attn.q_proj.bias.data = original_q_bias
                        attn.v_proj.bias.data = original_v_bias
                elif model.config.model_type == "mistral":
                    config = model.config
                    original_q_weight = attn.q_proj.weight.data
                    original_v_weight = attn.v_proj.weight.data
                    head_dim = config.hidden_size // config.num_attention_heads
                    attn.q_proj = nn.Linear(config.hidden_size, config.hidden_size).to(original_q_weight.device)
                    attn.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * head_dim).to(
                        original_v_weight.device)
                    if self.float16:
                        attn.q_proj.half()
                        attn.v_proj.half()
                    attn.q_proj.weight.data = original_q_weight
                    attn.v_proj.weight.data = original_v_weight
                elif model.config.model_type == "gpt2":
                    original_c_attn_weight = attn.c_attn.weight.data
                    original_c_attn_bias = attn.c_attn.bias.data
                    original_c_proj_weight = attn.c_proj.weight.data
                    original_c_prof_bias = attn.c_proj.bias.data
                    attn.c_attn = Conv1D(self.hidden_dim * 3, self.hidden_dim).to(original_c_attn_weight.device)
                    attn.c_proj = Conv1D(self.hidden_dim, self.hidden_dim).to(original_c_proj_weight.device)
                    if self.float16:
                        attn.c_attn.half()
                        attn.c_proj.half()
                    attn.c_attn.weight.data = original_c_attn_weight
                    attn.c_attn.bias.data = original_c_attn_bias
                    attn.c_proj.weight.data = original_c_proj_weight
                    attn.c_proj.bias.data = original_c_prof_bias
                else:
                    raise NotImplementedError

        return model
