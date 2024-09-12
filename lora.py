import logging
from typing import Union, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import torch
from torch import nn
from torch.nn import functional as F
import math


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
        else:
            raise NotImplementedError

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
                                                          lora_ranks=r, lora_alphas=alpha, bias=model.config.enable_bias).to(
                            original_q_weight.device)
                        attn.v_proj = MultiTaskLoRALinear(tasks, model.config.hidden_size, model.config.hidden_size,
                                                          lora_ranks=r, lora_alphas=alpha, bias=model.config.enable_bias).to(
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
                else:
                    raise NotImplementedError

        # Freeze non-LoRA parameters
        for n, p in model.named_parameters():
            if "lora" not in n:
                p.requires_grad = False
