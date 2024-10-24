import importlib
import math
import re
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from ..utils import PeftConfig, PeftType, transpose

import torch
from functools import partial
import torch.nn as nn
import torch.nn.functional as F 
import tensorly as tl
from tensorly.decomposition import tensor_train
import math
from typing import Optional, Tuple



#changed
@dataclass
class TTLoraConfig(PeftConfig):
    """Configuration class of TTLoRA"""

    tt_rank: int = field(default=4, metadata={"help": "TTLoRA rank for tensor decomposition"})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of modules to replace with TTLoRA layers"
        },
    )
    tt_alpha: int = field(default=None, metadata={"help": "TTLora alpha"})
    num_experts: int = field(default=None, metadata={"help": "Numbers of TTLora Experts"}) 
    blc_alpha: int = field(default=None, metadata={"help": "Alpha of blcloss"})
    blc_weight: int = field(default=None, metadata={"help": "Weight of blcloss"})
    tt_dropout: float = field(default=None, metadata={"help": "TTLora dropout"})
    tt_shape: Tuple[int, ...] = field(default=None, metadata={"help": "Shape of the TT decomposition."})

    merge_weights: bool = field(
        default=False, metadata={"help": "Merge weights of the original model and the TTLora model"}
    )
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    enable_ttlora: Optional[List[bool]] = field(default=None, metadata={"help": "Used with `TTlora.MergedLinear`."})
    bias: str = field(default="none", metadata={"help": "Bias type for TTLora. Can be 'none', 'all' or 'ttlora_only'"})
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from TTLoRA layers to be set as trainable and saved in the final checkpoint. "
        },
    )


    def __post_init__(self):
        self.peft_type = PeftType.TTLORA

#changed
class TTLoraModel(torch.nn.Module):
    """
    Creates TTLora model from a pretrained transformers model.
    """

    def __init__(self, config, model): # TTLoraConfig, CasualLM
        super().__init__()
        self.peft_config = config
        self.model = model
        self._find_and_replace()
        mark_only_ttlora_as_trainable(self.model, self.peft_config.bias)
        self.forward = self.model.forward

    def _find_and_replace(self):
        loaded_in_4bit = getattr(self.model, "is_loaded_in_4bit", False)
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        if (loaded_in_4bit or loaded_in_8bit):
            raise ImportError(
                "To use TTLora with 8-bit or 4-bit quantization, please install the `bitsandbytes` package. "
                "You can install it with `pip install bitsandbytes`."
            )
        is_target_modules_in_base_model = False
        is_hf_device_map_available = hasattr(self.model, "hf_device_map")
        kwargs = {
            "tt_rank": self.peft_config.tt_rank,
            "tt_alpha": self.peft_config.tt_alpha,
            "tt_dropout": self.peft_config.tt_dropout,
            "num_experts": self.peft_config.num_experts,
            "blc_alpha": self.peft_config.blc_alpha,
            "blc_weight": self.peft_config.blc_weight,
            "fan_in_fan_out": self.peft_config.fan_in_fan_out,
            "tt_shape": self.peft_config.tt_shape,
            "merge_weights": (self.peft_config.merge_weights or self.peft_config.inference_mode)
            and not is_hf_device_map_available,
        }
        key_list = [key for key, _ in self.model.named_modules()]
        for key in key_list:
            if isinstance(self.peft_config.target_modules, str):
                target_module_found = re.fullmatch(self.peft_config.target_modules, key)
            else:
                target_module_found = any(key.endswith(target_key) for target_key in self.peft_config.target_modules)
            if target_module_found: # here
                if not is_target_modules_in_base_model:
                    is_target_modules_in_base_model = True
                parent, target, target_name = self._get_submodules(key)
                bias = target.bias is not None

                if isinstance(target, torch.nn.Linear) and self.peft_config.enable_ttlora is None:
                    new_module = Linear(target.in_features, target.out_features, bias=bias, **kwargs)

                self._replace_module(parent, target_name, new_module, target)
        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {self.peft_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def _get_submodules(self, key):
        parent = self.model.get_submodule(".".join(key.split(".")[:-1]))
        target_name = key.split(".")[-1]
        target = self.model.get_submodule(key)
        return parent, target, target_name

    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        new_module.weight = old_module.weight
        if old_module.bias is not None:
            new_module.bias = old_module.bias
        if getattr(old_module, "state", None) is not None:
            new_module.state = old_module.state
            new_module.to(old_module.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if "lora_" in name:
                module.to(old_module.weight.device)

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    @property
    def modules_to_save(self):
        return None

    def get_peft_config_as_dict(self, inference: bool = False):
        config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(self.peft_config).items()}
        if inference:
            config["inference_mode"] = True
        return config

    def _set_adapter_layers(self, enabled=True):
        for module in self.model.modules():
            if isinstance(module, TTLoraLayer):
                module.disable_adapters = False if enabled else True

    def enable_adapter_layers(self):
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self):
        self._set_adapter_layers(enabled=False)


# Below code is based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# and modified to work with PyTorch FSDP


#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


# had to adapt it for `lora_only` to work
def mark_only_ttlora_as_trainable(model: nn.Module, bias: str = "none") -> None:
    for n, p in model.named_parameters():
        if "ttlora_" not in n:
            p.requires_grad = False
    if bias == "none":
        return
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "ttlora_only":
        for m in model.modules():
            if isinstance(m, TTLoraLayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


class TTLoraLayer:
    def __init__(
        self,
        tt_rank: int,
        tt_alpha: int,
        tt_dropout: float,
        merge_weights: bool,
        tt_shape:torch.Tensor,
    ):
        self.tt_rank = tt_rank
        self.tt_alpha = tt_alpha
        self.tt_shape = tt_shape
        # Optional dropout
        if tt_dropout > 0.0:
            self.tt_dropout = nn.Dropout(p=tt_dropout)
        else:
            self.tt_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights
        self.disable_adapters = False


class Linear(nn.Linear, TTLoraLayer):
    # TTLora implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        tt_rank: int = 4,
        tt_alpha: int = 1,
        tt_shape: torch.Tensor = torch.tensor([16, 16, 16, 16, 16, 16]),
        num_experts: int = 4,
        blc_alpha: float = 0.1,
        blc_weight: float = 0.1,
        tt_dropout: float = 0.05,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs,
    ):
        #initializes the Weights 
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        #initializes the TTLora layer
        TTLoraLayer.__init__(self, tt_rank=tt_rank, tt_alpha=tt_alpha, tt_shape=tt_shape, tt_dropout=tt_dropout, merge_weights=merge_weights)

        self.num_experts = num_experts
        self.blc_alpha = blc_alpha
        self.blc_weight = blc_weight
        self.tt_shape = tt_shape
        
        self.fan_in_fan_out = fan_in_fan_out


        # Actual trainable parameters
        if tt_alpha > 0:
            #initializes another layer of routing with dimension in_features * no_of_experts
            self.ttlora_route = nn.Linear(in_features, self.num_experts, bias=False)
            for i in range(self.num_experts):
                # setattr(self, f"lora_A{i}", nn.Linear(in_features, r, bias=False))
                # setattr(self, f"lora_B{i}", nn.Linear(r, out_features, bias=False))
                setattr(self, f"W_delta{i}", torch.zeros((self.in_features, self.out_features)))

            # self.scaling = self.tt_alpha
            
            # Freezing the dense layer's weights
            self.weight.requires_grad = False
        self.reset_parameters()

        # self.W_10d = self.reshape_to_10d(torch.tensor(self.W_delta))
        for i in range(self.num_experts):
        # Convert W_delta to 10D using reshape_to_10d and store it back
            W_delta = getattr(self, f"W_delta{i}")
            W_10d = self.reshape_to_10d(W_delta)
            setattr(self, f"W_10d{i}", W_10d)
        
        #initialize tt_cores as random
        self.tt_cores = nn.ModuleList([
            nn.ParameterList([nn.Parameter(self.init_core(*shape)) for shape in self.get_tt_shapes()])
            for _ in range(self.num_experts)
        ])
        
        #initialize core_dummy based on W_10d(projected tensor in higher dimension)
        #and transfer it to tt_cores
        for i in range(self.num_experts):
            W_10d = getattr(self, f"W_10d{i}")
            tt_cores_dummy = tensor_train(W_10d, self.tt_rank)
            for j, core in enumerate(tt_cores_dummy):
                self.tt_cores[i][j].data = torch.tensor(core, dtype=torch.float32)

        #Make all parameters of tt_cores trainable
        for cores in self.tt_cores:
            for core in cores:
                core.requires_grad = True

        if self.bias is not None:
                self.bias.requires_grad = False

        #Transpose if it is stored differently than expected (happens in some cases)
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        #initializes the weight of Linear layer
        nn.Linear.reset_parameters(self)
        if hasattr(self, "W_delta0"):
            #initializes the weights of deltaW
            for i in range(self.num_experts):
                torch.manual_seed(42)
                nn.init.kaiming_uniform_(getattr(self, f"W_delta{i}").weight, a=math.sqrt(8))
                # nn.init.zeros_(getattr(self, f"lora_B{i}").weight)

            #initializes the weights of routing layer
            nn.init.kaiming_uniform_(self.lora_route.weight, a=math.sqrt(5))

    def reshape_to_10d(self, tensor):
            return tensor.reshape(*self.tt_shape)
    
    def get_tt_shapes(self):
            shapes = []
            ranks = self.tt_rank
            for i in range(len(self.tt_shape)):
                shape = (ranks[i], self.tt_shape[i], ranks[i + 1])
                shapes.append(shape)
            return shapes
    
    def init_core(self, *shape):
            std = 1.0 / math.sqrt(shape[1])
            #generates a tensor shape based on shape variable and assigns value based on normal distribution multiplied by std
            return torch.randn(*shape) * std


    # def train(self, mode: bool = True):
    #     #train mode is set on and sets all the layers' train mode on
    #     nn.Linear.train(self, mode)
    #     self.lora_route.train(mode)
    #     for i in range(self.lora_num):
    #         getattr(self, f"lora_A{i}").train(mode)
    #         getattr(self, f"lora_B{i}").train(mode)

    def train(self, mode: bool = True):
        # Set the base linear layer and routing layer in train mode
        nn.Linear.train(self, mode)
        self.ttlora_route.train(mode)
        # Set the TT cores in train mode for each expert
        for i in range(self.num_experts):
            # Access the tt_cores for the current expert, which is a ParameterList
            for core in self.tt_cores[i]:
                core.train(mode)

    #all layers are set to evaluation mode
    # def eval(self):
    #     nn.Linear.eval(self)
    #     self.lora_route.eval()
    #     for i in range(self.lora_num):
    #         getattr(self, f"lora_A{i}").eval()
    #         getattr(self, f"lora_B{i}").eval()

    def eval(self):
        # Set the base linear layer and routing layer in eval mode
        nn.Linear.eval(self)
        self.ttlora_route.eval()
        # Set the TT cores in eval mode for each expert
        for i in range(self.num_experts):
            # Access the tt_cores for the current expert, which is a ParameterList
            for core in self.tt_cores[i]:
                core.eval()


    def cv_squared(self, x):
        #measures the dispersion of the spread
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)[0]
        return x.float().var() / (x.float().mean()**2 + eps)

    def forward(self, x: torch.Tensor, task_types=None):
        #x is input and comes from the previous layer in the forward pass
        #handling of adapters disable when needed
        if self.disable_adapters:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
            raise ImportError("Adapters are disabled") 
        #when adapter is applied and not merged to the pre-trained weights
        elif self.tt_alpha > 0 and not self.merged:
            #passes through the dense layer
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

            #calculates the routing weights
            route_weight = nn.functional.softmax(self.ttlora_route(x), dim=-1, dtype=torch.float32).to(result.dtype)

            #iterates over the experts - formula implementation B(A(Dropout(x)))
            for i in range(self.num_experts):
                #Get TT cores for the current expert
                tt_cores_expert = self.tt_cores[i]
                tt_weights = self.compute_tt_weights(tt_cores_expert)

                #Compute contribution of the current expert
                expert_contribution = torch.unsqueeze(route_weight[:, :, i], -1) * F.linear(x, tt_weights.reshape(self.in_features, self.out_features)) * self.tt_alpha
                result = result + expert_contribution
                
                # result = result + torch.unsqueeze(route_weight[:,:,i], -1) * getattr(self, f"lora_B{i}")(getattr(self, f"lora_A{i}")(self.lora_dropout(x))) * self.scaling

        blcls = torch.zeros(1)[0].to(result)
        if task_types != None:
            if self.blc_weight != 0:
                #changes into column vector
                task_types = task_types.view(-1, 1)

                blcls = self.cv_squared((
                    #sums the routing weight across each batch 
                    route_weight.sum(dim=(1)) * torch.where(
                        torch.concat(
                            ((task_types==1).repeat(1, self.num_experts//2), (task_types==0).repeat(1, self.num_experts//2)), dim=-1
                            ), 1.0+self.blc_alpha, 1.0-self.blc_alpha
                        )
                    ).flatten()
                ) * self.blc_weight

        def compute_tt_weights(self, tt_cores_expert):
            # Reconstruct the high-dimensional weight using tensor contraction based on TT cores
            if len(self.tt_shape) == 4:
                return torch.einsum('ijk,klm,mno,opq->jlnp', *tt_cores_expert)
            elif len(self.tt_shape) == 6:
                return torch.einsum('ijk,klm,mno,opq,qrs,stu->jlnprt', *tt_cores_expert)
            elif len(self.tt_shape) == 7:
                return torch.einsum('ijk,klm,mno,opq,qrs,stu,uvw->jlnprtv', *tt_cores_expert)
            elif len(self.tt_shape) == 8:
                return torch.einsum('ijk,klm,mno,opq,qrs,stu,uvw,wxy->jlnprtvx', *tt_cores_expert)
            elif len(self.tt_shape) == 10:
                return torch.einsum('ijk,klm,mno,opq,qrs,stu,uvw,wxy,yza,abc->jlnprtvxzb', *tt_cores_expert)
            elif len(self.tt_shape) == 12:
                return torch.einsum('ijk,klm,mno,opq,qrs,stu,uvw,wxy,yza,abc,cde,efg->jlnprtvxzbdf', *tt_cores_expert)

        return result, blcls

