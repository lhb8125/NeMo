# coding=utf-8
# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import torch
from nemo.collections.nlp.modules.common.megatron.utils import _cast_if_autocast_enabled

try:
    from apex.normalization import MixedFusedRMSNorm

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False

def _set_sequence_parallel_enabled(
    param: torch.Tensor,
    sequence_parallel_enabled: bool,
) -> None:
    setattr(param, "sequence_parallel_enabled", sequence_parallel_enabled)

if HAVE_APEX:
    # TODO: use Apex implementation
    class RMSNormP(MixedFusedRMSNorm):
        def __init__(self, hidden_size, layernorm_epsilon, sequence_parallel_enabled: bool = False):
            super().__init__(hidden_size, layernorm_epsilon)
            self.sequence_parallel_enabled = sequence_parallel_enabled
            _set_sequence_parallel_enabled(self.weight, self.sequence_parallel_enabled)

else:

    class RMSNormP(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            raise NotImplementedError('RMSNormP available only with apex installed')



