# Copyright 2020 The HuggingFace Team. All rights reserved.
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

import warnings
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from transformers.utils import ModelOutput

@dataclass
class SplitModelPartAOutput(ModelOutput):
    """
    Custom output class for split model architecture (Part A), containing hidden states, position information,
    and attention-related tensors.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of this model part.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings.
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices (typically in the range [0, 1]).
            Mask values selected in [0, 1]:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors
            of shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            the model is used as a decoder, 2 additional tensors of shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            the model is used as a decoder, in the cross-attention blocks) that can be used (see `past_key_values` input)
            to speed up sequential decoding.
    """

    last_hidden_state: torch.FloatTensor = None
    position_ids: torch.LongTensor = None
    attention_mask: torch.Tensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None