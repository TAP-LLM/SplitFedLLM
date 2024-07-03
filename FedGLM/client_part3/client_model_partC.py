import math
import copy
import os
import warnings
import re
import sys

import torch
import torch.utils.checkpoint
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss, LayerNorm
from torch.nn.utils import skip_init
from typing import Optional, Tuple, Union, List, Callable, Dict, Any

from transformers import AutoTokenizer, AutoModel
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    BaseModelOutputWithPastAndCrossAttentions,
)
from transformers.utils import logging
from transformers.generation.utils import LogitsProcessorList, StoppingCriteriaList, GenerationConfig, ModelOutput

import modeling_chatglm
from modeling_chatglm import ChatGLMModel, ChatGLMPreTrainedModel, InvalidScoreLogitsProcessor, default_init, PrefixEncoder, GLMBlock
from client_partC_config import ClientCChatGLMConfig
from outputs import SplitModelPartAOutput


logger = logging.get_logger(__name__)

# client side model partC

class ChatGLMModelClientSideC(ChatGLMPreTrainedModel):
    """
    This model is the client side's model of splitfedGLM.

    The ChatGLMModel model can behave as an encoder (with only self-attention) as well
    as a decoder, in which case a layer of cross-attention is added between
    the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani,
    Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the
    `is_decoder` argument of the configuration set to `True`.
    To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder`
    argument and `add_cross_attention` set to `True`; an
    `encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self, config: ClientCChatGLMConfig, empty_init=True):
        super().__init__(config)
        if empty_init:
            init_method = skip_init
        else:
            init_method = default_init
        # recording parameters
        self.max_sequence_length = config.max_sequence_length
        self.hidden_size = config.hidden_size
        self.params_dtype = torch.half
        self.num_attention_heads = config.num_attention_heads
        self.vocab_size = config.vocab_size
        self.client_num_layers = config.num_clients_layers  # new config!! means num of glmblocks in client's side
        self.layernorm_epsilon = config.layernorm_epsilon
        self.inner_hidden_size = config.inner_hidden_size
        self.hidden_size_per_attention_head = self.hidden_size // self.num_attention_heads
        self.position_encoding_2d = config.position_encoding_2d
        self.pre_seq_len = config.pre_seq_len
        self.prefix_projection = config.prefix_projection

        # self.word_embeddings = init_method(
        #     torch.nn.Embedding,
        #     num_embeddings=self.vocab_size, embedding_dim=self.hidden_size,
        #     dtype=self.params_dtype
        # )
        self.gradient_checkpointing = False

        def get_layer(layer_id):
            return GLMBlock(
                self.hidden_size,
                self.num_attention_heads,
                self.layernorm_epsilon,
                layer_id,
                inner_hidden_size=self.inner_hidden_size,
                hidden_size_per_attention_head=self.hidden_size_per_attention_head,
                layernorm=LayerNorm,
                use_bias=True,
                params_dtype=self.params_dtype,
                position_encoding_2d=self.position_encoding_2d,
                empty_init=empty_init
            )

        self.layers = torch.nn.ModuleList(
            [get_layer(layer_id) for layer_id in range(self.client_num_layers)]
        )  # modulelist  重写可以实现任意切割，但是要重写两个类。。。

        # Final layer norm before output.
        self.final_layernorm = LayerNorm(self.hidden_size, eps=self.layernorm_epsilon)
       
        if self.pre_seq_len is not None: 
            for param in self.parameters():
                param.requires_grad = False
        #     self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        #     self.prefix_encoder = PrefixEncoder(config)
        #     self.dropout = torch.nn.Dropout(0.1)

            # total_params = sum(p.numel() for p in self.parameters())
            # trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            # print("Using p-tuning v2: # trainable_params = {} / {}".format(trainable_params, total_params))

    def get_input_embeddings(self):
        return self.word_embeddings

    def set_input_embeddings(self, new_embeddings: torch.Tensor):
        self.word_embeddings = new_embeddings

    def get_prompt(self, batch_size, device, dtype=torch.half):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(device)
        past_key_values = self.prefix_encoder(prefix_tokens).type(dtype)
        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            self.num_layers * 2,  # 28层均需要在partA生成，partB、C不可再调用该函数！
            self.num_attention_heads,
            self.hidden_size // self.num_attention_heads
        )
        # seq_len, b, nh, hidden_size
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 1, 0, 3, 4]).split(2)
        # past_key_values = [(v[0], v[1]) for v in past_key_values]
        return past_key_values

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            hidden_states: Optional[torch.Tensor] = None, # from clients
            past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
            inputs_embeds: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor, ...], BaseModelOutputWithPast]:
        # 四个判断全部false！
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.gradient_checkpointing and self.training: # 无用
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False
        # partA仅使用input_ids:tensor{batch_size, seq_length}
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None: 
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        elif hidden_states is not None:   # client modelC!!
           seq_length, batch_size = hidden_states.shape[:2] # (seq_length, batch_size, hidden_size)
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # if inputs_embeds is None:
        #     inputs_embeds = self.word_embeddings(input_ids)

        if past_key_values is None:
            if self.pre_seq_len is not None:
                past_key_values = self.get_prompt(batch_size=input_ids.shape[0], device=input_ids.device,
                                                  dtype=inputs_embeds.dtype) # 每一步均需调用！
            else:
                past_key_values = tuple([None] * len(self.layers))

            if attention_mask is None:
                attention_mask = self.get_masks(
                    input_ids,
                    device=input_ids.device
                )

            if position_ids is None:
                MASK, gMASK = self.config.mask_token_id, self.config.gmask_token_id
                seqs = input_ids.tolist()

                mask_positions, use_gmasks = [], []
                for seq in seqs:
                    mask_token = gMASK if gMASK in seq else MASK
                    use_gmask = mask_token == gMASK
                    mask_positions.append(seq.index(mask_token))
                    use_gmasks.append(use_gmask)

                position_ids = self.get_position_ids(
                    input_ids,
                    mask_positions=mask_positions,
                    device=input_ids.device,
                    use_gmasks=use_gmasks
                )
        # bc模型的attention mask已经扩张过，不用再处理，pre_seq_len参数影响模型参数是否冻结
        # if self.pre_seq_len is not None and attention_mask is not None:
        #     prefix_attention_mask = torch.ones(batch_size, 1, input_ids.size(-1), self.pre_seq_len).to(
        #         attention_mask.device)
        #     prefix_attention_mask = (prefix_attention_mask < 0.5).bool()
        #     attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=3)

        # [seq_len, batch, hidden_size]
        # hidden_states = inputs_embeds.transpose(0, 1)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        if attention_mask is None:
            attention_mask = torch.zeros(1, 1, device=input_ids.device).bool()
        else: # here
            attention_mask = attention_mask.to(hidden_states.device)

        for i, layer in enumerate(self.layers):

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_past = past_key_values[i]

            if self.gradient_checkpointing and self.training:
                layer_ret = torch.utils.checkpoint.checkpoint(
                    layer,
                    hidden_states,
                    position_ids,
                    attention_mask,
                    torch.tensor(i),
                    layer_past,
                    use_cache,
                    output_attentions
                )
            else:
                layer_ret = layer(
                    hidden_states,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    layer_id=torch.tensor(i),
                    layer_past=layer_past,
                    use_cache=use_cache,
                    output_attentions=output_attentions
                )

            hidden_states = layer_ret[0]

            if use_cache:
                presents = presents + (layer_ret[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_ret[2 if use_cache else 1],)

        # Final layer norm. 
        hidden_states = self.final_layernorm(hidden_states) # clientsideC model include final layernorm.

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict: # return_dict=true!
            # return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)
            return tuple(v for v in [hidden_states, position_ids, attention_mask, past_key_values[self.client_num_layers:]] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

class ChatGLMForConditionalGenerationClientSideC(ChatGLMPreTrainedModel): 
    def __init__(self, config: ClientCChatGLMConfig, empty_init=True): 
        super().__init__(config)
        if empty_init:
            init_method = skip_init
        else:
            init_method = default_init

        # self.hidden_size = config.hidden_size
        # self.params_dtype = torch.half
        # self.vocab_size = config.vocab_size
        self.max_sequence_length = config.max_sequence_length

        self.position_encoding_2d = config.position_encoding_2d

        self.transformer = ChatGLMModelClientSideC(config, empty_init=empty_init)  # changed!!

        self.lm_head = init_method(
            nn.Linear,
            config.hidden_size,
            config.vocab_size,
            bias=False,
            dtype=torch.half
        )

        self.config = config

        self.quantized = False

        if self.config.quantization_bit:
            self.quantize(self.config.quantization_bit, empty_init=True)

        # 冻结一下参数，我找不到这个head层到底是在哪里冻结的
        if self.config.pre_seq_len is not None:
            if self.config.freeze_head:
                self.lm_head.requires_grad_(False)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
    ) -> Dict[str, Any]:
        # update past_key_values
        model_kwargs["past_key_values"] = self._extract_past_from_model_output(
            outputs, standardize_cache_format=standardize_cache_format
        )

        # update attention mask
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            if attention_mask is not None and attention_mask.dtype == torch.bool:
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_ones((*attention_mask.shape[:3], 1))], dim=3)
                new_attention_mask = attention_mask[:, :, -1:].clone()
                new_attention_mask[..., -1] = False
                model_kwargs["attention_mask"] = torch.cat(
                    [attention_mask, new_attention_mask], dim=2
                )

        # update position ids
        if "position_ids" in model_kwargs:
            position_ids = model_kwargs["position_ids"]
            new_position_id = position_ids[..., -1:].clone()
            new_position_id[:, 1, :] += 1
            model_kwargs["position_ids"] = torch.cat(
                [position_ids, new_position_id], dim=-1
            )

        return model_kwargs

    def prepare_inputs_for_generation(
            self,
            input_ids: torch.LongTensor,
            past: Optional[torch.Tensor] = None,
            past_key_values: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            **kwargs
    ) -> dict:
        batch_size, seq_length = input_ids.shape
        MASK, gMASK = self.config.mask_token_id, self.config.gmask_token_id
        seqs = input_ids.tolist()
        mask_positions, use_gmasks = [], []
        for seq in seqs:
            mask_token = gMASK if gMASK in seq else MASK
            use_gmask = mask_token == gMASK
            mask_positions.append(seq.index(mask_token))
            use_gmasks.append(use_gmask)

        # only last token for input_ids if past is not None
        if past is not None or past_key_values is not None:
            last_token = input_ids[:, -1].unsqueeze(-1)
            if attention_mask is not None and attention_mask.dtype == torch.bool:
                attention_mask = attention_mask[:, :, -1:]
            else:
                attention_mask = None
            if position_ids is not None:
                position_ids = position_ids[..., -1:]
            else:
                context_lengths = [seq.index(self.config.bos_token_id) for seq in seqs]
                if self.position_encoding_2d:
                    position_ids = torch.tensor(
                        [[mask_position, seq_length - context_length] for mask_position, context_length in
                         zip(mask_positions, context_lengths)], dtype=torch.long, device=input_ids.device).unsqueeze(-1)
                else:
                    position_ids = torch.tensor([mask_position for mask_position in mask_positions], dtype=torch.long,
                                                device=input_ids.device).unsqueeze(-1)

            if past is None:
                past = past_key_values
            return {
                "input_ids": last_token,
                "past_key_values": past,
                "position_ids": position_ids,
                "attention_mask": attention_mask
            }
        else:
            if attention_mask is not None and attention_mask.dtype != torch.bool:
                logger.warning_once(f"The dtype of attention mask ({attention_mask.dtype}) is not bool")
                attention_mask = None
            if attention_mask is None:
                attention_mask = self.get_masks(
                    input_ids,
                    device=input_ids.device
                )
            if position_ids is None:
                position_ids = self.get_position_ids(
                    input_ids,
                    device=input_ids.device,
                    mask_positions=mask_positions,
                    use_gmasks=use_gmasks
                )

            return {
                "input_ids": input_ids,
                "past_key_values": past,
                "position_ids": position_ids,
                "attention_mask": attention_mask
            }

    def forward( 
            self,
            input_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            hidden_state: Optional[torch.Tensor] = None,# from client
            past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None, # Cmodel input
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            hidden_states=hidden_state,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        ) # SplitModelPartAOutput

        hidden_states = transformer_outputs[0]
        #print(f"client_model_partC_hidden_states: {hidden_states}")
        # print('modelC final hidden_states',  hidden_states)
        # 6/16新增
        # torch.Size([128, 1, 130528])
        lm_logits = self.lm_head(hidden_states).permute(1, 0, 2).contiguous() # 这是模型的最后一层, 未冻结情况下，eps为1e-8时会在更新参数后出现nan值，
        #lm_logits.size(): torch.Size([1, 128, 130528])

        # lm_logits = None
        loss = None
        if labels is not None:  # !!
            print("labels is not None")
            lm_logits = lm_logits.to(torch.float32)

            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            lm_logits = lm_logits.to(hidden_states.dtype)
            loss = loss.to(hidden_states.dtype)

        if not return_dict: # return_dict=true
            print("return_dict is False")
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output
            # return transformer_outputs # tuple(hidden_states, position_ids, attention_mask, past_key_values[self.client_num_layers:])

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    @staticmethod
    def _reorder_cache(
            past: Tuple[Tuple[torch.Tensor, torch.Tensor], ...], beam_idx: torch.LongTensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.

        Output shares the same memory storage as `past`.
        """
        return tuple(
            (
                layer_past[0].index_select(1, beam_idx.to(layer_past[0].device)),
                layer_past[1].index_select(1, beam_idx.to(layer_past[1].device)),
            )
            for layer_past in past
        )

    def process_response(self, response):
        response = response.strip()
        response = response.replace("[[训练时间]]", "2023年")
        punkts = [
            [",", "，"],
            ["!", "！"],
            [":", "："],
            [";", "；"],
            ["\?", "？"],
        ]
        for item in punkts:
            response = re.sub(r"([\u4e00-\u9fff])%s" % item[0], r"\1%s" % item[1], response)
            response = re.sub(r"%s([\u4e00-\u9fff])" % item[0], r"%s\1" % item[1], response)
        return response

    @torch.no_grad()
    def chat(self, tokenizer, query: str, history: List[Tuple[str, str]] = None, max_length: int = 2048, num_beams=1,
             do_sample=True, top_p=0.7, temperature=0.95, logits_processor=None, **kwargs):
        if history is None:
            history = []
        if logits_processor is None:
            logits_processor = LogitsProcessorList()
        logits_processor.append(InvalidScoreLogitsProcessor())
        gen_kwargs = {"max_length": max_length, "num_beams": num_beams, "do_sample": do_sample, "top_p": top_p,
                      "temperature": temperature, "logits_processor": logits_processor, **kwargs}
        if not history:
            prompt = query
        else:
            prompt = ""
            for i, (old_query, response) in enumerate(history):
                prompt += "[Round {}]\n问：{}\n答：{}\n".format(i, old_query, response)
            prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)
        inputs = tokenizer([prompt], return_tensors="pt")
        # 6/16新增
        print(f"inputs: {inputs}")
        # 输出形状
        print(f"inputs['input_ids'].shape: {inputs['input_ids'].shape}")
        inputs = inputs.to(self.device)
        outputs = self.generate(**inputs, **gen_kwargs)
        print(f"outputs: {outputs}")
        # 输出形状
        print(f"outputs.shape: {outputs.shape}")
        outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):]
        response = tokenizer.decode(outputs)
        response = self.process_response(response)
        history = history + [(query, response)]
        return response, history

    @torch.no_grad()
    def stream_chat(self, tokenizer, query: str, history: List[Tuple[str, str]] = None, max_length: int = 2048,
                    do_sample=True, top_p=0.7, temperature=0.95, logits_processor=None, **kwargs):
        if history is None:
            history = []
        if logits_processor is None:
            logits_processor = LogitsProcessorList()
        logits_processor.append(InvalidScoreLogitsProcessor())
        gen_kwargs = {"max_length": max_length, "do_sample": do_sample, "top_p": top_p,
                      "temperature": temperature, "logits_processor": logits_processor, **kwargs}
        if not history:
            prompt = query
        else:
            prompt = ""
            for i, (old_query, response) in enumerate(history):
                prompt += "[Round {}]\n问：{}\n答：{}\n".format(i, old_query, response)
            prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)
        inputs = tokenizer([prompt], return_tensors="pt")
        inputs = inputs.to(self.device)
        for outputs in self.stream_generate(**inputs, **gen_kwargs):
            outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):]
            response = tokenizer.decode(outputs)
            response = self.process_response(response)
            new_history = history + [(query, response)]
            yield response, new_history

    @torch.no_grad()
    def stream_generate(
            self,
            input_ids,
            generation_config: Optional[GenerationConfig] = None,
            logits_processor: Optional[LogitsProcessorList] = None,
            stopping_criteria: Optional[StoppingCriteriaList] = None,
            prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
            **kwargs,
    ):
        batch_size, input_ids_seq_length = input_ids.shape[0], input_ids.shape[-1]

        if generation_config is None:
            generation_config = self.generation_config
        generation_config = copy.deepcopy(generation_config)
        model_kwargs = generation_config.update(**kwargs)
        bos_token_id, eos_token_id = generation_config.bos_token_id, generation_config.eos_token_id

        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]

        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        if has_default_max_length and generation_config.max_new_tokens is None:
            warnings.warn(
                f"Using `max_length`'s default ({generation_config.max_length}) to control the generation length. "
                "This behaviour is deprecated and will be removed from the config in v5 of Transformers -- we"
                " recommend using `max_new_tokens` to control the maximum length of the generation.",
                UserWarning,
            )
        elif generation_config.max_new_tokens is not None:
            generation_config.max_length = generation_config.max_new_tokens + input_ids_seq_length
            if not has_default_max_length:
                logger.warn(
                    f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                    f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                    "Please refer to the documentation for more information. "
                    "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)",
                    UserWarning,
                )

        if input_ids_seq_length >= generation_config.max_length:
            input_ids_string = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
            logger.warning(
                f"Input length of {input_ids_string} is {input_ids_seq_length}, but `max_length` is set to"
                f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
                " increasing `max_new_tokens`."
            )

        # 2. Set generation parameters if not already defined
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_seq_length,
            encoder_input_ids=input_ids,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
        )

        stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config, stopping_criteria=stopping_criteria
        )
        logits_warper = self._get_logits_warper(generation_config)

        unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
        scores = None
        while True:
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
            )

            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)

            # sample
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            if generation_config.do_sample:
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(probs, dim=-1)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            unfinished_sequences = unfinished_sequences.mul((sum(next_tokens != i for i in eos_token_id)).long())

            # stop when each sentence is finished, or if we exceed the maximum length
            if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
                break
            yield input_ids

    def quantize(self, bits: int, empty_init=False, **kwargs):
        if bits == 0:
            return

        from quantization import quantize

        if self.quantized:
            logger.info("Already quantized.")
            return self

        self.quantized = True

        self.config.quantization_bit = bits

        self.transformer = quantize(self.transformer, bits, empty_init=empty_init, **kwargs)
        return self
    
