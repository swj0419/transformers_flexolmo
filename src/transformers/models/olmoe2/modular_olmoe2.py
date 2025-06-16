from typing import Callable, Optional, Tuple

import torch

from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from ...cache_utils import Cache
from ...pytorch_utils import ALL_LAYERNORM_LAYERS
from ...utils import logging
from ..llama.modeling_llama import LlamaRMSNorm, eager_attention_forward
from ..olmoe.configuration_olmoe import OlmoeConfig
from ..olmoe.modeling_olmoe import (
    OlmoeAttention,
    OlmoeDecoderLayer,
    OlmoeForCausalLM,
    OlmoeModel,
    apply_rotary_pos_emb,
)


logger = logging.get_logger(__name__)


# class Olmoe2Config(OlmoeConfig):
#     r"""
#     This is the configuration class to store the configuration of a [`Olmoe2Model`]. It is used to instantiate an OLMoE2
#     model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
#     defaults will yield a similar configuration to that of the [allenai/OLMoE2-1B-7B-0924](https://huggingface.co/allenai/OLMoE2-1B-7B-0924).

#     Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
#     documentation from [`PretrainedConfig`] for more information.


#     Args:
#         vocab_size (`int`, *optional*, defaults to 50304):
#             Vocabulary size of the OLMoE2 model. Defines the number of different tokens that can be represented by the
#             `inputs_ids` passed when calling [`Olmoe2Model`]
#         hidden_size (`int`, *optional*, defaults to 2048):
#             Dimension of the hidden representations.
#         intermediate_size (`int`, *optional*, defaults to 2048):
#             Dimension of the MLP representations.
#         num_hidden_layers (`int`, *optional*, defaults to 16):
#             Number of hidden layers in the Transformer decoder.
#         num_attention_heads (`int`, *optional*, defaults to 16):
#             Number of attention heads for each attention layer in the Transformer decoder.
#         num_key_value_heads (`int`, *optional*):
#             This is the number of key_value heads that should be used to implement Grouped Query Attention. If
#             `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
#             `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
#             converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
#             by meanpooling all the original heads within that group. For more details checkout [this
#             paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
#             `num_attention_heads`.
#         hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
#             The non-linear activation function (function or string) in the decoder.
#         max_position_embeddings (`int`, *optional*, defaults to 4096):
#             The maximum sequence length that this model might ever be used with.
#         initializer_range (`float`, *optional*, defaults to 0.02):
#             The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
#         rms_norm_eps (`float`, *optional*, defaults to 1e-05):
#             The epsilon used by the rms normalization layers.
#         use_cache (`bool`, *optional*, defaults to `True`):
#             Whether or not the model should return the last key/values attentions (not used by all models). Only
#             relevant if `config.is_decoder=True`.
#         pad_token_id (`int`, *optional*, defaults to 1):
#             Padding token id.
#         bos_token_id (`int`, *optional*):
#             Beginning of stream token id.
#         eos_token_id (`int`, *optional*, defaults to 50279):
#             End of stream token id.
#         tie_word_embeddings (`bool`, *optional*, defaults to `False`):
#             Whether to tie weight embeddings
#         rope_theta (`float`, *optional*, defaults to 10000.0):
#             The base period of the RoPE embeddings.
#         rope_scaling (`Dict`, *optional*):
#             Dictionary containing the scaling configuration for the RoPE embeddings. Currently supports two scaling
#             strategies: linear and dynamic. Their scaling factor must be a float greater than 1. The expected format is
#             `{"type": strategy name, "factor": scaling factor}`. When using this flag, don't update
#             `max_position_embeddings` to the expected new maximum. See the following thread for more information on how
#             these scaling strategies behave:
#             https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/. This is an
#             experimental feature, subject to breaking API changes in future versions.
#         attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
#             Whether to use a bias in the query, key, value and output projection layers during self-attention.
#         attention_dropout (`float`, *optional*, defaults to 0.0):
#             The dropout ratio for the attention probabilities.
#         clip_qkv (`float`, *optional*):
#             If not `None`, elements of query, key and value attention states are clipped so that their
#             absolute value does not exceed this value.
#         num_experts_per_tok (`int`, *optional*, defaults to 8):
#             Number of selected experts.
#         num_experts (`int`, *optional*, defaults to 64):
#             Number of routed experts.
#         output_router_logits (`bool`, *optional*, defaults to `False`):
#             Whether or not the router logits should be returned by the model. Enabeling this will also
#             allow the model to output the auxiliary loss, including load balancing loss and router z-loss.
#         router_aux_loss_coef (`float`, *optional*, defaults to 0.01):
#             The aux loss factor for the total loss.
#         norm_topk_prob (`bool`, *optional*, defaults to `False`):
#             Whether to normalize the topk probabilities.
#         rms_norm_eps (`float`, *optional*, defaults to 1e-05):
#             The epsilon used by the rms normalization layers.

#     ```python
#     >>> from transformers import Olmoe2Model, Olmoe2Config

#     >>> # Initializing a OLMoE2 7B A1B style configuration
#     >>> configuration = Olmoe2Config()

#     >>> # Initializing a model from the OLMoE2 7B A1B style configuration
#     >>> model = Olmoe2Model(configuration)

#     >>> # Accessing the model configuration
#     >>> configuration = model.config
#     ```"""

#     model_type = "olmoe2"
#     keys_to_ignore_at_inference = ["past_key_values"]

#     def __init__(
#         self,
#         vocab_size=50304,
#         hidden_size=2048,
#         intermediate_size=2048,
#         num_hidden_layers=16,
#         num_attention_heads=16,
#         num_key_value_heads=None,
#         hidden_act="silu",
#         max_position_embeddings=4096,
#         initializer_range=0.02,
#         rms_norm_eps=1e-05,
#         use_cache=True,
#         pad_token_id=1,
#         bos_token_id=None,
#         eos_token_id=50279,
#         tie_word_embeddings=False,
#         rope_theta=10000.0,
#         rope_scaling=None,
#         attention_bias=False,
#         attention_dropout=0.0,
#         clip_qkv=None,
#         num_experts_per_tok=8,
#         num_experts=64,
#         output_router_logits=False,
#         router_aux_loss_coef=0.01,
#         norm_topk_prob=False,
#         **kwargs,
#     ):
#         super().__init__(
#             vocab_size=vocab_size,
#             hidden_size=hidden_size,
#             intermediate_size=intermediate_size,
#             num_hidden_layers=num_hidden_layers,
#             num_attention_heads=num_attention_heads,
#             num_key_value_heads=num_key_value_heads,
#             hidden_act=hidden_act,
#             max_position_embeddings=max_position_embeddings,
#             initializer_range=initializer_range,
#             rms_norm_eps=rms_norm_eps,
#             use_cache=use_cache,
#             pad_token_id=pad_token_id,
#             bos_token_id=bos_token_id,
#             eos_token_id=eos_token_id,
#             tie_word_embeddings=tie_word_embeddings,
#             rope_theta=rope_theta,
#             rope_scaling=rope_scaling,
#             attention_bias=attention_bias,
#             attention_dropout=attention_dropout,
#             clip_qkv=clip_qkv,
#             num_experts_per_tok=num_experts_per_tok,
#             num_experts=num_experts,
#             output_router_logits=output_router_logits,
#             router_aux_loss_coef=router_aux_loss_coef,
#             norm_topk_prob=norm_topk_prob,
#             **kwargs,
#         )


class Olmoe2Config(OlmoeConfig):
    pass


class Olmoe2RMSNorm(LlamaRMSNorm):
    pass


ALL_LAYERNORM_LAYERS.append(Olmoe2RMSNorm)


class Olmoe2Attention(OlmoeAttention):
    def __init__(self, config: Olmoe2Config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx=layer_idx)
        self.scaling = self.head_dim**-0.5

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states))
        key_states = self.k_norm(self.k_proj(hidden_states))
        value_states = self.v_proj(hidden_states)

        if self.config.clip_qkv is not None:
            query_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
            key_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
            value_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)

        query_states = query_states.view(hidden_shape).transpose(1, 2)
        key_states = key_states.view(hidden_shape).transpose(1, 2)
        value_states = value_states.view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights, past_key_value


class Olmoe2DecoderLayer(OlmoeDecoderLayer):
    def __init__(self, config: Olmoe2Config, layer_idx: int):
        super().__init__(config, layer_idx=layer_idx)
        self.post_attention_layernorm = Olmoe2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Olmoe2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Olmoe2Attention(config=config, layer_idx=layer_idx)
        del self.input_layernorm

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states, router_logits = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if output_router_logits:
            outputs += (router_logits,)

        return outputs


class Olmoe2Model(OlmoeModel):
    pass


class Olmoe2ForCausalLM(OlmoeForCausalLM):
    pass


__all__ = [
    "Olmoe2Config",
    "Olmoe2ForCausalLM",
    "Olmoe2Model",
    "Olmoe2PreTrainedModel",  # noqa: F822
]
