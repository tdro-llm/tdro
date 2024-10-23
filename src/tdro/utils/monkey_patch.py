import torch
import transformers

import logging
logger = logging.getLogger(__name__)

def fused_rms_forward(self, hidden_states, residual=None, prenorm=False, residual_in_fp32=False):
    from flash_attn.ops.triton.layer_norm import rms_norm_fn

    # x: Input tensor.
    # weight, bias: Learnable parameters used in LayerNorm.
    # residual: Optional residual input, if provided, will be added to the output after LayerNorm.
    # x1, weight1, bias1: Input and corresponding learnable parameters for the second path, used for parallel LayerNorm.
    # eps: Numerical stability constant used for LayerNorm.
    # dropout_p: Dropout probability.
    # rowscale: Optional row scaling factor.
    # prenorm: A boolean indicating whether to include the original LayerNorm input in the return value.
    # dropout_mask, dropout_mask1: Optional dropout masks used to specify which elements should be zeroed out.
    # upcast: Boolean indicating whether to cast inputs and parameters to floating point (float) for computation.
    return rms_norm_fn(
            x=hidden_states,
            weight=self.weight,
            bias=None,
            residual=residual,
            eps=self.variance_epsilon,
            dropout_p=0.0,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32,
        )

def hacking_fused_rms_norm():
    import importlib.metadata
    from packaging import version

    if version.parse(importlib.metadata.version("torch")) >= version.parse("2.4.0"):
        # RMSNorm with weight (element affine)
        transformers.models.mistral.modeling_mistral.MistralRMSNorm = torch.nn.RMSNorm
        transformers.models.llama.modeling_llama.LlamaRMSNorm = torch.nn.RMSNorm
        transformers.models.qwen2.modeling_qwen2.Qwen2RMSNorm = torch.nn.RMSNorm
    else:
        # Triton Bug: https://github.com/state-spaces/mamba/issues/84
        # You should fix this before using triton with DataParallel
        transformers.models.mistral.modeling_mistral.MistralRMSNorm.forward = fused_rms_forward
        transformers.models.llama.modeling_llama.LlamaRMSNorm.forward = fused_rms_forward
        transformers.models.qwen2.modeling_qwen2.Qwen2RMSNorm.forward = fused_rms_forward

