"""Evo2 native recurrent/KV-cache rollouts and full causal training forwards.

Caches live for ONE rollout only; never reuse them after a policy update.
Actual sampling log probabilities are retained for importance ratios.
"""
import torch
from .autoregressive_rl import BASE_IDS, CausalPolicy


def enable_fp32_compute(model):
    """Keep frozen linear weights compact, but compute activations in FP32.

    Cast one projection at a time; autograd retains casts only downstream of
    trainable adapters. This avoids storing the entire 7B model in FP32.
    """
    from types import MethodType
    from vortex.model.layers import TELinear
    linear_parameters = set()

    def forward(module, inputs):
        bias = getattr(module, 'bias', None)
        output = torch.nn.functional.linear(inputs.float(), module.weight.float(),
                                            None if bias is None else bias.float())
        if isinstance(module, TELinear):
            # Evo2's projections expect the Transformer Engine tuple convention.
            if module.te_return_bias:
                raise RuntimeError('Separate TE bias return is not supported')
            return output, None
        return output

    for module in model.modules():
        if isinstance(module, (torch.nn.Linear, TELinear)):
            if any(p.requires_grad for p in module.parameters()):
                raise RuntimeError('Only frozen base projections may use compact storage')
            linear_parameters.update(id(p) for p in module.parameters())
            module.forward = MethodType(forward, module)
        for flag in ('use_flashfft', 'use_flash_depthwise', 'use_flash_rmsnorm',
                     'use_hcs_kernel', 'use_hcm_kernel', 'use_hcl_kernel'):
            if getattr(module, flag, False):
                raise RuntimeError(f'Disable {flag} before FP32 diagnostic')
    for parameter in model.parameters():
        if id(parameter) not in linear_parameters and parameter.is_floating_point():
            parameter.data = parameter.data.float()
    for module in model.modules():
        for name, buffer in module.named_buffers(recurse=False):
            if buffer.is_floating_point():
                setattr(module, name, buffer.float())
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def pad_fir_caches(model, state):
    """Represent absent causal history with zeros, including short prefills."""
    for module in model.modules():
        if not hasattr(module, 'short_filter_weight'):
            continue
        layer = module.layer_idx
        for key in ('hcl', 'hcm', 'hcs'):
            for attr, width in (
                ('fir_state_dict', module.short_filter_length - 1),
                ('fir_inner_state_dict', (module.fir_inner_filter_length or 1) - 1),
            ):
                cache = getattr(state[key], attr, {})
                if layer in cache and cache[layer].shape[-1] < width:
                    cache[layer] = torch.nn.functional.pad(cache[layer], (width-cache[layer].shape[-1], 0))


class CachedCausalPolicy(CausalPolicy):
    @torch.no_grad()
    def sample(self, prefix, length, generator):
        if not 0 < prefix.shape[1] < length:
            raise ValueError('Expected a nonempty prefix shorter than target length')
        tokens = prefix.clone()
        state = self.model.initialize_inference_params(max_seqlen=length)
        state['mha'].max_batch_size = prefix.shape[0]
        ids = torch.tensor(BASE_IDS, device=prefix.device)
        logps, values = [], []
        for position in range(prefix.shape[1], length):
            captured = []
            hook = None
            if self.value_head is not None:
                hook = self.model.norm.register_forward_hook(lambda module, args, out: captured.append(out))
            first = position == prefix.shape[1]
            if not first:
                # Match Vortex Generator's prefill/step offset convention.
                for key in ('mha', 'hcl', 'hcm', 'hcs'):
                    state[key].seqlen_offset = position - 1
            try:
                logits, state = self.model(tokens if first else tokens[:, -1:], inference_params_dict=state)
            finally:
                if hook is not None:
                    hook.remove()
            if first:
                # Vortex step_fir slices the WRONG end of short-filter weights
                # when prefill is shorter than its kernel. Represent the missing
                # causal history explicitly as zeros before the first step.
                pad_fir_caches(self.model, state)
            distribution = logits[:, -1, list(BASE_IDS)].float().log_softmax(-1)
            action = torch.multinomial(distribution.exp(), 1, generator=generator)
            logps.append(distribution.gather(-1, action).squeeze(-1))
            if self.value_head is not None:
                if len(captured) != 1:
                    raise RuntimeError('Expected one causal critic activation')
                values.append(self.value_head(captured[0][:, -1].float()).squeeze(-1))
            tokens = torch.cat((tokens, ids[action]), dim=1)
        return tokens, torch.stack(logps, 1), torch.stack(values, 1) if values else None


def parity_metrics(actual, recomputed):
    error = (actual.float() - recomputed.float()).detach()
    if not torch.isfinite(error).all():
        raise RuntimeError('Nonfinite rollout/training probabilities')
    ratio_error = (error.exp()-1).abs()
    return {'mean_abs_logp_error': error.abs().mean().item(),
            'max_abs_logp_error': error.abs().max().item(),
            'max_ratio_error': ratio_error.max().item()}
