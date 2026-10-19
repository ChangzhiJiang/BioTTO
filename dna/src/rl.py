"""Sequence-edit PPO and GRPO primitives for LoRA-adapted Evo 2.

The policy is a contextual edit distribution. Given a fixed starting DNA
sequence, Evo 2's causal logits define an independent categorical action at
each editable position. A stay bias keeps the policy local to the shared
Design-oracle start while LoRA parameters remain the only trainable weights.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F


BASE_TOKEN_IDS = (ord("A"), ord("C"), ord("G"), ord("T"))


def edit_policy_logits(
    model_logits: torch.Tensor,
    state_tokens: torch.Tensor,
    stay_bias: float,
    temperature: float,
) -> torch.Tensor:
    """Convert causal LM outputs to per-position A/C/G/T edit logits.

    Position ``i - 1`` predicts the editable base at position ``i``. Position
    zero is kept fixed and excluded from all policy losses.
    """
    if model_logits.ndim != 3 or state_tokens.ndim != 2:
        raise ValueError("model logits and state tokens must be rank 3 and 2")
    if model_logits.shape[:2] != state_tokens.shape:
        raise ValueError("model logits and state tokens have incompatible shapes")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    base_ids = torch.tensor(
        BASE_TOKEN_IDS, device=model_logits.device, dtype=torch.long
    )
    causal = model_logits.index_select(-1, base_ids).float()
    policy = torch.zeros_like(causal)
    if state_tokens.shape[1] > 1:
        policy[:, 1:] = causal[:, :-1]

    state_bases = torch.empty_like(state_tokens)
    matched = torch.zeros_like(state_tokens, dtype=torch.bool)
    for index, token_id in enumerate(BASE_TOKEN_IDS):
        mask = state_tokens == token_id
        state_bases[mask] = index
        matched |= mask
    if not bool(matched.all()):
        raise ValueError("state contains non-ACGT tokens")
    policy.scatter_add_(
        -1,
        state_bases.unsqueeze(-1),
        torch.full_like(state_bases.unsqueeze(-1), float(stay_bias), dtype=policy.dtype),
    )
    return policy / temperature


def sample_edit_actions(
    policy_logits: torch.Tensor,
    state_base_indices: torch.Tensor,
    group_size: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample ``group_size`` parallel edited sequences per state."""
    if policy_logits.ndim != 3 or policy_logits.shape[-1] != 4:
        raise ValueError("policy logits must have shape [batch, length, 4]")
    if state_base_indices.shape != policy_logits.shape[:2]:
        raise ValueError("state base indices do not match policy logits")
    if group_size <= 0:
        raise ValueError("group size must be positive")
    batch, length, _ = policy_logits.shape
    probabilities = F.softmax(policy_logits, dim=-1)
    draws = torch.multinomial(
        probabilities.reshape(batch * length, 4),
        group_size,
        replacement=True,
        generator=generator,
    )
    actions = draws.reshape(batch, length, group_size).permute(0, 2, 1)
    actions[:, :, 0] = state_base_indices[:, 0].unsqueeze(1)
    return actions


def action_log_probs(
    policy_logits: torch.Tensor, actions: torch.Tensor
) -> torch.Tensor:
    """Return editable-position token log probabilities as ``[B, G, L-1]``."""
    if actions.ndim != 3 or actions.shape[0] != policy_logits.shape[0]:
        raise ValueError("actions must have shape [batch, group, length]")
    if actions.shape[2] != policy_logits.shape[1]:
        raise ValueError("action length does not match policy logits")
    log_probs = F.log_softmax(policy_logits[:, 1:], dim=-1)
    expanded = log_probs.unsqueeze(1).expand(-1, actions.shape[1], -1, -1)
    return expanded.gather(-1, actions[:, :, 1:].unsqueeze(-1)).squeeze(-1)


def categorical_kl(
    policy_logits: torch.Tensor, reference_logits: torch.Tensor
) -> torch.Tensor:
    """Mean exact categorical KL(current || reference), excluding base zero."""
    if policy_logits.shape != reference_logits.shape:
        raise ValueError("policy and reference logits must have equal shapes")
    current_log = F.log_softmax(policy_logits[:, 1:], dim=-1)
    reference_log = F.log_softmax(reference_logits[:, 1:], dim=-1)
    current = current_log.exp()
    return (current * (current_log - reference_log)).sum(dim=-1).mean()


def expected_edit_fraction(
    policy_logits: torch.Tensor, state_base_indices: torch.Tensor
) -> torch.Tensor:
    """Expected edited fraction under the current policy."""
    probabilities = F.softmax(policy_logits[:, 1:], dim=-1)
    keep = probabilities.gather(
        -1, state_base_indices[:, 1:].unsqueeze(-1)
    ).squeeze(-1)
    return (1.0 - keep).mean()


def policy_entropy(policy_logits: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(policy_logits[:, 1:], dim=-1)
    probabilities = log_probs.exp()
    return -(probabilities * log_probs).sum(dim=-1).mean()


def clipped_policy_loss(
    current_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_epsilon: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Token-level PPO/GRPO clipped surrogate for terminal sequence rewards."""
    if current_log_probs.shape != old_log_probs.shape:
        raise ValueError("current and old log probabilities must match")
    if advantages.shape != current_log_probs.shape[:2]:
        raise ValueError("advantages must have shape [batch, group]")
    if not 0 < clip_epsilon < 1:
        raise ValueError("clip epsilon must be between zero and one")
    log_ratio = (current_log_probs - old_log_probs).clamp(-12.0, 12.0)
    ratio = log_ratio.exp()
    advantage = advantages.unsqueeze(-1)
    unclipped = ratio * advantage
    clipped = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantage
    loss = -torch.minimum(unclipped, clipped).mean()
    diagnostics = {
        "ratio_mean": ratio.mean().detach(),
        "clip_fraction": ((ratio - 1.0).abs() > clip_epsilon).float().mean().detach(),
        "approx_kl_old": (old_log_probs - current_log_probs).mean().detach(),
    }
    return loss, diagnostics


def normalize_advantages(values: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    if values.numel() < 2:
        return values - values.mean()
    return (values - values.mean()) / values.std(unbiased=True).clamp_min(epsilon)


def grpo_advantages(rewards: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """Standardize rewards independently within each rollout group."""
    if rewards.ndim != 2 or rewards.shape[1] < 2:
        raise ValueError("GRPO rewards must have shape [batch, group>=2]")
    center = rewards.mean(dim=1, keepdim=True)
    scale = rewards.std(dim=1, unbiased=True, keepdim=True).clamp_min(epsilon)
    return (rewards - center) / scale


def ppo_advantages(
    rewards: torch.Tensor, baselines: torch.Tensor
) -> torch.Tensor:
    """Compute globally normalized contextual-bandit PPO advantages."""
    if baselines.ndim != 1 or baselines.shape[0] != rewards.shape[0]:
        raise ValueError("one baseline is required for each state")
    return normalize_advantages(rewards - baselines.unsqueeze(1))

