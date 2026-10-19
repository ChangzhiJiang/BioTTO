"""Causal, ACGT-constrained PPO/GRPO primitives (not contextual editing)."""
import torch

BASE_IDS = (65, 67, 71, 84)


def gae(rewards, values, gamma=1.0, lam=0.95):
    """Fixed-length, genuinely terminal episodes; no bootstrap past termination."""
    advantage = torch.zeros_like(values)
    carry = torch.zeros_like(values[:, 0])
    for t in reversed(range(values.shape[1])):
        next_value = values[:, t + 1] if t + 1 < values.shape[1] else 0.0
        carry = rewards[:, t] + gamma * next_value - values[:, t] + gamma * lam * carry
        advantage[:, t] = carry
    return advantage, advantage + values


def group_advantages(rewards, group_size, eps=1e-8):
    grouped = rewards.reshape(-1, group_size)
    return ((grouped - grouped.mean(-1, keepdim=True)) /
            grouped.std(-1, unbiased=False, keepdim=True).clamp_min(eps)).flatten()


def policy_loss(logp, old_logp, advantages, clip=0.2):
    ratio = (logp - old_logp).exp()
    return -torch.minimum(ratio * advantages, ratio.clamp(1-clip, 1+clip) * advantages).mean()


def value_loss(values, old_values, returns, clip=0.2):
    clipped = old_values + (values - old_values).clamp(-clip, clip)
    return 0.5 * torch.maximum((values-returns).square(), (clipped-returns).square()).mean()


def exact_kl(log_distribution, reference):
    return (log_distribution.exp() * (log_distribution - reference)).sum(-1).mean()


class CausalPolicy:
    def __init__(self, model, value_head=None):
        self.model, self.value_head = model, value_head

    def forward(self, tokens):
        captured = []
        hook = self.model.norm.register_forward_hook(lambda module, args, out: captured.append(out))
        try:
            logits, _ = self.model(tokens)
        finally:
            hook.remove()
        distribution = logits[..., list(BASE_IDS)].float().log_softmax(-1)
        values = None
        if self.value_head is not None:
            if len(captured) != 1:
                raise RuntimeError('Expected one final norm activation for causal critic')
            values = self.value_head(captured[0].float()).squeeze(-1)
        return distribution, values

    def action(self, tokens, position):
        """Recompute exactly the prefix used during rollout (also in low precision)."""
        distribution, values = self.forward(tokens[:, :position])
        distribution = distribution[:, -1:]
        ids = torch.tensor(BASE_IDS, device=tokens.device)
        action = tokens[:, position:position+1]
        indices = (action[..., None] == ids).long().argmax(-1)
        logp = distribution.gather(-1, indices[..., None]).squeeze(-1)
        return distribution, logp, None if values is None else values[:, -1:]

    def teacher(self, tokens, prefix_length):
        distribution, values = self.forward(tokens[:, :-1])
        distribution = distribution[:, prefix_length-1:]
        actions = tokens[:, prefix_length:]
        ids = torch.tensor(BASE_IDS, device=tokens.device)
        if not (actions[..., None] == ids).any(-1).all():
            raise ValueError('Non-ACGT action')
        indices = (actions[..., None] == ids).long().argmax(-1)
        logp = distribution.gather(-1, indices[..., None]).squeeze(-1)
        return distribution, logp, None if values is None else values[:, prefix_length-1:]

    @torch.no_grad()
    def sample(self, prefix, length, generator):
        tokens = prefix.clone()
        logps, values = [], []
        ids = torch.tensor(BASE_IDS, device=tokens.device)
        while tokens.shape[1] < length:
            distribution, value = self.forward(tokens)
            last = distribution[:, -1]
            action = torch.multinomial(last.exp(), 1, generator=generator)
            logps.append(last.gather(-1, action).squeeze(-1))
            if value is not None:
                values.append(value[:, -1])
            tokens = torch.cat((tokens, ids[action]), dim=1)
        return tokens, torch.stack(logps, 1), torch.stack(values, 1) if values else None
