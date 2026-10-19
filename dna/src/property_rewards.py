"""Differentiable Design-stage property rewards for ToDNA supplements.

These rewards consume a soft A/C/G/T distribution ``[batch, length, 4]``.
They are intentionally separate from post-lock diagnostics: a caller must
construct them before search and record their source/configuration hashes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import torch
import torch.nn.functional as F
from torch import nn

from .motif_reward import JasparMotifScorer


def _check_q(q: torch.Tensor) -> None:
    if q.ndim != 3 or q.shape[-1] != 4:
        raise ValueError("reward expects a soft DNA tensor [batch, length, 4]")
    if q.shape[1] < 1:
        raise ValueError("sequence length must be positive")


class GReluAccessibilitySurrogate(nn.Module):
    """Differentiable wrapper for the pinned gReLU accessibility checkpoint.

    The external evaluator uses hard one-hot strings after locking.  This
    wrapper deliberately keeps the same checkpoint and task head but accepts a
    soft ``[batch, length, 4]`` tensor during Design.  Center padding/cropping
    matches the evaluator's sequence-resize convention; no post-lock score is
    read by the optimizer.
    """

    def __init__(self, lightning_model: nn.Module, input_length: int, task_indices: Mapping[str, int] | None = None) -> None:
        super().__init__()
        self.lightning_model = lightning_model.eval()
        for parameter in self.lightning_model.parameters():
            parameter.requires_grad_(False)
        self.input_length = int(input_length)
        self.task_indices = dict(task_indices or {})

    @classmethod
    def from_checkpoint(cls, checkpoint: str | Path, enformer_state_dict: str | Path, device: torch.device | str = "cpu") -> "GReluAccessibilitySurrogate":
        import os
        import grelu.resources
        from grelu.lightning import LightningModel

        class LocalArtifact:
            def download(self, root: str) -> str:
                destination = Path(root) / "human.h5"
                if not destination.exists():
                    os.symlink(Path(enformer_state_dict).resolve(), destination)
                return str(root)

        original = grelu.resources.get_artifact
        grelu.resources.get_artifact = lambda *args, **kwargs: LocalArtifact()
        try:
            model = LightningModel.load_from_checkpoint(str(checkpoint), map_location="cpu")
        finally:
            grelu.resources.get_artifact = original
        train_params = model.data_params.get("train", {})
        input_length = int(train_params.get("seq_len", 200))
        names = [str(value) for value in model.data_params["tasks"]["name"]]
        normalized = {name.lower().replace("-", ""): index for index, name in enumerate(names)}
        return cls(model.to(device), input_length, normalized)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        _check_q(q)
        x = q.transpose(1, 2).contiguous()
        length = x.shape[-1]
        if length > self.input_length:
            start = (length - self.input_length) // 2
            x = x[..., start : start + self.input_length]
        elif length < self.input_length:
            left = (self.input_length - length) // 2
            right = self.input_length - length - left
            x = F.pad(x, (left, right))
        output = self.lightning_model(x)
        if output.ndim == 3:
            output = output.mean(dim=-1)
        if output.ndim != 2:
            raise ValueError(f"accessibility surrogate emitted {tuple(output.shape)}")
        return output


def expected_kmer_profile(q: torch.Tensor, k: int = 3) -> torch.Tensor:
    """Return the differentiable expected k-mer frequency profile."""
    _check_q(q)
    if not 1 <= k <= q.shape[1]:
        raise ValueError("k must be in [1, sequence length]")
    joint = q[:, : q.shape[1] - k + 1]
    for offset in range(1, k):
        joint = joint.unsqueeze(-1) * q[:, offset : q.shape[1] - k + 1 + offset].unsqueeze(-2)
        joint = joint.reshape(q.shape[0], q.shape[1] - k + 1, -1)
    profile = joint.sum(dim=1)
    return profile / profile.sum(dim=-1, keepdim=True).clamp_min(1e-8)


class AccessibilitySurrogateReward(nn.Module):
    """Reward a frozen differentiable accessibility surrogate.

    ``predictor`` must emit either probabilities or logits.  The output is
    selected by ``task_index`` when it has multiple columns.
    """

    def __init__(
        self,
        predictor: nn.Module,
        task_index: int = 0,
        *,
        from_logits: bool = False,
        target: float | None = None,
        huber_delta: float = 0.05,
    ) -> None:
        super().__init__()
        if not 0 <= task_index:
            raise ValueError("task_index must be non-negative")
        if huber_delta <= 0:
            raise ValueError("huber_delta must be positive")
        self.predictor = predictor.eval()
        for parameter in self.predictor.parameters():
            parameter.requires_grad_(False)
        self.task_index = int(task_index)
        self.from_logits = bool(from_logits)
        self.target = None if target is None else float(target)
        self.huber_delta = float(huber_delta)

    def per_sequence(self, q: torch.Tensor) -> torch.Tensor:
        _check_q(q)
        output = self.predictor(q)
        if output.ndim == 1:
            values = output
        elif output.ndim == 2 and self.task_index < output.shape[-1]:
            values = output[:, self.task_index]
        else:
            raise ValueError("accessibility surrogate must emit [batch] or [batch, tasks]")
        if self.from_logits:
            values = values.sigmoid()
        if self.target is None:
            return values
        target = values.new_full(values.shape, self.target)
        return -F.smooth_l1_loss(values, target, beta=self.huber_delta, reduction="none")

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        return self.per_sequence(q).mean()


class AbsoluteSetpointReward(nn.Module):
    """Huber reward for a fixed normalized target-cell activity setpoint."""

    def __init__(
        self,
        predictor: nn.Module,
        target_index: int,
        setpoint: float,
        activity_min: tuple[float, ...],
        activity_max: tuple[float, ...],
        scale: float,
        huber_delta: float = 0.05,
    ) -> None:
        super().__init__()
        if not 0.0 <= setpoint <= 1.0 or scale <= 0 or huber_delta <= 0:
            raise ValueError("invalid setpoint reward parameters")
        if len(activity_min) != len(activity_max) or not 0 <= target_index < len(activity_min):
            raise ValueError("invalid target activity normalization")
        self.predictor = predictor.eval()
        for parameter in self.predictor.parameters():
            parameter.requires_grad_(False)
        self.target_index = int(target_index)
        self.setpoint = float(setpoint)
        self.scale = float(scale)
        self.huber_delta = float(huber_delta)
        self.register_buffer("activity_min", torch.tensor(activity_min, dtype=torch.float32))
        self.register_buffer("activity_max", torch.tensor(activity_max, dtype=torch.float32))

    def per_sequence(self, q: torch.Tensor) -> torch.Tensor:
        _check_q(q)
        activity = self.predictor(q)
        if activity.ndim != 2 or activity.shape[-1] != self.activity_min.numel():
            raise ValueError("setpoint predictor output does not match activity normalization")
        normalized = (activity - self.activity_min.to(activity)) / (self.activity_max.to(activity) - self.activity_min.to(activity))
        error = normalized[:, self.target_index] - self.setpoint
        return -F.smooth_l1_loss(error, torch.zeros_like(error), beta=self.huber_delta, reduction="none") / self.scale

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        return self.per_sequence(q).mean()


class SoftPWMReward(nn.Module):
    """Differentiable soft motif/PWM score for selected motif columns."""

    def __init__(self, scorer: JasparMotifScorer, motif_weights: Mapping[int, float] | None = None) -> None:
        super().__init__()
        self.scorer = scorer.eval()
        for parameter in self.scorer.parameters():
            parameter.requires_grad_(False)
        weights = motif_weights or {}
        self.register_buffer(
            "weights",
            torch.tensor(
                [float(weights.get(index, 1.0)) for index in range(len(self.scorer.motif_ids))],
                dtype=torch.float32,
            ),
        )

    def per_sequence(self, q: torch.Tensor) -> torch.Tensor:
        scores = self.scorer(q)
        weights = self.weights.to(scores.device, scores.dtype)
        return (scores * weights).sum(dim=-1) / weights.abs().sum().clamp_min(1e-8)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        return self.per_sequence(q).mean()


class ThreeMerCompositionReward(nn.Module):
    """Match a frozen 3-mer target profile using a differentiable expectation."""

    def __init__(self, reference_profile: torch.Tensor | list[float] | Path, *, scale: float = 1.0) -> None:
        super().__init__()
        if isinstance(reference_profile, Path):
            payload = json.loads(reference_profile.read_text(encoding="utf-8"))
            reference_profile = payload.get("profile", payload)
        reference = torch.as_tensor(reference_profile, dtype=torch.float32).flatten()
        if reference.numel() != 64 or torch.any(reference < 0):
            raise ValueError("3-mer reference must contain 64 non-negative values")
        reference = reference / reference.sum().clamp_min(1e-8)
        self.register_buffer("reference", reference)
        self.scale = float(scale)

    def per_sequence(self, q: torch.Tensor) -> torch.Tensor:
        profile = expected_kmer_profile(q, 3)
        reference = self.reference.to(profile.device, profile.dtype)
        return -F.smooth_l1_loss(profile, reference.expand_as(profile), beta=0.01, reduction="none").mean(dim=-1) * self.scale

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        return self.per_sequence(q).mean()


def _expected_homopolymer_surrogate(q: torch.Tensor, max_run: int) -> torch.Tensor:
    """Smoothly penalize long same-base runs without hard sequence filtering."""
    same = (q[:, :-1] * q[:, 1:]).sum(dim=-1)
    window = max(int(max_run), 1)
    if same.shape[1] < window:
        return same.new_zeros((q.shape[0],))
    # A run of max_run+1 bases requires max_run adjacent equal-base events.
    # Averaging window products avoids a hard max while preserving gradients at
    # every position.
    return same.unfold(1, window, 1).prod(dim=-1).mean(dim=-1)


class SequenceComplianceReward(nn.Module):
    """Differentiable GC/homopolymer compliance surrogate."""

    def __init__(self, min_gc: float = 0.30, max_gc: float = 0.70, max_homopolymer: int = 6, temperature: float = 0.02) -> None:
        super().__init__()
        if not 0 <= min_gc <= max_gc <= 1 or max_homopolymer < 1 or temperature <= 0:
            raise ValueError("invalid sequence-compliance bounds")
        self.min_gc = float(min_gc)
        self.max_gc = float(max_gc)
        self.max_homopolymer = int(max_homopolymer)
        self.temperature = float(temperature)

    def per_sequence(self, q: torch.Tensor) -> torch.Tensor:
        _check_q(q)
        gc = q[..., 1:3].sum(dim=-1).mean(dim=-1)
        gc_penalty = F.softplus((self.min_gc - gc) / self.temperature) + F.softplus((gc - self.max_gc) / self.temperature)
        run = _expected_homopolymer_surrogate(q, self.max_homopolymer)
        run_penalty = run
        return -(gc_penalty + run_penalty)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        return self.per_sequence(q).mean()


class MultiObjectiveReward(nn.Module):
    """Weighted sum of Design-stage property rewards."""

    def __init__(self, rewards: Mapping[str, nn.Module], weights: Mapping[str, float]) -> None:
        super().__init__()
        if not rewards or set(rewards) != set(weights):
            raise ValueError("rewards and weights must contain the same non-empty keys")
        self.rewards = nn.ModuleDict(rewards)
        self.weights = {name: float(weights[name]) for name in rewards}

    def per_sequence(self, q: torch.Tensor) -> torch.Tensor:
        total = q.new_zeros(q.shape[0])
        for name, reward in self.rewards.items():
            values = reward.per_sequence(q) if hasattr(reward, "per_sequence") else reward(q)
            if values.ndim == 0:
                values = values.expand(q.shape[0])
            total = total + self.weights[name] * values
        return total

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        return self.per_sequence(q).mean()


if __name__ == "__main__":
    q = torch.softmax(torch.randn(2, 32, 4), dim=-1).requires_grad_()
    reward = SequenceComplianceReward()
    reward(q).backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()
    assert expected_kmer_profile(q.detach()).shape == (2, 64)
    print("property_rewards_smoke_ok")
