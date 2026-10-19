"""Differentiable reward functions for cell-type-specific DNA design."""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn


CTRL_DNA_ACTIVITY_MIN = (-5.857445, -6.051336, -7.283977)
CTRL_DNA_ACTIVITY_MAX = (10.781755, 10.992575, 12.888308)


class MultiCellActivityReward(nn.Module):
    """MinGap reward over a differentiable multi-cell activity predictor.

    The predictor must map ``[batch, length, 4]`` DNA tensors to
    ``[batch, number_of_cells]`` activity scores. The smooth maximum gives
    gradients to all off-target cells while approaching a hard maximum as the
    temperature decreases.
    """

    def __init__(
        self,
        predictor: nn.Module,
        target_index: int,
        off_target_indices: Iterable[int],
        smooth_max_temperature: float = 0.1,
    ) -> None:
        super().__init__()
        self.predictor = predictor
        self.predictor.eval()
        for parameter in self.predictor.parameters():
            parameter.requires_grad_(False)
        self.target_index = target_index
        self.off_target_indices = tuple(off_target_indices)
        self.smooth_max_temperature = smooth_max_temperature

        if not self.off_target_indices:
            raise ValueError("At least one off-target cell is required")
        if target_index in self.off_target_indices:
            raise ValueError("Target cell cannot also be an off-target cell")
        if smooth_max_temperature <= 0:
            raise ValueError("smooth_max_temperature must be positive")

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        return self.per_sequence(q).mean()

    def per_sequence(self, q: torch.Tensor) -> torch.Tensor:
        activity = self.predictor(q)
        if activity.ndim != 2:
            raise ValueError("predictor output must have shape [batch, cells]")

        target = activity[:, self.target_index]
        off_target = activity[:, self.off_target_indices]
        temperature = self.smooth_max_temperature
        smooth_max = temperature * torch.logsumexp(off_target / temperature, dim=-1)
        return target - smooth_max


class NormalizedDeltaRReward(nn.Module):
    """Ctrl-DNA-style target minus mean off-target reward.

    Activity predictions are independently min-max normalized with the public
    Ctrl-DNA enhancer constants. Values are intentionally not clipped, matching
    the released Ctrl-DNA normalization code.
    """

    def __init__(
        self,
        predictor: nn.Module,
        target_index: int,
        off_target_indices: Iterable[int],
        activity_min: Iterable[float] = CTRL_DNA_ACTIVITY_MIN,
        activity_max: Iterable[float] = CTRL_DNA_ACTIVITY_MAX,
    ) -> None:
        super().__init__()
        self.predictor = predictor
        self.predictor.eval()
        for parameter in self.predictor.parameters():
            parameter.requires_grad_(False)
        self.target_index = target_index
        self.off_target_indices = tuple(off_target_indices)
        self.activity_min = tuple(float(value) for value in activity_min)
        self.activity_max = tuple(float(value) for value in activity_max)

        if not self.off_target_indices:
            raise ValueError("At least one off-target cell is required")
        if target_index in self.off_target_indices:
            raise ValueError("Target cell cannot also be an off-target cell")
        if len(self.activity_min) != len(self.activity_max):
            raise ValueError("activity min/max lengths must match")
        if target_index >= len(self.activity_min) or max(self.off_target_indices) >= len(
            self.activity_min
        ):
            raise ValueError("activity normalization does not cover all cell indices")
        if any(high <= low for low, high in zip(self.activity_min, self.activity_max)):
            raise ValueError("each activity maximum must exceed its minimum")

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        return self.per_sequence(q).mean()

    def per_sequence(self, q: torch.Tensor) -> torch.Tensor:
        activity = self.predictor(q)
        if activity.ndim != 2 or activity.shape[-1] != len(self.activity_min):
            raise ValueError(
                f"predictor output must have shape [batch, {len(self.activity_min)}]"
            )
        low = activity.new_tensor(self.activity_min)
        high = activity.new_tensor(self.activity_max)
        normalized = (activity - low) / (high - low)
        target = normalized[:, self.target_index]
        off_target = normalized[:, self.off_target_indices]
        return target - off_target.mean(dim=-1)


class NormalizedTargetActivityReward(nn.Module):
    """Absolute target-cell activity on the public normalized scale.

    This is the single-cell activity arm for reward-scaling experiments.  The
    normalization keeps the magnitude comparable to the Delta-R arms while
    deliberately removing the off-target term.  The predictor is frozen in
    exactly the same way as the other activity rewards.
    """

    def __init__(
        self,
        predictor: nn.Module,
        target_index: int,
        activity_min: Iterable[float] = CTRL_DNA_ACTIVITY_MIN,
        activity_max: Iterable[float] = CTRL_DNA_ACTIVITY_MAX,
    ) -> None:
        super().__init__()
        self.predictor = predictor
        self.predictor.eval()
        for parameter in self.predictor.parameters():
            parameter.requires_grad_(False)
        self.target_index = int(target_index)
        self.activity_min = tuple(float(value) for value in activity_min)
        self.activity_max = tuple(float(value) for value in activity_max)
        if len(self.activity_min) != len(self.activity_max):
            raise ValueError("activity min/max lengths must match")
        if not 0 <= self.target_index < len(self.activity_min):
            raise ValueError("target index is outside activity normalization")
        if any(high <= low for low, high in zip(self.activity_min, self.activity_max)):
            raise ValueError("each activity maximum must exceed its minimum")

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        return self.per_sequence(q).mean()

    def per_sequence(self, q: torch.Tensor) -> torch.Tensor:
        activity = self.predictor(q)
        if activity.ndim != 2 or activity.shape[-1] != len(self.activity_min):
            raise ValueError(
                f"predictor output must have shape [batch, {len(self.activity_min)}]"
            )
        low = activity.new_tensor(self.activity_min)
        high = activity.new_tensor(self.activity_max)
        normalized = (activity - low) / (high - low)
        return normalized[:, self.target_index]


class NormalizedDeltaRV2Reward(NormalizedDeltaRReward):
    """Normalized target minus a smooth maximum of normalized off-targets."""

    def __init__(self, *args, tau: float = 0.10, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if tau <= 0:
            raise ValueError("tau must be positive")
        self.tau = float(tau)

    def per_sequence(self, q: torch.Tensor) -> torch.Tensor:
        activity = self.predictor(q)
        if activity.ndim != 2 or activity.shape[-1] != len(self.activity_min):
            raise ValueError(
                f"predictor output must have shape [batch, {len(self.activity_min)}]"
            )
        low = activity.new_tensor(self.activity_min)
        high = activity.new_tensor(self.activity_max)
        normalized = (activity - low) / (high - low)
        target = normalized[:, self.target_index]
        off_target = normalized[:, self.off_target_indices]
        smooth_max = self.tau * torch.logsumexp(off_target / self.tau, dim=-1)
        return target - smooth_max


class StandardizedReward(nn.Module):
    """Apply fixed initial-pool robust calibration to a per-sequence reward."""

    def __init__(self, reward: nn.Module, center: float, scale: float) -> None:
        super().__init__()
        if scale <= 0:
            raise ValueError("calibration scale must be positive")
        self.reward = reward
        self.center = float(center)
        self.scale = float(scale)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        return self.per_sequence(q).mean()

    def per_sequence(self, q: torch.Tensor) -> torch.Tensor:
        values = self.reward.per_sequence(q)
        return (values - self.center) / self.scale
