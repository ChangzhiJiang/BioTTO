"""Differentiable JASPAR motif-profile regularization for TTO."""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


def load_meme_pwms(path: Path) -> tuple[list[str], list[torch.Tensor]]:
    motif_ids: list[str] = []
    matrices: list[torch.Tensor] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.startswith("MOTIF "):
            index += 1
            continue
        motif_id = line.split()[1]
        index += 1
        while index < len(lines) and "letter-probability matrix:" not in lines[index]:
            index += 1
        if index >= len(lines):
            raise ValueError(f"missing matrix for motif {motif_id}")
        match = re.search(r"\bw=\s*(\d+)", lines[index])
        if match is None:
            raise ValueError(f"missing width for motif {motif_id}")
        width = int(match.group(1))
        rows = [[float(value) for value in row.split()] for row in lines[index + 1 : index + 1 + width]]
        if len(rows) != width or any(len(row) != 4 for row in rows):
            raise ValueError(f"invalid matrix for motif {motif_id}")
        motif_ids.append(motif_id)
        matrices.append(torch.tensor(rows, dtype=torch.float32))
        index += width + 1
    if len(motif_ids) < 100 or len(set(motif_ids)) != len(motif_ids):
        raise ValueError(f"unexpected MEME motif set: {len(motif_ids)} motifs")
    return motif_ids, matrices


class JasparMotifScorer(nn.Module):
    def __init__(self, meme_path: Path) -> None:
        super().__init__()
        self.motif_ids, matrices = load_meme_pwms(meme_path)
        groups: dict[int, list[tuple[int, torch.Tensor]]] = {}
        for motif_index, matrix in enumerate(matrices):
            pwm = matrix.clamp_min(1e-5)
            pwm = pwm / pwm.sum(dim=1, keepdim=True)
            log_odds = torch.log(pwm / 0.25)
            reverse = log_odds.flip(0)[:, [3, 2, 1, 0]]
            groups.setdefault(matrix.shape[0], []).append(
                (motif_index, torch.stack((log_odds.t(), reverse.t()), dim=0))
            )
        self._group_names: list[str] = []
        concat_order: list[int] = []
        for group_index, (_, entries) in enumerate(sorted(groups.items())):
            entries.sort(key=lambda item: item[0])
            name = f"weights_{group_index}"
            self.register_buffer(name, torch.cat([item[1] for item in entries], dim=0))
            self._group_names.append(name)
            concat_order.extend(item[0] for item in entries)
        reorder = [concat_order.index(index) for index in range(len(self.motif_ids))]
        self.register_buffer("_reorder", torch.tensor(reorder, dtype=torch.long))

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        if q.ndim != 3 or q.shape[-1] != 4:
            raise ValueError("motif scorer expects [batch, length, 4]")
        sequence = q.transpose(1, 2)
        parts = []
        for name in self._group_names:
            values = F.conv1d(sequence, getattr(self, name))
            parts.append(values.amax(dim=-1).reshape(q.shape[0], -1, 2).amax(dim=-1))
        return torch.cat(parts, dim=1).index_select(1, self._reorder)


class NaturalMotifProfileReward(nn.Module):
    def __init__(self, meme_path: Path, reference_path: Path, target: str) -> None:
        super().__init__()
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        motif_ids, _ = load_meme_pwms(meme_path)
        if reference["motif_ids"] != motif_ids:
            raise ValueError("reference motif order does not match MEME file")
        target_reference = reference["targets"][target]
        self.scorer = JasparMotifScorer(meme_path)
        self.register_buffer("reference_mean", torch.tensor(target_reference["score_mean"]))
        self.register_buffer(
            "reference_scale",
            torch.tensor(target_reference["score_std"]).clamp_min(0.25),
        )
        self.reward_scale = float(reference.get("reward_scale", 1.0))

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        return self.per_sequence(q).mean()

    def per_sequence(self, q: torch.Tensor) -> torch.Tensor:
        normalized_error = (self.scorer(q) - self.reference_mean) / self.reference_scale
        return -F.smooth_l1_loss(
            normalized_error,
            torch.zeros_like(normalized_error),
            beta=1.0,
            reduction="none",
        ).mean(dim=1) * self.reward_scale
