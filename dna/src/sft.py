"""Data and loss helpers for Evo 2 enhancer-domain SFT."""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch.nn import functional as F


BASES = "ACGT"
COMPLEMENT = str.maketrans("ACGT", "TGCA")
BASE_TOKEN_IDS = (ord("A"), ord("C"), ord("G"), ord("T"))


def reverse_complement(sequence: str) -> str:
    return sequence.translate(COMPLEMENT)[::-1]


def validate_sequences(sequences: Iterable[str], length: int = 200) -> list[str]:
    validated = []
    for sequence in sequences:
        sequence = sequence.upper()
        if len(sequence) != length or set(sequence) - set(BASES):
            raise ValueError(f"invalid {length} bp DNA sequence")
        validated.append(sequence)
    if not validated:
        raise ValueError("sequence corpus is empty")
    return validated


def read_sequence_jsonl(path: Path, length: int = 200) -> list[str]:
    sequences = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            sequences.append(record["sequence"])
    return validate_sequences(sequences, length)


def tokenize_sequences(sequences: Sequence[str], device: torch.device | str) -> torch.Tensor:
    if not sequences:
        raise ValueError("cannot tokenize an empty batch")
    length = len(sequences[0])
    if any(len(sequence) != length for sequence in sequences):
        raise ValueError("all sequences in a batch must have equal length")
    raw = bytearray("".join(sequences), "ascii")
    return torch.tensor(raw, dtype=torch.long, device=device).reshape(len(sequences), length)


def causal_lm_losses(logits: torch.Tensor, token_ids: torch.Tensor) -> dict[str, torch.Tensor]:
    if logits.ndim != 3 or token_ids.ndim != 2 or logits.shape[:2] != token_ids.shape:
        raise ValueError("logits and token IDs have incompatible shapes")
    predictions = logits[:, :-1].float()
    targets = token_ids[:, 1:]
    full_nll = F.cross_entropy(
        predictions.reshape(-1, predictions.shape[-1]), targets.reshape(-1)
    )
    base_ids = torch.tensor(BASE_TOKEN_IDS, device=logits.device, dtype=torch.long)
    base_logits = predictions.index_select(-1, base_ids)
    base_targets = torch.empty_like(targets)
    matched = torch.zeros_like(targets, dtype=torch.bool)
    for index, token_id in enumerate(BASE_TOKEN_IDS):
        mask = targets == token_id
        base_targets[mask] = index
        matched |= mask
    if not bool(matched.all()):
        raise ValueError("targets contain non-ACGT tokens")
    dna_nll = F.cross_entropy(base_logits.reshape(-1, 4), base_targets.reshape(-1))
    return {"full_vocab_nll": full_nll, "dna_conditional_nll": dna_nll}


def deterministic_orientation(sequence: str, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}\0{sequence}".encode()).digest()
    return reverse_complement(sequence) if digest[0] & 1 else sequence


def random_orientations(
    sequences: Sequence[str], rng: random.Random, probability: float = 0.5
) -> list[str]:
    return [
        reverse_complement(sequence) if rng.random() < probability else sequence
        for sequence in sequences
    ]


def perplexity(nll: float) -> float:
    return math.exp(min(nll, 20.0))
