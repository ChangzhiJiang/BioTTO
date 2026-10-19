"""Hard sequence checks used after test-time optimization."""

from __future__ import annotations

from dataclasses import dataclass

from .dto import BASES, validate_sequence


@dataclass(frozen=True)
class SequenceConstraints:
    min_gc: float = 0.30
    max_gc: float = 0.70
    max_homopolymer: int = 6
    max_evo2_nll: float | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.min_gc <= self.max_gc <= 1:
            raise ValueError("GC bounds must satisfy 0 <= min_gc <= max_gc <= 1")
        if self.max_homopolymer < 1:
            raise ValueError("max_homopolymer must be positive")

    def check(self, sequence: str, evo2_nll: float | None = None) -> tuple[bool, dict[str, float]]:
        sequence = validate_sequence(sequence)
        gc_fraction = sum(base in "GC" for base in sequence) / len(sequence)
        max_run = 1
        current_run = 1
        for previous, current in zip(sequence, sequence[1:]):
            current_run = current_run + 1 if current == previous else 1
            max_run = max(max_run, current_run)

        metrics = {
            "gc_fraction": float(gc_fraction),
            "max_homopolymer": float(max_run),
        }
        accepted = self.min_gc <= gc_fraction <= self.max_gc
        accepted = accepted and max_run <= self.max_homopolymer
        if self.max_evo2_nll is not None:
            if evo2_nll is None:
                raise ValueError("evo2_nll is required when max_evo2_nll is set")
            metrics["evo2_nll"] = float(evo2_nll)
            accepted = accepted and evo2_nll <= self.max_evo2_nll
        return accepted, metrics
