"""Evo 2 test-time DNA design with rejection sampling and Top-K selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from collections import Counter
import time
from typing import Callable, Iterable, Sequence
from pathlib import Path

import torch

from .constraints import SequenceConstraints
from .dto import BASES, DtoConfig, Evo2Dto, RewardFn, validate_sequence


@dataclass(frozen=True)
class DesignTarget:
    """Cell-type objective interpreted by a multi-output activity predictor."""

    target_index: int
    off_target_indices: tuple[int, ...]
    name: str = "target"


@dataclass(frozen=True)
class SearchConfig:
    num_initial: int = 8
    top_k: int = 4
    initial_temperature: float = 1.0
    initial_top_k: int = 4
    initial_top_p: float = 1.0
    dto: DtoConfig = field(default_factory=DtoConfig)
    constraints: SequenceConstraints = field(default_factory=SequenceConstraints)
    prior_penalty: float = 0.20


@dataclass
class CandidateRecord:
    initial_sequence: str
    optimized_sequence: str
    initial_reward: float
    optimized_reward: float
    evo2_log_probability: float
    evo2_nll: float
    ranking_score: float
    accepted: bool
    rejection_reason: str = ""
    constraint_metrics: dict[str, float] = field(default_factory=dict)
    hamming_edits: int = 0
    trust_kl: float = 0.0
    expected_edit_fraction: float = 0.0
    history: list[dict[str, float]] = field(default_factory=list)
    design_oracle_calls: int = 0
    wall_clock_seconds: float = 0.0
    peak_gpu_memory_bytes: int = 0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class Evo2TestTimeDesigner:
    """Generate candidates with Evo 2 and optimize each candidate at inference time."""

    def __init__(
        self,
        dto: Evo2Dto | None = None,
        model_name: str = "evo2_7b",
        lora_adapter: str | Path | None = None,
    ) -> None:
        if dto is not None and lora_adapter is not None:
            raise ValueError("pass either dto or lora_adapter, not both")
        self.dto = dto or Evo2Dto(model_name, lora_adapter=lora_adapter)
        self.last_search_summary: dict[str, object] = {}

    @staticmethod
    def _fixed_length(sequence: str, length: int, prompt: str) -> str:
        sequence = validate_sequence(sequence)
        prompt = validate_sequence(prompt) if prompt else ""
        # Vortex 1.1 returns only the newly generated suffix, while some Evo 2
        # releases return prompt plus suffix. Normalize both to one sequence.
        if prompt and len(sequence) == length - len(prompt):
            return (prompt + sequence)[:length]
        if sequence.startswith(prompt) and len(sequence) >= length:
            return sequence[:length]
        if len(sequence) >= length:
            return sequence[:length]
        raise ValueError(f"Evo 2 returned {len(sequence)} bases; expected at least {length}")

    def sample_initial(
        self,
        prompt: str,
        length: int,
        config: SearchConfig,
    ) -> list[str]:
        prompt = validate_sequence(prompt)
        if length < len(prompt):
            raise ValueError("length must be at least the prompt length")
        generation = self.dto.evo2.generate(
            prompt_seqs=[prompt] * config.num_initial,
            # Vortex can return a few fewer decoded bases on the non-Flash
            # short-sequence fallback. Request a small margin and truncate.
            n_tokens=length - len(prompt) + 16,
            temperature=config.initial_temperature,
            top_k=config.initial_top_k,
            top_p=config.initial_top_p,
            batched=True,
            cached_generation=True,
            verbose=0,
        )
        # Cached generation creates rotary/state buffers under inference mode.
        # Convert only those newly created buffers before optimizing inputs.
        self.dto._materialize_autograd_tensors(self.dto.model)
        # Evo 2 0.6/Vortex returns GenerationOutput; older releases returned
        # ``(sequences, scores)``. Keep both forms usable for reproducibility.
        sequences = (
            generation.sequences
            if hasattr(generation, "sequences")
            else generation[0]
        )
        return [self._fixed_length(sequence, length, prompt) for sequence in sequences]

    def one_hot(self, sequence: str) -> torch.Tensor:
        sequence = validate_sequence(sequence)
        indices = torch.tensor(
            [BASES.index(base) for base in sequence],
            dtype=torch.long,
            device=self.dto.device,
        )
        return torch.nn.functional.one_hot(indices, num_classes=4).float().unsqueeze(0)

    def search(
        self,
        prompt: str,
        length: int,
        reward_fn: RewardFn,
        config: SearchConfig | None = None,
        initial_sequences: Sequence[str] | None = None,
    ) -> list[CandidateRecord]:
        config = config or SearchConfig()
        if initial_sequences is None:
            initial_sequences = self.sample_initial(prompt, length, config)
        else:
            initial_sequences = [
                validate_sequence(sequence) for sequence in initial_sequences
            ]
            if not initial_sequences:
                raise ValueError("initial_sequences must not be empty")
            if any(len(sequence) != length for sequence in initial_sequences):
                raise ValueError("all initial sequences must match the requested length")
        records: list[CandidateRecord] = []
        seen: set[str] = set()
        rejection_counts: Counter[str] = Counter()
        duplicate_count = 0
        total_initial = len(initial_sequences)
        device = torch.device(self.dto.device)
        for candidate_index, initial_sequence in enumerate(initial_sequences, start=1):
            started = time.perf_counter()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            result = self.dto.optimize(initial_sequence, reward_fn, config.dto)
            elapsed = time.perf_counter() - started
            peak_memory = (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda" else 0
            )
            optimized_sequence = validate_sequence(result.optimized_sequence)
            if optimized_sequence in seen:
                duplicate_count += 1
                continue
            seen.add(optimized_sequence)

            # Rank and reject only after hard projection and discrete Evo 2 scoring.
            log_probability = self.dto.score_sequence(optimized_sequence)
            evo2_nll = -log_probability
            accepted, constraint_metrics = config.constraints.check(
                optimized_sequence, evo2_nll=evo2_nll
            )
            reason = "" if accepted else self._rejection_reason(constraint_metrics, config.constraints)
            if not accepted:
                rejection_counts[reason] += 1
            ranking_score = result.final_reward - config.prior_penalty * evo2_nll
            records.append(
                CandidateRecord(
                    initial_sequence=initial_sequence,
                    optimized_sequence=optimized_sequence,
                    initial_reward=result.initial_reward,
                    optimized_reward=result.final_reward,
                    evo2_log_probability=log_probability,
                    evo2_nll=evo2_nll,
                    ranking_score=ranking_score,
                    accepted=accepted,
                    rejection_reason=reason,
                    constraint_metrics=constraint_metrics,
                    hamming_edits=result.hamming_edits,
                    trust_kl=result.final_trust_kl,
                    expected_edit_fraction=result.final_expected_edit_fraction,
                    history=result.history,
                    design_oracle_calls=int(config.dto.steps),
                    wall_clock_seconds=float(elapsed),
                    peak_gpu_memory_bytes=peak_memory,
                )
            )
            if candidate_index % 16 == 0 or candidate_index == total_initial:
                print(
                    f"SEARCH_PROGRESS attempted={candidate_index}/{total_initial} "
                    f"unique={len(records)} accepted={sum(record.accepted for record in records)}",
                    flush=True,
                )

        accepted_records = [record for record in records if record.accepted]
        accepted_records.sort(key=lambda record: record.ranking_score, reverse=True)
        self.last_search_summary = {
            "attempted": len(initial_sequences),
            "unique_projected": len(records),
            "accepted": len(accepted_records),
            "duplicates": duplicate_count,
            "rejection_counts": dict(rejection_counts),
        }
        return accepted_records[: config.top_k]

    @staticmethod
    def _rejection_reason(metrics: dict[str, float], constraints: SequenceConstraints) -> str:
        reasons: list[str] = []
        gc = metrics["gc_fraction"]
        if gc < constraints.min_gc or gc > constraints.max_gc:
            reasons.append("gc_range")
        if metrics["max_homopolymer"] > constraints.max_homopolymer:
            reasons.append("homopolymer")
        if constraints.max_evo2_nll is not None and metrics.get("evo2_nll", 0) > constraints.max_evo2_nll:
            reasons.append("evo2_nll")
        return ",".join(reasons) or "constraint"
