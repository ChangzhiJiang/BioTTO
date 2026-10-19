#!/usr/bin/env python3
"""Run fixed-pool search configurations with one Evo 2 model load.

This script intentionally loads only the Design-Model. Evaluation-only models
must be run later, after a configuration has been locked.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import re
import statistics
from pathlib import Path

import torch

from src import (
    CandidateRecord,
    DtoConfig,
    Evo2TestTimeDesigner,
    MultiCellActivityReward,
    SearchConfig,
    SequenceConstraints,
    load_activity_checkpoint,
)


CELL_NAMES = ("K562", "HepG2", "SK-N-SH")
SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


def load_sequences(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    sequences = payload.get("sequences") if isinstance(payload, dict) else payload
    if not isinstance(sequences, list) or not all(
        isinstance(sequence, str) for sequence in sequences
    ):
        raise ValueError("initial sequence file must contain a JSON string list")
    if not sequences:
        raise ValueError("initial sequence pool must not be empty")
    return sequences


def load_configs(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    configs = payload.get("configs") if isinstance(payload, dict) else payload
    if not isinstance(configs, list) or not configs:
        raise ValueError("config file must contain a non-empty 'configs' list")
    for config in configs:
        if not isinstance(config, dict):
            raise ValueError("each search config must be a JSON object")
        name = config.get("name")
        if not isinstance(name, str) or not SAFE_NAME.fullmatch(name):
            raise ValueError("each config needs a filesystem-safe name")
    return configs


def mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def candidate_summary(records: list[CandidateRecord]) -> dict[str, float | int | None]:
    sequences = [record.optimized_sequence for record in records]
    pairwise = [
        sum(a != b for a, b in zip(first, second)) / len(first)
        for first, second in itertools.combinations(sequences, 2)
    ]
    positional_entropies: list[float] = []
    if sequences:
        for position in zip(*sequences):
            entropy = 0.0
            for base in "ACGT":
                probability = position.count(base) / len(position)
                if probability:
                    entropy -= probability * math.log2(probability)
            positional_entropies.append(entropy)
    reward_changes = [
        record.optimized_reward - record.initial_reward for record in records
    ]
    return {
        "returned": len(records),
        "mean_design_reward_change": mean(reward_changes),
        "median_design_reward_change": (
            statistics.median(reward_changes) if reward_changes else None
        ),
        "mean_evo2_nll": mean([record.evo2_nll for record in records]),
        "mean_hamming_edits": mean([float(record.hamming_edits) for record in records]),
        "mean_trust_kl": mean([record.trust_kl for record in records]),
        "mean_expected_edit_fraction": mean(
            [record.expected_edit_fraction for record in records]
        ),
        "mean_pairwise_hamming_fraction": mean(pairwise),
        "mean_positional_entropy_bits": mean(positional_entropies),
    }


def build_search_config(
    values: dict[str, object], num_initial: int, top_k: int
) -> SearchConfig:
    prior_weight = float(values.get("prior_weight", 0.5))
    return SearchConfig(
        num_initial=num_initial,
        top_k=top_k,
        dto=DtoConfig(
            steps=int(values.get("steps", 5)),
            learning_rate=float(values.get("learning_rate", 0.1)),
            temperature=float(values.get("temperature", 0.7)),
            reward_weight=float(values.get("reward_weight", 1.0)),
            prior_weight=prior_weight,
            gradient_clip_norm=float(values.get("gradient_clip_norm", 5.0)),
            initial_sequence_margin=float(values.get("initial_margin", 0.25)),
            trust_region_weight=float(values.get("trust_region_weight", 1.0)),
            edit_penalty_weight=float(values.get("edit_penalty_weight", 2.0)),
            max_edits=(
                int(values["max_edits"])
                if values.get("max_edits") is not None
                else None
            ),
            initial_logit_noise_std=float(values.get("trajectory_noise_std", 0.0)),
            initial_logit_noise_seed=(
                int(values["trajectory_seed_base"])
                if values.get("trajectory_seed_base") is not None
                else None
            ),
            audit_hashes=bool(values.get("audit_hashes", False)),
        ),
        constraints=SequenceConstraints(
            min_gc=float(values.get("min_gc", 0.30)),
            max_gc=float(values.get("max_gc", 0.70)),
            max_homopolymer=int(values.get("max_homopolymer", 6)),
            max_evo2_nll=(
                float(values["max_evo2_nll"])
                if values.get("max_evo2_nll") is not None
                else None
            ),
        ),
        prior_penalty=float(values.get("ranking_prior_penalty", prior_weight)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design-checkpoint", type=Path, required=True)
    parser.add_argument("--initial-sequences", type=Path, required=True)
    parser.add_argument("--configs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--targets", nargs="+", choices=CELL_NAMES, default=CELL_NAMES)
    parser.add_argument("--length", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=128)
    parser.add_argument(
        "--max-initial",
        type=int,
        help="Use the first N sequences from the fixed pool for development runs.",
    )
    parser.add_argument("--seed", type=int, default=20240827)
    args = parser.parse_args()

    sequences = load_sequences(args.initial_sequences)
    if args.max_initial is not None:
        if args.max_initial < 1:
            raise ValueError("--max-initial must be positive")
        sequences = sequences[: args.max_initial]
    if any(len(sequence) != args.length for sequence in sequences):
        raise ValueError("all initial sequences must match --length")
    configs = load_configs(args.configs)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    designer = Evo2TestTimeDesigner(model_name="evo2_7b")
    designer.dto.disable_flash_attention()
    predictor = load_activity_checkpoint(
        str(args.design_checkpoint), device=designer.dto.device
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for target in args.targets:
        target_index = CELL_NAMES.index(target)
        reward = MultiCellActivityReward(
            predictor=predictor,
            target_index=target_index,
            off_target_indices=[index for index in range(3) if index != target_index],
            smooth_max_temperature=0.1,
        )
        for values in configs:
            name = str(values["name"])
            search_config = build_search_config(
                values, num_initial=len(sequences), top_k=args.top_k
            )
            records = designer.search(
                prompt="ACGT",
                length=args.length,
                reward_fn=reward,
                config=search_config,
                initial_sequences=sequences,
            )
            payload = {
                "method": "evo2_7b_nabla_test_time_design",
                "selection_boundary": "design_model_evo2_and_hard_constraints_only",
                "target": target,
                "cell_order": CELL_NAMES,
                "seed": args.seed,
                "length": args.length,
                "num_initial": len(sequences),
                "top_k": args.top_k,
                "initial_sequences_file": str(args.initial_sequences),
                "config_name": name,
                "config": values,
                "search_summary": designer.last_search_summary,
                "internal_summary": candidate_summary(records),
                "candidates": [record.to_dict() for record in records],
            }
            target_slug = target.lower().replace("-", "")
            output = args.output_dir / f"{target_slug}_{name}_seed{args.seed}.json"
            output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            print(
                target,
                name,
                json.dumps(payload["internal_summary"], sort_keys=True),
                flush=True,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
