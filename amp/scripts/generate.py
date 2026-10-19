#!/usr/bin/env python3
"""Run resumable, config-driven PeptideDTO generation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from peptidedto.benchmark import atomic_write_json, load_records, sha256_file, validate_records
from peptidedto.config import load_target_config


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--lora-checkpoint", default=None)
    parser.add_argument("--without-sft", action="store_true")
    parser.add_argument("--data-csv", default="data/processed/mulit_peptide_train.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-sequences", type=int, default=None)
    parser.add_argument("--shard-size", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--model-dtype",
        default=None,
        choices=("float32", "float16", "bfloat16"),
        help="Optional runtime override for the model dtype.",
    )
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=0,
        help="Offset added to the configured seed for this independent worker.",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--hard-prior-length-weight",
        type=float,
        default=None,
        help="Optional override for the length-prior coefficient in hard acceptance.",
    )
    parser.add_argument(
        "--hard-charge-coeff",
        type=float,
        default=None,
        help="Optional override for the charge coefficient in hard acceptance.",
    )
    return parser.parse_args()


def valid_shard(path: Path, expected_n: int, min_length: int, max_length: int) -> bool:
    if not path.is_file():
        return False
    try:
        validate_records(
            load_records(path),
            expected_n=expected_n,
            min_length=min_length,
            max_length=max_length,
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return True


def main() -> None:
    args = parse_args()
    config_path = resolve_repo_path(args.config)
    config = load_target_config(config_path)
    target = config["target"]["name"]
    output_dir = Path(args.output_dir).expanduser().resolve()
    shard_dir = output_dir / "shards"
    log_dir = output_dir / "logs"
    shard_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    total = int(config["benchmark"]["n_sequences"]) if args.n_sequences is None else int(args.n_sequences)
    if total < 1:
        raise ValueError("--n-sequences must be positive")
    if args.shard_size < 1:
        raise ValueError("--shard-size must be positive")
    min_length = int(config["benchmark"]["min_length"])
    max_length = int(config["benchmark"]["max_length"])
    batch_size = (
        int(config["benchmark"]["generation_batch_size"])
        if args.batch_size is None else int(args.batch_size)
    )
    if batch_size < 1:
        raise ValueError("--batch-size must be positive")
    predictor = resolve_repo_path(config["paths"]["activity_predictor"])
    prior = resolve_repo_path(config["paths"]["prior"])
    tokenizer = resolve_repo_path(config["paths"]["tokenizer"])
    data_csv = resolve_repo_path(args.data_csv)
    configured_lora = config["paths"].get("lora_checkpoint")
    lora_value = None if args.without_sft else (args.lora_checkpoint or configured_lora)
    if not args.without_sft and not lora_value:
        raise ValueError(
            "Provide --lora-checkpoint, set paths.lora_checkpoint, or use --without-sft"
        )
    lora = resolve_repo_path(lora_value) if lora_value else None

    mic_config = config.get("mic", {})
    mic_enabled = bool(mic_config.get("enabled", False))
    mic_predictor = None
    if mic_enabled:
        mic_path_value = mic_config.get("model_path") or config["paths"].get("mic_predictor")
        if not mic_path_value:
            raise ValueError("MIC reward is enabled but no mic.model_path is configured")
        mic_predictor = resolve_repo_path(mic_path_value)

    required = [predictor, prior, tokenizer, data_csv]
    if lora:
        required.append(lora)
    if mic_predictor:
        required.append(mic_predictor)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required artifacts: {missing}")

    base_seed = int(config["benchmark"]["seed"]) + int(args.seed_offset)
    # Keep the paper-faithful default while allowing controlled reward
    # ablations to select a backend without changing the generation code.
    # The ablation configs use ``reward.source=esm_mlp_reg`` to remove the
    # task-prior term; all other variants retain the default prior+
    # regularized-ESM backend and set one component coefficient to zero.
    reward_source = str(
        config.get("reward", {}).get("source", "prior+esm_mlp_reg")
    )
    commands: list[list[str]] = []
    for shard in range((total + args.shard_size - 1) // args.shard_size):
        n = min(args.shard_size, total - shard * args.shard_size)
        shard_path = shard_dir / f"shard_{shard:04d}.json"
        if not args.force and valid_shard(shard_path, n, min_length, max_length):
            print(f"[skip] {shard_path}", flush=True)
            continue
        command = [
            args.python,
            "-m",
            "peptidedto.core",
            "--base-model",
            args.base_model,
            "--tokenizer-path",
            str(tokenizer),
            "--target-type",
            target,
            "--model-dtype",
            str(args.model_dtype or config["benchmark"]["model_dtype"]),
            "--n-sequences",
            str(n),
            "--n-ttgd-rounds",
            str(config["tgl"]["rounds"]),
            "--n-ttgd-iters",
            str(config["tgl"]["iterations"]),
            "--min-length",
            str(min_length),
            "--max-length",
            str(max_length),
            "--max-attempt-factor",
            str(config["benchmark"]["max_attempt_factor"]),
            "--gen-batch-size",
            str(batch_size),
            "--kl-coeff",
            str(config["tgl"]["kl_coeff"]),
            "--reward-coeff",
            str(config["tgl"]["reward_coeff"]),
            "--nll-coeff",
            str(config["tgl"]["nll_coeff"]),
            "--lr",
            str(config["tgl"]["learning_rate"]),
            "--rollout-temperature",
            str(config["sampling"]["temperature"]),
            "--rollout-top-p",
            str(config["sampling"]["top_p"]),
            "--rollout-top-k",
            str(config["sampling"]["top_k"]),
            "--resample-temperature",
            str(config["sampling"]["temperature"]),
            "--resample-top-p",
            str(config["sampling"]["top_p"]),
            "--resample-top-k",
            str(config["sampling"]["top_k"]),
            "--seed",
            str(base_seed + shard),
            "--device",
            args.device,
            "--data-csv",
            str(data_csv),
            "--prior-path",
            str(prior),
            "--reward-source",
            reward_source,
            "--esm-reward-model-path",
            str(predictor),
            "--paper-activity-coeff",
            "0.0",
            "--esm-activity-coeff",
            str(config["reward"]["activity_coeff"]),
            "--stability-coeff",
            str(config["reward"]["stability_coeff"]),
            "--similarity-coeff",
            str(config["reward"]["similarity_coeff"]),
            "--similarity-mode",
            "esm_similarity",
            "--length-coeff",
            str(config["reward"]["length_coeff"]),
            "--target-length",
            str(config["reward"]["target_length"]),
            "--stability-target",
            str(config["reward"]["stability_target"]),
            "--stability-scale",
            str(config["reward"]["stability_scale"]),
            "--length-scale",
            str(config["reward"]["length_scale"]),
            "--regularizer-reference-csv",
            str(data_csv),
            "--regularizer-max-references",
            str(config["reward"]["max_references"]),
            "--output",
            str(shard_path),
            "--verbose",
            "0",
        ]
        if lora:
            command.extend(["--lora-checkpoint", str(lora)])
        if args.hard_prior_length_weight is not None:
            command.extend(["--hard-prior-length-weight", str(args.hard_prior_length_weight)])
        if args.hard_charge_coeff is not None:
            command.extend(["--hard-charge-coeff", str(args.hard_charge_coeff)])
        if mic_predictor:
            command.extend(
                [
                    "--mic-regression-model-path",
                    str(mic_predictor),
                    "--mic-reward-coeff",
                    str(mic_config.get("reward_coeff", 0.25)),
                ]
            )
            if mic_config.get("reward_threshold") is not None:
                command.extend(
                    ["--mic-reward-threshold", str(mic_config["reward_threshold"])]
                )
            if mic_config.get("reward_scale") is not None:
                command.extend(["--mic-reward-scale", str(mic_config["reward_scale"])])
        log_path = log_dir / f"shard_{shard:04d}.log"
        print(f"[run] {target} shard={shard} n={n} seed={base_seed + shard}", flush=True)
        with log_path.open("w") as log:
            subprocess.run(command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        if not valid_shard(shard_path, n, min_length, max_length):
            raise RuntimeError(f"Generated shard failed validation: {shard_path}")
        commands.append(command)

    records = []
    shard_hashes = []
    for shard in range((total + args.shard_size - 1) // args.shard_size):
        path = shard_dir / f"shard_{shard:04d}.json"
        records.extend(load_records(path))
        shard_hashes.append({"path": str(path), "sha256": sha256_file(path)})
    records = records[:total]
    canonical = output_dir / f"peptidedto_{target}{total}.json"
    atomic_write_json(canonical, records)
    validation = validate_records(
        records,
        expected_n=total,
        min_length=min_length,
        max_length=max_length,
    )
    manifest = {
        "method": "PeptideDTO" if lora else "PeptideDTO (w/o SFT)",
        "target": target.upper(),
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "base_model": args.base_model,
        "lora_checkpoint": str(lora) if lora else None,
        "activity_predictor": str(predictor),
        "activity_predictor_sha256": sha256_file(predictor),
        "mic_predictor": str(mic_predictor) if mic_predictor else None,
        "mic_predictor_sha256": sha256_file(mic_predictor) if mic_predictor else None,
        "mic_reward": mic_config if mic_enabled else None,
        "prior_sha256": sha256_file(prior),
        "data_sha256": sha256_file(data_csv),
        "canonical_json": str(canonical),
        "canonical_json_sha256": sha256_file(canonical),
        "validation": validation,
        "shards": shard_hashes,
        "commands_executed_this_invocation": commands,
    }
    atomic_write_json(output_dir / "run_manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
