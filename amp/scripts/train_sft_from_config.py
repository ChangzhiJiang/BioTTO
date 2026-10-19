#!/usr/bin/env python3
"""Launch target-specific LoRA SFT from a PeptideDTO YAML config."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from peptidedto.config import load_target_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--data-csv", default="data/processed/mulit_peptide_train.csv")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fp32", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = load_target_config(config_path)
    target = config["target"]
    sft = config["sft"]
    output_dir = args.output_dir or f"checkpoints/lora/{target['name']}"
    command = [
        sys.executable,
        str(ROOT / "scripts/train_sft.py"),
        "--data-csv",
        args.data_csv,
        "--output-dir",
        output_dir,
        "--base-model",
        args.base_model,
        "--tokenizer-path",
        "assets/vocab.txt",
        "--label",
        str(target["label"]),
        "--prompt",
        "1",
        "--min-len",
        str(config["benchmark"]["min_length"]),
        "--max-seq-len",
        str(config["benchmark"]["max_length"]),
        "--epochs",
        str(sft["epochs"]),
        "--max-steps",
        str(sft["max_steps"]),
        "--batch-size",
        str(sft["batch_size"]),
        "--grad-accum",
        str(sft["gradient_accumulation"]),
        "--lr",
        str(sft["learning_rate"]),
        "--weight-decay",
        str(sft["weight_decay"]),
        "--warmup-ratio",
        str(sft["warmup_ratio"]),
        "--seed",
        str(sft["seed"]),
        "--lora-r",
        str(sft["lora_rank"]),
        "--lora-alpha",
        str(sft["lora_alpha"]),
        "--lora-dropout",
        str(sft["lora_dropout"]),
        "--device",
        args.device,
    ]
    if args.fp32:
        command.append("--fp32")
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
