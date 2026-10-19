#!/usr/bin/env python3
"""Generate a fixed Evo 2 sequence pool without reloading the model."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch

from src import Evo2TestTimeDesigner, SearchConfig


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--length", type=int, default=200)
    parser.add_argument("--num-sequences", type=int, default=32)
    parser.add_argument("--generation-batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20240827)
    parser.add_argument("--prompt", default="ACGT")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument(
        "--skip-invalid",
        action="store_true",
        help="Discard generated sequences containing unsupported bases and continue.",
    )
    args = parser.parse_args()
    if args.num_sequences < 1:
        raise ValueError("--num-sequences must be positive")
    if args.generation_batch_size < 1:
        raise ValueError("--generation-batch-size must be positive")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    designer = Evo2TestTimeDesigner(
        model_name="evo2_7b", lora_adapter=args.adapter
    )
    designer.dto.disable_flash_attention()

    sequences: list[str] = []
    seen: set[str] = set()
    batches = 0
    invalid_batches = 0
    max_batches = max(16, args.num_sequences * 4)
    while len(sequences) < args.num_sequences and batches < max_batches:
        batch_size = min(
            args.generation_batch_size, args.num_sequences - len(sequences)
        )
        config = SearchConfig(
            num_initial=batch_size,
            initial_temperature=args.temperature,
            initial_top_k=args.top_k,
            initial_top_p=args.top_p,
        )
        try:
            generated = designer.sample_initial(args.prompt, args.length, config)
        except ValueError as exc:
            if not args.skip_invalid or "unsupported bases" not in str(exc):
                raise
            invalid_batches += 1
            batches += 1
            print(
                f"discarded_invalid_batch={invalid_batches} reason={exc}",
                flush=True,
            )
            continue
        for sequence in generated:
            if sequence not in seen:
                seen.add(sequence)
                sequences.append(sequence)
        batches += 1
        print(f"generated_unique={len(sequences)}/{args.num_sequences}", flush=True)

    if len(sequences) != args.num_sequences:
        raise RuntimeError(
            f"generated only {len(sequences)} unique sequences after {batches} batches"
        )

    payload = {
        "model": "evo2_7b",
        "adapter": str(args.adapter) if args.adapter else None,
        "adapter_sha256": sha256(args.adapter) if args.adapter else None,
        "seed": args.seed,
        "prompt": args.prompt,
        "length": args.length,
        "num_sequences": len(sequences),
        "generation_batch_size": args.generation_batch_size,
        "generation_batches": batches,
        "invalid_batches_discarded": invalid_batches,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "sequences": sequences,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    summary = {key: value for key, value in payload.items() if key != "sequences"}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
