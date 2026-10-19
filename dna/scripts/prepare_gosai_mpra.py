#!/usr/bin/env python3
"""Audit and split the Gosai MPRA table for a cross-paper benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


REQUIRED_COLUMNS = (
    "IDs",
    "chr",
    "data_project",
    "OL",
    "class",
    "K562_log2FC",
    "HepG2_log2FC",
    "SKNSH_log2FC",
    "K562_lfcSE",
    "HepG2_lfcSE",
    "SKNSH_lfcSE",
    "sequence",
)
ACTIVITY_COLUMNS = ("K562_log2FC", "HepG2_log2FC", "SKNSH_log2FC")
SE_COLUMNS = ("K562_lfcSE", "HepG2_lfcSE", "SKNSH_lfcSE")


def split_key(seed: int, row_index: int, row: dict[str, str]) -> bytes:
    """Return an implementation-stable key for one row."""
    payload = f"{seed}\0{row_index}\0{row['IDs']}\0{row['sequence']}".encode("ascii")
    return hashlib.sha256(payload).digest()


def validate_header(fieldnames: list[str] | None) -> list[str]:
    if fieldnames is None:
        raise ValueError("input table has no header")
    missing = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
    if missing:
        raise ValueError(f"input table is missing required columns: {missing}")
    return fieldnames


def audit_and_rank(
    path: Path, seed: int, sequence_length: int
) -> tuple[list[tuple[bytes, int]], dict[str, object]]:
    ranked_rows: list[tuple[bytes, int]] = []
    sequence_lengths: Counter[int] = Counter()
    projects: Counter[str] = Counter()
    invalid_sequence_rows = 0
    non_finite_rows = 0

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        validate_header(reader.fieldnames)
        for row_index, row in enumerate(reader):
            sequence = row["sequence"].strip().upper()
            sequence_lengths[len(sequence)] += 1
            projects[row["data_project"]] += 1
            if not sequence or set(sequence) - set("ACGT"):
                invalid_sequence_rows += 1

            numeric_values = [row[column] for column in ACTIVITY_COLUMNS + SE_COLUMNS]
            try:
                finite = all(math.isfinite(float(value)) for value in numeric_values)
            except ValueError:
                finite = False
            non_finite_rows += int(not finite)
            if len(sequence) == sequence_length:
                ranked_rows.append((split_key(seed, row_index, row), row_index))

    if not ranked_rows:
        raise ValueError("input table contains no data rows")
    if invalid_sequence_rows:
        raise ValueError(f"found {invalid_sequence_rows} rows with invalid DNA")
    if non_finite_rows:
        raise ValueError(f"found {non_finite_rows} rows with non-finite targets")

    ranked_rows.sort()
    metadata: dict[str, object] = {
        "source": str(path.resolve()),
        "source_size_bytes": path.stat().st_size,
        "seed": seed,
        "split_method": "exact_half_by_sha256(seed,row_index,IDs,sequence)",
        "protocol": "fixed cross-paper Gosai MPRA benchmark split",
        "official_author_split": False,
        "official_split_caveat": (
            "This benchmark intentionally defines a new fixed split; it does not "
            "attempt to reconstruct the DNA-CRAFT authors' private split."
        ),
        "selected_sequence_length": sequence_length,
        "row_count": len(ranked_rows),
        "source_row_count": sum(sequence_lengths.values()),
        "excluded_row_count": sum(
            count for length, count in sequence_lengths.items() if length != sequence_length
        ),
        "sequence_length_counts": {
            str(length): count for length, count in sorted(sequence_lengths.items())
        },
        "data_project_counts": dict(projects.most_common()),
        "activity_columns": list(ACTIVITY_COLUMNS),
        "activity_order": ["K562", "HepG2", "SK-N-SH"],
    }
    return ranked_rows, metadata


def write_splits(
    source: Path, output_dir: Path, design_indices: set[int], sequence_length: int
) -> tuple[int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    design_path = output_dir / "gosai_design_model.tsv"
    evaluation_path = output_dir / "gosai_evaluation_model.tsv"
    manifest_path = output_dir / "gosai_split_manifest.tsv"
    counts = {"design": 0, "evaluation": 0}

    with (
        source.open("r", encoding="utf-8", newline="") as source_handle,
        design_path.open("w", encoding="utf-8", newline="") as design_handle,
        evaluation_path.open("w", encoding="utf-8", newline="") as evaluation_handle,
        manifest_path.open("w", encoding="utf-8", newline="") as manifest_handle,
    ):
        reader = csv.DictReader(source_handle, delimiter="\t")
        fieldnames = validate_header(reader.fieldnames)
        writers = {
            "design": csv.DictWriter(design_handle, fieldnames=fieldnames, delimiter="\t"),
            "evaluation": csv.DictWriter(evaluation_handle, fieldnames=fieldnames, delimiter="\t"),
        }
        for writer in writers.values():
            writer.writeheader()
        manifest = csv.writer(manifest_handle, delimiter="\t", lineterminator="\n")
        manifest.writerow(("row_index", "IDs", "split"))

        for row_index, row in enumerate(reader):
            if len(row["sequence"].strip()) != sequence_length:
                continue
            split = "design" if row_index in design_indices else "evaluation"
            writers[split].writerow(row)
            manifest.writerow((row_index, row["IDs"], split))
            counts[split] += 1

    return counts["design"], counts["evaluation"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="Table_S2__MPRA_dataset.txt")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20240827)
    parser.add_argument("--sequence-length", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ranked_rows, metadata = audit_and_rank(args.input, args.seed, args.sequence_length)
    design_size = len(ranked_rows) // 2
    design_indices = {row_index for _, row_index in ranked_rows[:design_size]}
    design_count, evaluation_count = write_splits(
        args.input, args.output_dir, design_indices, args.sequence_length
    )
    metadata["design_row_count"] = design_count
    metadata["evaluation_row_count"] = evaluation_count
    metadata_path = args.output_dir / "gosai_split_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
