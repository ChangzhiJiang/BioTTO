from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable


AA_ALPHABET = set("ACDEFGHIKLMNPQRSTVWY")


def clean_sequence(value: object) -> str:
    return "".join(char for char in str(value).upper() if char in AA_ALPHABET)


def sequence_of(record: Any) -> str:
    if isinstance(record, str):
        return clean_sequence(record)
    if isinstance(record, dict):
        for key in ("sequence", "seq", "peptide", "peptide_sequence"):
            if record.get(key):
                return clean_sequence(record[key])
    return ""


def load_records(path: str | Path) -> list[Any]:
    with Path(path).open() as handle:
        data = json.load(handle)
    if isinstance(data, dict) and "results" in data:
        data = data["results"]
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list or a dict with results")
    return data


def validate_records(
    records: Iterable[Any],
    *,
    expected_n: int,
    min_length: int = 5,
    max_length: int = 50,
) -> dict[str, Any]:
    rows = list(records)
    if len(rows) != expected_n:
        raise ValueError(f"Expected {expected_n} records, found {len(rows)}")
    sequences = [sequence_of(row) for row in rows]
    invalid = [
        {"index": index, "sequence": sequence, "length": len(sequence)}
        for index, sequence in enumerate(sequences)
        if not min_length <= len(sequence) <= max_length
    ]
    if invalid:
        preview = invalid[:10]
        raise ValueError(
            f"Found {len(invalid)} sequences outside {min_length}-{max_length}: {preview}"
        )
    return {
        "n": len(sequences),
        "unique_n": len(set(sequences)),
        "unique_pct": 100.0 * len(set(sequences)) / max(len(sequences), 1),
        "min_length": min(map(len, sequences)),
        "max_length": max(map(len, sequences)),
        "mean_length": sum(map(len, sequences)) / len(sequences),
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output.name}.",
        suffix=".tmp",
        dir=output.parent,
    )
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        os.replace(temporary, output)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as handle:
        config = json.load(handle)
    if config["length"] != {"min": 5, "max": 50}:
        raise ValueError("This runner is restricted to the paper-faithful 5-50 benchmark")
    return config
