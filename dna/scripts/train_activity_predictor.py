#!/usr/bin/env python3
"""Train one independent short-input Enformer activity predictor."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from src.predictor import ShortEnformerActivity


BASE_TO_INDEX = {"A": 0, "C": 1, "G": 2, "T": 3}
TARGET_COLUMNS = ("K562_log2FC", "HepG2_log2FC", "SKNSH_log2FC")


class GosaiDataset(Dataset[tuple[str, torch.Tensor]]):
    def __init__(
        self, table: Path, sequence_length: int = 200, max_rows: int | None = None
    ) -> None:
        self.sequences: list[str] = []
        labels: list[list[float]] = []
        with table.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                sequence = row["sequence"].strip().upper()
                if len(sequence) != sequence_length or set(sequence) - set(BASE_TO_INDEX):
                    continue
                self.sequences.append(sequence)
                labels.append([float(row[column]) for column in TARGET_COLUMNS])
                if max_rows is not None and len(self.sequences) >= max_rows:
                    break
        if not self.sequences:
            raise ValueError(f"no valid {sequence_length} bp rows found in {table}")
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> tuple[str, torch.Tensor]:
        return self.sequences[index], self.labels[index]


def collate(batch: list[tuple[str, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    sequences, labels = zip(*batch)
    indices = torch.tensor(
        [[BASE_TO_INDEX[base] for base in sequence] for sequence in sequences],
        dtype=torch.long,
    )
    return F.one_hot(indices, num_classes=4).float(), torch.stack(labels)


def run_epoch(
    model: ShortEnformerActivity,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_items = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        prediction = model(x)
        loss = F.mse_loss(prediction, y)
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        count = y.shape[0]
        total_loss += float(loss.detach()) * count
        total_items += count
    return total_loss / max(total_items, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--valid-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20240827)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-rows", type=int)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = GosaiDataset(args.table, max_rows=args.max_rows)
    indices = list(range(len(dataset)))
    random.Random(args.seed).shuffle(indices)
    valid_size = max(1, int(len(indices) * args.valid_fraction))
    valid_indices = indices[:valid_size]
    train_indices = indices[valid_size:]
    common = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": collate,
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(Subset(dataset, train_indices), shuffle=True, **common)
    valid_loader = DataLoader(Subset(dataset, valid_indices), shuffle=False, **common)

    model = ShortEnformerActivity().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_valid = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []
    for epoch in range(args.epochs):
        train_loss = run_epoch(model, train_loader, optimizer, device)
        with torch.no_grad():
            valid_loss = run_epoch(model, valid_loader, None, device)
        history.append({"epoch": float(epoch), "train_mse": train_loss, "valid_mse": valid_loss})
        print(json.dumps(history[-1]), flush=True)
        if valid_loss < best_valid:
            best_valid = valid_loss
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("no checkpoint was selected")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "model_config": model.config,
            "target_columns": TARGET_COLUMNS,
            "table": str(args.table.resolve()),
            "seed": args.seed,
            "train_rows": len(train_indices),
            "valid_rows": len(valid_indices),
            "best_valid_mse": best_valid,
            "history": history,
        },
        args.output,
    )
    print(json.dumps({"checkpoint": str(args.output), "best_valid_mse": best_valid}))


if __name__ == "__main__":
    main()
