#!/usr/bin/env python3
"""Target-specific LoRA SFT for the ProGen2 prior used by PeptideDTO.

This script fine-tunes only the PEFT LoRA weights and saves a plain
``model.state_dict()`` checkpoint compatible with PeptideDTO generation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from peptidedto.modeling import load_pretrained_progen_model


AA_ALPHABET = set("ACDEFGHIKLMNPQRSTVWY")


def clean_sequence(value: object) -> str:
    return "".join(ch for ch in str(value).upper() if ch in AA_ALPHABET)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_target_sequences(
    csv_path: Path,
    *,
    label: int,
    max_len: int,
    min_len: int,
    limit: int | None,
    seed: int,
) -> list[str]:
    df = pd.read_csv(csv_path)
    if "label" in df.columns:
        df = df[df["label"].astype(int) == int(label)]
    if "sequence" not in df.columns:
        raise ValueError(f"{csv_path} must contain a sequence column")
    seqs = [clean_sequence(seq) for seq in df["sequence"].tolist()]
    seqs = [seq for seq in seqs if min_len <= len(seq) <= max_len]
    seen = set()
    deduped: list[str] = []
    for seq in seqs:
        if seq not in seen:
            seen.add(seq)
            deduped.append(seq)
    if limit is not None and limit > 0 and len(deduped) > limit:
        rng = random.Random(seed)
        rng.shuffle(deduped)
        deduped = deduped[:limit]
    if not deduped:
        raise RuntimeError(f"No label={label} sequences left after filtering")
    return deduped


class PeptideSFTDataset(Dataset):
    def __init__(
        self,
        sequences: Iterable[str],
        tokenizer,
        *,
        prompt: str,
        max_length: int,
    ) -> None:
        self.examples = []
        prompt_tokens = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        # BertTokenizer encodes prompt "1" as [CLS] [UNK] [SEP].  During
        # training the final SEP should be replaced by residues, so only the
        # [CLS] [UNK] prefix is masked.
        self.prompt_len = max(len(prompt_tokens) - 1, 0)
        for seq in sequences:
            text = prompt + " " + " ".join(seq.lower())
            enc = tokenizer(
                text,
                add_special_tokens=True,
                truncation=True,
                max_length=max_length,
            )
            input_ids = enc["input_ids"]
            attention_mask = enc["attention_mask"]
            labels = list(input_ids)
            for idx in range(min(self.prompt_len, len(labels))):
                labels[idx] = -100
            self.examples.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        return self.examples[idx]


def collate_batch(batch: list[dict[str, list[int]]], pad_token_id: int) -> dict[str, torch.Tensor]:
    max_len = max(len(row["input_ids"]) for row in batch)
    input_ids = []
    attention_mask = []
    labels = []
    for row in batch:
        pad = max_len - len(row["input_ids"])
        input_ids.append(row["input_ids"] + [pad_token_id] * pad)
        attention_mask.append(row["attention_mask"] + [0] * pad)
        labels.append(row["labels"] + [-100] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


@dataclass
class TrainConfig:
    data_csv: str
    output_dir: str
    base_model: str
    tokenizer_path: str
    label: int
    prompt: str
    min_len: int
    max_seq_len: int
    limit: int | None
    epochs: int
    max_steps: int
    batch_size: int
    grad_accum: int
    lr: float
    weight_decay: float
    warmup_ratio: float
    val_fraction: float
    seed: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    device: str
    fp32: bool


def evaluate(model, loader: DataLoader, device: str) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model(**batch).loss
            losses.append(float(loss.detach().cpu()))
    model.train()
    return float(sum(losses) / max(len(losses), 1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-csv", default=str(ROOT / "data/processed/mulit_peptide_train.csv"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--label", type=int, default=0)
    parser.add_argument("--prompt", default="1")
    parser.add_argument("--min-len", type=int, default=5)
    parser.add_argument("--max-seq-len", type=int, default=50)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260617)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--fp32", action="store_true")
    args = parser.parse_args()

    cfg = TrainConfig(**vars(args))
    set_seed(cfg.seed)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seqs = read_target_sequences(
        Path(cfg.data_csv),
        label=cfg.label,
        max_len=cfg.max_seq_len,
        min_len=cfg.min_len,
        limit=cfg.limit,
        seed=cfg.seed,
    )
    dtype = torch.float32 if cfg.fp32 else torch.float16
    tokenizer, model = load_pretrained_progen_model(
        cfg.base_model,
        cfg.tokenizer_path,
        device_map=cfg.device,
        torch_dtype=dtype,
        lora_r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        inference_mode=False,
    )
    model.train()
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    max_token_len = cfg.max_seq_len + 3
    dataset = PeptideSFTDataset(
        seqs,
        tokenizer,
        prompt=cfg.prompt,
        max_length=max_token_len,
    )
    val_size = max(1, int(len(dataset) * cfg.val_fraction)) if cfg.val_fraction > 0 else 0
    train_size = len(dataset) - val_size
    if val_size > 0:
        train_ds, val_ds = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(cfg.seed),
        )
    else:
        train_ds, val_ds = dataset, None

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_batch(batch, pad_token_id),
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            collate_fn=lambda batch: collate_batch(batch, pad_token_id),
        )
        if val_ds is not None
        else None
    )

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
    updates_per_epoch = math.ceil(len(train_loader) / max(cfg.grad_accum, 1))
    total_steps = cfg.max_steps if cfg.max_steps > 0 else max(1, cfg.epochs * updates_per_epoch)
    warmup_steps = int(total_steps * cfg.warmup_ratio)

    def lr_scale(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        remain = max(total_steps - step, 0)
        decay_span = max(total_steps - warmup_steps, 1)
        return max(0.1, remain / decay_span)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_scale)

    history = []
    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(total=total_steps, desc=f"label={cfg.label} SFT updates")
    for epoch in range(cfg.epochs):
        running = 0.0
        micro = 0
        for batch_idx, batch in enumerate(train_loader):
            batch = {key: value.to(cfg.device) for key, value in batch.items()}
            loss = model(**batch).loss / max(cfg.grad_accum, 1)
            loss.backward()
            running += float(loss.detach().cpu()) * max(cfg.grad_accum, 1)
            micro += 1
            should_step = micro % cfg.grad_accum == 0 or batch_idx + 1 == len(train_loader)
            if not should_step:
                continue
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            pbar.update(1)
            avg_loss = running / max(micro, 1)
            pbar.set_postfix(epoch=epoch + 1, loss=f"{avg_loss:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
            if global_step >= total_steps:
                break
        val_loss = evaluate(model, val_loader, cfg.device) if val_loader is not None else float("nan")
        history.append(
            {
                "epoch": epoch + 1,
                "global_step": global_step,
                "train_loss": running / max(micro, 1),
                "val_loss": val_loss,
            }
        )
        if global_step >= total_steps:
            break
    pbar.close()

    checkpoint_path = output_dir / "progen2_sft_lora_state.pt"
    lora_state = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if "lora_" in key
    }
    if not lora_state:
        raise RuntimeError("No LoRA weights found in model.state_dict(); refusing to save an empty checkpoint")
    torch.save(lora_state, checkpoint_path)
    try:
        model.save_pretrained(output_dir / "peft_adapter")
    except Exception as exc:
        print(f"Warning: could not save PEFT adapter directory: {exc}", flush=True)

    metadata = {
        "config": asdict(cfg),
        "n_sequences": len(seqs),
        "n_train": train_size,
        "n_val": val_size,
        "max_token_len": max_token_len,
        "checkpoint": str(checkpoint_path),
        "history": history,
    }
    with (output_dir / "sft_metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
