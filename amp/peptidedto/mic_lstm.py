"""Independent LSTM MIC evaluator adapted from the Huang et al. baseline.

The evaluator is deliberately separate from :mod:`peptidedto.core`: it is
used to audit generated peptides with an architecture that is independent of
the ESM+MLP surrogate used by PeptideDTO.  The default architecture follows
the public Huang implementation (50-dimensional residue embeddings, a
two-layer unidirectional LSTM with hidden size 128, mean pooling, and a
128 -> 64 -> 1 regression head).

Targets are predicted in log10(MIC in micromolar).  This module does not
provide a relaxed-space reward and is therefore not part of TTGD's gradient
path.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_ID = {aa: index + 1 for index, aa in enumerate(AA_ALPHABET)}
PAD_ID = 0
CHECKPOINT_KIND = "peptidedto_huang_lstm_mic_v1"
REFERENCE_DOI = "10.1038/s41551-022-00991-2"
DATA_REFERENCE_DOI = "10.1126/sciadv.ads8932"


def clean_lstm_sequence(sequence: str | None) -> str:
    """Keep canonical amino-acid symbols in a sequence.

    Missing JSON/CSV values are treated as invalid rather than converted to a
    string (``str(None)`` contains canonical ``N`` residues and could
    otherwise be scored as a real peptide).
    """

    if sequence is None:
        return ""
    return "".join(char for char in str(sequence).upper() if char in AA_TO_ID)


def encode_sequences(
    sequences: Sequence[str],
    *,
    device: str | torch.device = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """Encode variable-length peptide strings as padded ids and lengths.

    Returns ``(ids, lengths, cleaned_sequences)``.  Empty inputs and sequences
    that contain no canonical residues are rejected rather than silently
    turning into an all-padding example.
    """

    cleaned = [clean_lstm_sequence(sequence) for sequence in sequences]
    if not cleaned:
        raise ValueError("At least one peptide sequence is required")
    if any(not sequence for sequence in cleaned):
        raise ValueError("Sequences must contain at least one canonical amino acid")
    lengths = torch.tensor([len(sequence) for sequence in cleaned], dtype=torch.long)
    width = int(lengths.max().item())
    ids = torch.full(
        (len(cleaned), width),
        PAD_ID,
        dtype=torch.long,
        device=device,
    )
    for row, sequence in enumerate(cleaned):
        ids[row, : len(sequence)] = torch.tensor(
            [AA_TO_ID[char] for char in sequence],
            dtype=torch.long,
            device=device,
        )
    return ids, lengths.to(device), cleaned


class HuangLSTMRegressor(nn.Module):
    """One organism-specific MIC regressor.

    The mean is masked by the true sequence length.  The original reference
    code uses packed sequences and then averages the unpacked tensor; masking
    makes the intended mean pooling explicit and avoids batch-padding bias.
    """

    def __init__(
        self,
        *,
        embedding_dim: int = 50,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.7,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        if bidirectional:
            raise ValueError("The Huang baseline is unidirectional")
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.bidirectional = False
        self.embedding = nn.Embedding(
            len(AA_ALPHABET) + 1,
            self.embedding_dim,
            padding_idx=PAD_ID,
        )
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
        )
        self.regressor = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, token_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, length]")
        if lengths.ndim != 1 or lengths.shape[0] != token_ids.shape[0]:
            raise ValueError("lengths must have shape [batch]")
        lengths = lengths.to(token_ids.device)
        if bool((lengths <= 0).any()):
            raise ValueError("all sequence lengths must be positive")
        if bool((lengths > token_ids.shape[1]).any()):
            raise ValueError("a sequence length exceeds the padded width")

        embedded = self.embedding(token_ids.long())
        packed = pack_padded_sequence(
            embedded,
            lengths.detach().to("cpu"),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.lstm(packed)
        output, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=token_ids.shape[1],
        )
        mask = (
            torch.arange(token_ids.shape[1], device=token_ids.device)
            .view(1, -1)
            .lt(lengths.view(-1, 1))
            .to(output.dtype)
        )
        pooled = (output * mask.unsqueeze(-1)).sum(dim=1) / lengths.to(
            output.dtype
        ).clamp_min(1.0).unsqueeze(-1)
        return self.regressor(pooled).squeeze(-1)


class HuangLSTMMICModel(nn.Module):
    """Three independent Huang-style regressors in one checkpoint."""

    CHECKPOINT_KIND = CHECKPOINT_KIND

    def __init__(
        self,
        *,
        species_names: Sequence[str] = ("s_aureus", "e_coli", "p_aeruginosa"),
        embedding_dim: int = 50,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.7,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        names = tuple(str(name) for name in species_names)
        if not names:
            raise ValueError("At least one species is required")
        self.species_names = names
        self.models = nn.ModuleDict(
            {
                name: HuangLSTMRegressor(
                    embedding_dim=embedding_dim,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    dropout=dropout,
                )
                for name in names
            }
        )
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.to(device)
        self.eval()

    def forward(
        self,
        token_ids: torch.Tensor,
        lengths: torch.Tensor,
        species: str,
    ) -> torch.Tensor:
        if species not in self.models:
            raise KeyError(f"Unknown species {species!r}; choose from {self.species_names}")
        return self.models[species](token_ids, lengths)

    @torch.no_grad()
    def score_log_mic(
        self,
        sequences: Sequence[str],
        *,
        batch_size: int = 128,
    ) -> torch.Tensor:
        """Return ``[n_sequences, n_species]`` log10-MIC predictions."""

        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        values: list[torch.Tensor] = []
        device = next(self.parameters()).device
        for start in range(0, len(sequences), batch_size):
            batch = sequences[start : start + batch_size]
            ids, lengths, _cleaned = encode_sequences(batch, device=device)
            values.append(
                torch.stack(
                    [self.models[name](ids, lengths) for name in self.species_names],
                    dim=-1,
                )
            )
        if not values:
            return torch.empty((0, len(self.species_names)), device=device)
        return torch.cat(values, dim=0)

    @torch.no_grad()
    def score_mic_uM(
        self,
        sequences: Sequence[str],
        *,
        batch_size: int = 128,
    ) -> torch.Tensor:
        """Return MIC predictions in linear micromolar units."""

        return torch.pow(10.0, self.score_log_mic(sequences, batch_size=batch_size))


def load_huang_lstm_mic_model(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> HuangLSTMMICModel:
    """Load and validate an independent LSTM MIC checkpoint."""

    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"LSTM MIC checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if checkpoint.get("checkpoint_kind") != CHECKPOINT_KIND:
        raise ValueError(
            f"Unsupported LSTM MIC checkpoint kind {checkpoint.get('checkpoint_kind')!r}; "
            f"expected {CHECKPOINT_KIND!r}"
        )
    checkpoint_alphabet = checkpoint.get("aa_alphabet", AA_ALPHABET)
    if checkpoint_alphabet != AA_ALPHABET:
        raise ValueError(
            "Unsupported LSTM MIC alphabet; checkpoint was trained with "
            f"{checkpoint_alphabet!r}, expected {AA_ALPHABET!r}"
        )
    architecture = checkpoint.get("architecture", {})
    model = HuangLSTMMICModel(
        species_names=checkpoint["species_names"],
        embedding_dim=int(architecture.get("embedding_dim", 50)),
        hidden_dim=int(architecture.get("hidden_dim", 128)),
        num_layers=int(architecture.get("num_layers", 2)),
        dropout=float(architecture.get("dropout", 0.7)),
        device=device,
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    return model
