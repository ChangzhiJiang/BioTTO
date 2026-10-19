"""Numerically strict PeptideDTO for AMP/AFP/AVP peptide generation.

Algorithm (per sequence):
  1. Sample initial rollout from ProGen2 (PLM)
  2. Store per-token logits as a trainable "latent" representation
  3. For n_ttgd_rounds:
       a. Optimise logits via gradient descent to maximise the configured
          composite reward.  For AMP MIC runs this includes the differentiable
          auxiliary term ``mic_reward_coeff * R_MIC(y_soft)`` in addition to
          the class/activity, stability, similarity, and length terms.
          NLL and KL regularizers constrain the update relative to ProGen2.
       b. Resample from the optimised logits
       c. Accept the new sequence if its reward improves (optional rejection sampling)

Unlike the legacy implementation, v2 initializes the latent optimization from raw
autoregressive model logits, never from top-p-warped sampling scores. Every tensor
on the optimization path is checked for finite values before it can reach Adam.

This module is the corrected implementation used for the 5--50 residue benchmark.
"""
from __future__ import annotations
import argparse
import csv
import gzip
import json
import math
import os
import pickle
import random
import sys
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

_EXTERNAL_TOOL_PATHS = [
    p for p in os.environ.get("PEPTIDEDTO_EXTERNAL_PATHS", "").split(os.pathsep) if p
]
for _tool_path in _EXTERNAL_TOOL_PATHS:
    if os.path.isdir(_tool_path) and _tool_path not in sys.path:
        sys.path.append(_tool_path)

try:
    import numpy as np
except Exception:  # pragma: no cover - optional for --train-prior
    np = None

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - lets --train-prior work without torch
    TORCH_AVAILABLE = False

    class _TorchStub:
        def no_grad(self):
            def decorator(fn):
                return fn
            return decorator

        def __getattr__(self, name):
            raise RuntimeError("PyTorch is required for TTGD generation.")

    class _NNStub:
        class Module:
            pass

    class _FStub:
        pass

    torch = _TorchStub()
    nn = _NNStub()
    F = _FStub()

try:
    from Bio.SeqUtils import ProtParam, ProtParamData
except Exception:  # pragma: no cover - optional; fallback charge is below
    ProtParam = None
    ProtParamData = None

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **_kwargs):
        return x

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "MIC_predictor"))
try:
    from MIC_predictor import ApexMIC
except Exception:  # pragma: no cover - external AMP-only reward is optional
    ApexMIC = None

try:
    from .modeling import (
        get_progen_input_embeddings,
        load_pretrained_progen_model,
    )
except ImportError:  # pragma: no cover - supports direct script execution
    from peptidedto.modeling import (
        get_progen_input_embeddings,
        load_pretrained_progen_model,
    )

load_esm = None

try:
    from macrel_predictor.predictor import macrel_predictor
except Exception:  # pragma: no cover
    macrel_predictor = None


AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INDEX = {aa: i for i, aa in enumerate(AA_ALPHABET)}
LABEL_TO_TYPE = {0: "AMP", 1: "AFP", 2: "AVP"}
TYPE_TO_LABEL = {
    "amp": 0,
    "antimicrobial": 0,
    "antibacterial": 0,
    "afp": 1,
    "antifungal": 1,
    "avp": 2,
    "antiviral": 2,
}
REWARD_SOURCE_ALIASES = {
    "esm_mlp+regularized": "esm_mlp_reg",
    "regularized_esm_mlp": "esm_mlp_reg",
    "prior+esm_mlp+regularized": "prior+esm_mlp_reg",
    "prior+regularized_esm_mlp": "prior+esm_mlp_reg",
    "activity+esm_mlp+regularized": "activity+esm_mlp_reg",
    "activity+regularized_esm_mlp": "activity+esm_mlp_reg",
    "prior+activity+esm_mlp+regularized": "prior+activity+esm_mlp_reg",
    "prior+activity+regularized_esm_mlp": "prior+activity+esm_mlp_reg",
}
REWARD_SOURCES = {
    "prior",
    "activity",
    "prior+activity",
    "esm_mlp",
    "prior+esm_mlp",
    "activity+esm_mlp",
    "prior+activity+esm_mlp",
    "esm_mlp_reg",
    "prior+esm_mlp_reg",
    "activity+esm_mlp_reg",
    "prior+activity+esm_mlp_reg",
}
ACTIVITY_REWARD_SOURCES = {
    "activity",
    "prior+activity",
    "activity+esm_mlp",
    "prior+activity+esm_mlp",
    "activity+esm_mlp_reg",
    "prior+activity+esm_mlp_reg",
}
ESM_REWARD_SOURCES = {
    "esm_mlp",
    "prior+esm_mlp",
    "activity+esm_mlp",
    "prior+activity+esm_mlp",
    "esm_mlp_reg",
    "prior+esm_mlp_reg",
    "activity+esm_mlp_reg",
    "prior+activity+esm_mlp_reg",
}
REGULARIZED_REWARD_SOURCES = {
    "esm_mlp_reg",
    "prior+esm_mlp_reg",
    "activity+esm_mlp_reg",
    "prior+activity+esm_mlp_reg",
}
SIMILARITY_MODES = {"novelty", "similarity", "esm_similarity"}


def resolve_esm2_model_path(value: str) -> str:
    """Resolve the portable ESM2-8M alias used in released checkpoints."""
    if value != "ESM2-8M":
        return os.path.expanduser(value)
    override = os.environ.get("PEPTIDEDTO_ESM2_MODEL")
    if override:
        return os.path.expanduser(override)
    repository_model = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "models", "esm2_t6_8M_UR50D")
    )
    if os.path.isdir(repository_model):
        return repository_model
    return "facebook/esm2_t6_8M_UR50D"


def require_finite(tensor: torch.Tensor, name: str) -> torch.Tensor:
    """Raise with useful diagnostics when a TGLP optimization tensor is non-finite."""
    finite = torch.isfinite(tensor)
    if bool(finite.all()):
        return tensor
    bad = int((~finite).sum().detach().cpu())
    total = tensor.numel()
    finite_values = tensor.detach()[finite]
    if finite_values.numel():
        minimum = float(finite_values.min().cpu())
        maximum = float(finite_values.max().cpu())
        value_range = f"finite_range=[{minimum:.6g}, {maximum:.6g}]"
    else:
        value_range = "finite_range=empty"
    raise FloatingPointError(
        f"{name} contains {bad}/{total} non-finite values; {value_range}"
    )


def set_generation_seed(seed: int) -> None:
    """Seed every RNG used by rollout and latent-space resampling."""
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def categorical_reverse_kl(
    current_logits: torch.Tensor,
    reference_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Return mean KL(q_current || p_reference) for finite categorical logits."""
    require_finite(current_logits, "current_logits_for_kl")
    require_finite(reference_log_probs, "reference_log_probs_for_kl")
    current_log_probs = F.log_softmax(current_logits, dim=-1)
    current_probs = current_log_probs.exp()
    kl = (current_probs * (current_log_probs - reference_log_probs)).sum(dim=-1).mean()
    return require_finite(kl, "categorical_reverse_kl")


def canonical_reward_source(value: str) -> str:
    key = str(value).strip().lower()
    return REWARD_SOURCE_ALIASES.get(key, key)


def canonical_target(value: str | int) -> int:
    if isinstance(value, int):
        if value in LABEL_TO_TYPE:
            return value
        raise ValueError(f"Unknown peptide label: {value}")
    key = str(value).strip().lower()
    if key.isdigit():
        return canonical_target(int(key))
    if key not in TYPE_TO_LABEL:
        choices = ", ".join(sorted(TYPE_TO_LABEL))
        raise ValueError(f"Unknown peptide type {value!r}; choose one of: {choices}")
    return TYPE_TO_LABEL[key]


def clean_sequence(seq: str, *, min_len: int = 5, max_len: int = 50) -> str:
    seq = "".join(ch for ch in seq.upper() if ch in AA_ALPHABET)
    if min_len <= len(seq) <= max_len:
        return seq
    return ""


def load_target_sequences_from_csv(
    csv_path: str,
    target_label: int | str,
    *,
    min_len: int = 5,
    max_len: int = 50,
) -> List[str]:
    target = canonical_target(target_label)
    sequences: List[str] = []
    if not csv_path or not os.path.exists(csv_path):
        return sequences
    with open(csv_path, newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                label = canonical_target(row.get("label", ""))
            except Exception:
                continue
            if label != target:
                continue
            seq = clean_sequence(row.get("sequence", ""), min_len=min_len, max_len=max_len)
            if seq:
                sequences.append(seq)
    return sorted(set(sequences), key=lambda s: (len(s), s))


def select_reference_sequences(sequences: List[str], max_refs: int) -> List[str]:
    unique = sorted(set(sequences), key=lambda s: (len(s), s))
    if max_refs <= 0 or len(unique) <= max_refs:
        return unique
    if max_refs == 1:
        return [unique[len(unique) // 2]]
    selected: List[str] = []
    last_idx = len(unique) - 1
    for i in range(max_refs):
        idx = int(round(i * last_idx / (max_refs - 1)))
        selected.append(unique[idx])
    return sorted(set(selected), key=lambda s: (len(s), s))


def median_length(sequences: List[str], fallback: float = 18.0) -> float:
    lengths = sorted(len(seq) for seq in sequences if seq)
    if not lengths:
        return float(fallback)
    mid = len(lengths) // 2
    if len(lengths) % 2:
        return float(lengths[mid])
    return 0.5 * (lengths[mid - 1] + lengths[mid])


def sequence_instability_index(seq: str) -> float:
    seq = clean_sequence(seq)
    if not seq:
        return 100.0
    if ProtParam is not None:
        try:
            return float(ProtParam.ProteinAnalysis(seq).instability_index())
        except Exception:
            pass
    if ProtParamData is None or not hasattr(ProtParamData, "DIWV") or len(seq) < 2:
        return 50.0
    diwv = ProtParamData.DIWV
    total = 0.0
    for a, b in zip(seq[:-1], seq[1:]):
        value = None
        if isinstance(diwv.get(a), dict):
            value = diwv.get(a, {}).get(b)
        if value is None:
            value = diwv.get(a + b, 0.0)
        total += float(value)
    return 10.0 * total / max(len(seq), 1)


def _load_macrel_model_compat(path: str):
    """Load a macrel sklearn pickle saved with sklearn 0.22.

    Handles two breaking changes between sklearn 0.22 and current versions:
      1. Tree node dtype missing 'missing_go_to_left' field (added in sklearn 1.0).
      2. RandomForestClassifier.base_estimator renamed to .estimator (sklearn 1.0).
      3. DecisionTreeClassifier missing 'monotonic_cst' attribute (added in sklearn 1.1).

    Uses a pure-Python custom Unpickler to intercept the BUILD opcode and patch
    Tree node arrays before __setstate__ is called (Cython Tree is not monkey-patchable).
    """
    import io
    from sklearn.tree._tree import NODE_DTYPE

    _PureUnpickler = pickle._Unpickler
    _BUILD = pickle.BUILD[0]

    class _CompatUnpickler(_PureUnpickler):
        def load_build(self):
            stack = self.stack
            state = stack[-1]
            stack.pop()
            inst = stack[-1]
            if type(inst).__name__ == "Tree" and isinstance(state, dict):
                nodes = state.get("nodes")
                if nodes is not None and "missing_go_to_left" not in nodes.dtype.names:
                    new_nodes = np.zeros(nodes.shape[0], dtype=NODE_DTYPE)
                    for name in nodes.dtype.names:
                        new_nodes[name] = nodes[name]
                    new_nodes["missing_go_to_left"] = 1
                    state["nodes"] = new_nodes
            setstate = getattr(inst, "__setstate__", None)
            if setstate is not None:
                setstate(state)
                return
            if isinstance(state, tuple) and len(state) == 2:
                state, slotstate = state
            else:
                slotstate = None
            if state:
                inst.__dict__.update(state)
            if slotstate:
                for k, v in slotstate.items():
                    setattr(inst, k, v)
        dispatch = _PureUnpickler.dispatch.copy()
        dispatch[_BUILD] = load_build

    def _fix_compat(obj, _visited=None):
        if _visited is None:
            _visited = set()
        if id(obj) in _visited:
            return
        _visited.add(id(obj))
        if hasattr(obj, "base_estimator") and not hasattr(obj, "estimator"):
            obj.estimator = obj.base_estimator
        if not hasattr(obj, "monotonic_cst"):
            obj.monotonic_cst = None
        if hasattr(obj, "steps"):
            for _, step in obj.steps:
                _fix_compat(step, _visited)
        if hasattr(obj, "estimators_"):
            for est in obj.estimators_:
                _fix_compat(est, _visited)

    with gzip.open(path, "rb") as f:
        data = f.read()
    model = _CompatUnpickler(io.BytesIO(data)).load()
    _fix_compat(model)
    return model

# 11 equal-width edges -> 10 bins covering [0, 1]
_CLF_BINS = np.linspace(0, 1, 11) if np is not None else [i / 10 for i in range(11)]


def compute_charge(seq_list: List[str]) -> List[float]:
    """Return per-residue charge at pH 7.4 for each sequence.

    Mirrors compute_charge() in Reinforce.py: charge_at_pH(7.4) / len(seq).
    Returns 0.0 for empty or unparseable sequences.
    """
    charges = []
    for seq in seq_list:
        if not seq:
            charges.append(0.0)
            continue
        try:
            if ProtParam is not None:
                params = ProtParam.ProteinAnalysis(seq)
                charges.append(params.charge_at_pH(7.4) / len(seq))
            else:
                pos = sum(seq.count(ch) for ch in "KRH")
                neg = sum(seq.count(ch) for ch in "DE")
                charges.append((pos - neg) / len(seq))
        except Exception:
            charges.append(0.0)
    return charges


ACTIVITY_FEATURE_VERSION = 1
ACTIVITY_FEATURE_DIM = 30
_ACTIVITY_GROUPS = {
    "positive": "KRH",
    "negative": "DE",
    "aromatic": "FWY",
    "polar": "STNQCY",
    "hydrophobic": "AILMFWV",
    "tiny": "AGSTC",
}


def _activity_feature_extras(comp: torch.Tensor, length_norm: torch.Tensor) -> torch.Tensor:
    """Build differentiable AMP activity features from amino-acid composition."""
    def frac(chars: str) -> torch.Tensor:
        idx = [AA_TO_INDEX[ch] for ch in chars]
        return comp[:, idx].sum(dim=1, keepdim=True)

    pos = frac(_ACTIVITY_GROUPS["positive"])
    neg = frac(_ACTIVITY_GROUPS["negative"])
    extras = [
        length_norm.view(-1, 1),
        pos - neg,
        pos,
        neg,
        frac(_ACTIVITY_GROUPS["aromatic"]),
        frac(_ACTIVITY_GROUPS["polar"]),
        frac(_ACTIVITY_GROUPS["hydrophobic"]),
        frac(_ACTIVITY_GROUPS["tiny"]),
        comp[:, AA_TO_INDEX["P"]].view(-1, 1),
        comp[:, AA_TO_INDEX["G"]].view(-1, 1),
    ]
    return torch.cat([comp] + extras, dim=1)


def sequence_activity_features(seq_list: List[str], *, device: str = "cpu") -> torch.Tensor:
    """Return fixed-size features for the AMP activity classifier."""
    comps = []
    lengths = []
    for seq in seq_list:
        seq = clean_sequence(seq)
        counts = torch.zeros(len(AA_ALPHABET), dtype=torch.float32)
        if seq:
            for ch in seq:
                counts[AA_TO_INDEX[ch]] += 1.0
            counts /= max(len(seq), 1)
        comps.append(counts)
        lengths.append(min(len(seq), 50) / 50.0)
    comp = torch.stack(comps, dim=0).to(device)
    length_norm = torch.tensor(lengths, dtype=torch.float32, device=device)
    return _activity_feature_extras(comp, length_norm)


def soft_activity_features(
    soft_onehot: torch.Tensor,
    token_to_aa: torch.Tensor,
) -> torch.Tensor:
    """Differentiable analogue of sequence_activity_features for TTGD logits."""
    aa_probs = torch.matmul(soft_onehot.to(token_to_aa.dtype), token_to_aa)
    aa_mass = aa_probs.sum(dim=-1)
    denom = aa_mass.sum(dim=1).clamp_min(1e-6)
    comp = aa_probs.sum(dim=1) / denom.view(-1, 1)
    length_norm = (denom / 50.0).clamp(0.0, 1.0)
    return _activity_feature_extras(comp, length_norm)


class ActivityClassifier(nn.Module):
    """Small AMP activity classifier used as a TTGD reward backend."""

    def __init__(self, input_dim: int = ACTIVITY_FEATURE_DIM, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def load_activity_classifier(path: str, *, device: str = "cpu") -> ActivityClassifier:
    if not path:
        raise ValueError("--activity-classifier-path is required for activity reward")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Activity classifier checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device)
    state = checkpoint.get("state_dict", checkpoint)
    input_dim = int(checkpoint.get("input_dim", ACTIVITY_FEATURE_DIM))
    hidden_dim = int(checkpoint.get("hidden_dim", 64))
    dropout = float(checkpoint.get("dropout", 0.0))
    model = ActivityClassifier(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


class ESMMLPClassifier(nn.Module):
    """MLP classification head on top of frozen ESM sequence embeddings."""

    def __init__(self, input_dim: int = 320, hidden_dim: int = 256, num_labels: int = 3, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ESMMLPRewardModel(nn.Module):
    """Frozen ESM encoder plus MLP head used as a differentiable TTGD reward."""

    def __init__(
        self,
        esm_model_path: str,
        *,
        hidden_dim: int = 256,
        num_labels: int = 3,
        dropout: float = 0.0,
        pooling: str = "cls",
        positive_class_index: Optional[int] = None,
        checkpoint_target_label: Optional[int] = None,
        device: str = "cpu",
    ):
        super().__init__()
        if pooling not in {"cls", "mean"}:
            raise ValueError("ESM+MLP pooling must be one of: cls, mean")
        esm_model_path = resolve_esm2_model_path(esm_model_path)
        try:
            from transformers import AutoModel, AutoTokenizer
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("transformers is required for ESM+MLP reward models") from exc
        self.tokenizer = AutoTokenizer.from_pretrained(esm_model_path, trust_remote_code=True)
        self.esm = AutoModel.from_pretrained(esm_model_path, trust_remote_code=True)
        self.pooling = pooling
        self.num_labels = int(num_labels)
        self.positive_class_index = (
            None if positive_class_index is None else int(positive_class_index)
        )
        self.checkpoint_target_label = (
            None if checkpoint_target_label is None else int(checkpoint_target_label)
        )
        for param in self.esm.parameters():
            param.requires_grad_(False)
        input_dim = int(getattr(self.esm.config, "hidden_size", 320))
        self.classifier = ESMMLPClassifier(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_labels=num_labels,
            dropout=dropout,
        )
        self.to(device)
        self.eval()

    def target_class_index(self, target_label: int) -> int:
        """Map a peptide task label to the corresponding classifier output.

        Multiclass checkpoints use the repository labels directly. Target-vs-rest
        binary checkpoints instead declare the positive output explicitly.
        """
        target_label = int(target_label)
        if self.positive_class_index is not None:
            if (
                self.checkpoint_target_label is not None
                and target_label != self.checkpoint_target_label
            ):
                raise ValueError(
                    "Binary ESM+MLP checkpoint was trained for target label "
                    f"{self.checkpoint_target_label}, but label {target_label} was requested."
                )
            index = self.positive_class_index
        else:
            index = target_label
        if not 0 <= index < self.num_labels:
            raise IndexError(
                f"Classifier output index {index} is invalid for {self.num_labels} labels."
            )
        return index

    @property
    def esm_vocab(self) -> Dict[str, int]:
        return self.tokenizer.get_vocab()

    @property
    def esm_embedding_weight(self) -> torch.Tensor:
        return self.esm.embeddings.word_embeddings.weight

    def _pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "cls":
            return hidden[:, 0, :]
        mask = attention_mask.to(hidden.dtype).unsqueeze(-1)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def logits_from_embeds(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if input_ids is None:
            token_id = self.esm_vocab.get("A", self.esm_vocab.get("<unk>", 0))
            input_ids = torch.full(
                attention_mask.shape,
                int(token_id),
                dtype=torch.long,
                device=inputs_embeds.device,
            )
        handle = self.esm.embeddings.word_embeddings.register_forward_hook(
            lambda _module, _inputs, _output: inputs_embeds
        )
        try:
            out = self.esm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
        finally:
            handle.remove()
        return self.classifier(self._pool(out.last_hidden_state, attention_mask))

    def pooled_from_embeds(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if input_ids is None:
            token_id = self.esm_vocab.get("A", self.esm_vocab.get("<unk>", 0))
            input_ids = torch.full(
                attention_mask.shape,
                int(token_id),
                dtype=torch.long,
                device=inputs_embeds.device,
            )
        handle = self.esm.embeddings.word_embeddings.register_forward_hook(
            lambda _module, _inputs, _output: inputs_embeds
        )
        try:
            out = self.esm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
        finally:
            handle.remove()
        return self._pool(out.last_hidden_state, attention_mask)

    def soft_score_and_embedding(
        self,
        soft_onehot: torch.Tensor,
        token_to_esm: torch.Tensor,
        target_label: int,
        position_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        aligned = torch.matmul(soft_onehot.to(token_to_esm.dtype), token_to_esm)
        residue_embeds = torch.matmul(aligned.to(self.esm_embedding_weight.dtype), self.esm_embedding_weight)
        B, response_length = residue_embeds.shape[:2]
        device = residue_embeds.device
        cls_id = self.esm_vocab.get("<cls>", self.esm_vocab.get("[CLS]", 0))
        eos_id = self.esm_vocab.get("<eos>", self.esm_vocab.get("[SEP]", 2))
        aa_id = self.esm_vocab.get("A", self.esm_vocab.get("<unk>", 0))
        if position_mask is None:
            body_mask = torch.ones((B, response_length), dtype=torch.bool, device=device)
        else:
            if position_mask.shape != soft_onehot.shape[:2]:
                raise ValueError("ESM position mask must have shape [batch, response_length]")
            body_mask = position_mask.to(device=device, dtype=torch.bool)
        lengths = body_mask.sum(dim=1)
        prefix = torch.arange(response_length, device=device).view(1, -1) < lengths.view(-1, 1)
        if not bool(torch.equal(body_mask, prefix)):
            raise ValueError("ESM position mask must be a contiguous active prefix")
        if B == 0:
            raise ValueError("ESM soft sequence batch cannot be empty")
        pad_id = int(self.esm_vocab.get("<pad>", self.esm_vocab.get("[PAD]", 1)))
        max_length = int(lengths.max().detach().cpu())
        full_length = max_length + 2
        cls_embedding = self.esm_embedding_weight[int(cls_id)].view(1, -1)
        eos_embedding = self.esm_embedding_weight[int(eos_id)].view(1, -1)
        embed_rows = []
        id_rows = []
        mask_rows = []
        for row, length_value in enumerate(lengths.detach().cpu().tolist()):
            length = int(length_value)
            row_embeds = torch.cat(
                [cls_embedding, residue_embeds[row, :length, :], eos_embedding], dim=0
            )
            row_ids = torch.cat(
                [
                    torch.tensor([int(cls_id)], dtype=torch.long, device=device),
                    torch.full((length,), int(aa_id), dtype=torch.long, device=device),
                    torch.tensor([int(eos_id)], dtype=torch.long, device=device),
                ],
                dim=0,
            )
            row_mask = torch.ones(length + 2, dtype=torch.long, device=device)
            padding = full_length - row_embeds.shape[0]
            if padding:
                row_embeds = F.pad(row_embeds, (0, 0, 0, padding), value=0.0)
                row_ids = F.pad(row_ids, (0, padding), value=pad_id)
                row_mask = F.pad(row_mask, (0, padding), value=0)
            embed_rows.append(row_embeds)
            id_rows.append(row_ids)
            mask_rows.append(row_mask)
        full_embeds = torch.stack(embed_rows, dim=0)
        input_ids = torch.stack(id_rows, dim=0)
        attention_mask = torch.stack(mask_rows, dim=0)
        pooled = self.pooled_from_embeds(full_embeds, attention_mask, input_ids=input_ids)
        logits = self.classifier(pooled)
        probs = torch.softmax(logits, dim=-1)[:, self.target_class_index(target_label)]
        return probs, F.normalize(pooled, dim=-1)

    def score_soft(
        self,
        soft_onehot: torch.Tensor,
        token_to_esm: torch.Tensor,
        target_label: int,
        position_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        probs, _embedding = self.soft_score_and_embedding(
            soft_onehot,
            token_to_esm,
            target_label,
            position_mask=position_mask,
        )
        return probs

    @torch.no_grad()
    def embed_sequences(self, seq_list: List[str], *, batch_size: int = 64) -> torch.Tensor:
        embeddings: List[torch.Tensor] = []
        device = next(self.parameters()).device
        for start in range(0, len(seq_list), batch_size):
            batch = [clean_sequence(seq) or "A" for seq in seq_list[start:start + batch_size]]
            toks = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=52)
            toks = {key: value.to(device) for key, value in toks.items()}
            out = self.esm(**toks, return_dict=True)
            pooled = self._pool(out.last_hidden_state, toks["attention_mask"])
            embeddings.append(F.normalize(pooled.detach(), dim=-1))
        if not embeddings:
            hidden = int(getattr(self.esm.config, "hidden_size", 320))
            return torch.empty((0, hidden), dtype=torch.float32, device=device)
        return torch.cat(embeddings, dim=0)

    @torch.no_grad()
    def score_sequences(self, seq_list: List[str], target_label: int, *, batch_size: int = 64) -> List[float]:
        scores: List[float] = []
        device = next(self.parameters()).device
        for start in range(0, len(seq_list), batch_size):
            batch = [clean_sequence(seq) or "A" for seq in seq_list[start:start + batch_size]]
            toks = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=52)
            toks = {key: value.to(device) for key, value in toks.items()}
            out = self.esm(**toks, return_dict=True)
            logits = self.classifier(self._pool(out.last_hidden_state, toks["attention_mask"]))
            probs = torch.softmax(logits, dim=-1)[:, self.target_class_index(target_label)]
            scores.extend(float(x) for x in probs.detach().cpu())
        return scores


def load_esm_mlp_reward_model(path: str, *, device: str = "cpu") -> ESMMLPRewardModel:
    if not path:
        raise ValueError("--esm-reward-model-path is required for ESM+MLP reward")
    if not os.path.exists(path):
        raise FileNotFoundError(f"ESM+MLP reward model checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device)
    model = ESMMLPRewardModel(
        checkpoint.get("esm_model_path", "ESM2-8M"),
        hidden_dim=int(checkpoint.get("hidden_dim", 256)),
        num_labels=int(checkpoint.get("num_labels", 3)),
        dropout=float(checkpoint.get("dropout", 0.0)),
        pooling=checkpoint.get("pooling", "cls"),
        positive_class_index=checkpoint.get("positive_class_index"),
        checkpoint_target_label=checkpoint.get("target_label"),
        device=device,
    )
    model.classifier.load_state_dict(checkpoint["classifier_state_dict"])
    model.to(device).eval()
    return model


class ESMMICRegressionModel(nn.Module):
    """Frozen ESM encoder with species-specific continuous MIC regressors.

    The regression target is ``log10(MIC in micromolar)``.  A single frozen
    ESM-2 encoder is shared by all organism heads so that the three AMP-MIC
    models can be evaluated in one forward pass during TTGD.  The calibrated
    reward is monotone in the predicted MIC: lower MIC yields a larger value.
    ``score_soft`` and ``score_sequences`` report an equal-weight mean across
    the configured organism heads, so the default is a broad-spectrum
    surrogate rather than a claim about any one organism.

    Checkpoints intentionally contain only the regression heads and metadata;
    the ESM backbone is resolved from ``esm_model_path`` at load time.  This
    keeps the artifact small and makes the training/evaluation provenance
    explicit.
    """

    CHECKPOINT_KIND = "peptidedto_esm_mic_regression_v1"

    def __init__(
        self,
        esm_model_path: str,
        *,
        hidden_dim: int = 256,
        species_names: Sequence[str] = ("s_aureus", "e_coli", "p_aeruginosa"),
        pooling: str = "cls",
        target_mean: Optional[Sequence[float]] = None,
        target_std: Optional[Sequence[float]] = None,
        reward_threshold: float = math.log10(32.0),
        reward_scale: float = 0.5,
        device: str = "cpu",
    ):
        super().__init__()
        if pooling not in {"cls", "mean"}:
            raise ValueError("MIC ESM pooling must be one of: cls, mean")
        if not species_names:
            raise ValueError("At least one MIC species head is required")
        if reward_scale <= 0:
            raise ValueError("MIC reward scale must be positive")
        esm_model_path = resolve_esm2_model_path(esm_model_path)
        try:
            from transformers import AutoModel, AutoTokenizer
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("transformers is required for MIC regression models") from exc
        self.tokenizer = AutoTokenizer.from_pretrained(esm_model_path, trust_remote_code=True)
        self.esm = AutoModel.from_pretrained(esm_model_path, trust_remote_code=True)
        self.pooling = pooling
        self.species_names = tuple(str(name) for name in species_names)
        self.species_to_index = {name: index for index, name in enumerate(self.species_names)}
        input_dim = int(getattr(self.esm.config, "hidden_size", 320))
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, int(hidden_dim)),
                    nn.ReLU(),
                    nn.Linear(int(hidden_dim), 1),
                )
                for _ in self.species_names
            ]
        )
        means = list(target_mean) if target_mean is not None else [0.0] * len(self.species_names)
        scales = list(target_std) if target_std is not None else [1.0] * len(self.species_names)
        if len(means) != len(self.species_names) or len(scales) != len(self.species_names):
            raise ValueError("MIC target statistics must match the number of species heads")
        self.register_buffer("target_mean", torch.tensor(means, dtype=torch.float32))
        self.register_buffer("target_std", torch.tensor(scales, dtype=torch.float32).clamp_min(1e-6))
        self.reward_threshold = float(reward_threshold)
        self.reward_scale = float(reward_scale)
        for parameter in self.esm.parameters():
            parameter.requires_grad_(False)
        self.to(device).eval()

    @property
    def esm_vocab(self) -> Dict[str, int]:
        return self.tokenizer.get_vocab()

    @property
    def esm_embedding_weight(self) -> torch.Tensor:
        return self.esm.get_input_embeddings().weight

    def _pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "cls":
            return hidden[:, 0, :]
        mask = attention_mask.to(hidden.dtype).unsqueeze(-1)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def _pooled_from_embeds(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        embedding_layer = self.esm.get_input_embeddings()
        handle = embedding_layer.register_forward_hook(
            lambda _module, _inputs, _output: inputs_embeds
        )
        try:
            out = self.esm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
        finally:
            handle.remove()
        return self._pool(out.last_hidden_state, attention_mask)

    def _head_outputs(self, pooled: torch.Tensor) -> torch.Tensor:
        normalized = torch.cat([head(pooled) for head in self.heads], dim=-1)
        mean = self.target_mean.to(normalized.device, normalized.dtype).view(1, -1)
        std = self.target_std.to(normalized.device, normalized.dtype).view(1, -1)
        return normalized * std + mean

    def _soft_pooled(
        self,
        soft_onehot: torch.Tensor,
        token_to_esm: torch.Tensor,
        position_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Pool a soft sequence using the same active prefix as hard scoring.

        TTGD keeps a fixed response tensor for batching, while a hard tokenizer
        removes post-EOS padding before assigning positional embeddings.  A
        single masked forward would leave EOS at the fixed tensor's final
        position.  We therefore place EOS immediately after each active
        prefix, pad all rows to the longest prefix, and issue one masked ESM
        forward.  This preserves gradients through the active soft embeddings
        and aligns EOS/position IDs with ``score_sequences``.
        """
        aligned = torch.matmul(soft_onehot.to(token_to_esm.dtype), token_to_esm)
        residue_embeds = torch.matmul(
            aligned.to(self.esm_embedding_weight.dtype), self.esm_embedding_weight
        )
        batch = residue_embeds.shape[0]
        response_length = residue_embeds.shape[1]
        device = residue_embeds.device
        vocab = self.esm_vocab
        cls_id = int(vocab.get("<cls>", vocab.get("[CLS]", 0)))
        eos_id = int(vocab.get("<eos>", vocab.get("[SEP]", 2)))
        aa_id = int(vocab.get("A", vocab.get("<unk>", 0)))

        if position_mask is None:
            body_mask = torch.ones(
                (batch, response_length), dtype=torch.bool, device=device
            )
        else:
            if position_mask.shape != soft_onehot.shape[:2]:
                raise ValueError(
                    "MIC position mask must have shape [batch, response_length]"
                )
            body_mask = position_mask.to(device=device, dtype=torch.bool)

        # The generator supplies a contiguous prefix mask.  Rejecting a mask
        # with holes is preferable to silently changing the hard sequence's
        # order while compacting the soft representation.
        lengths = body_mask.sum(dim=1)
        prefix = torch.arange(response_length, device=device).view(1, -1) < lengths.view(-1, 1)
        if not bool(torch.equal(body_mask, prefix)):
            raise ValueError("MIC position mask must be a contiguous active prefix")

        # Build one padded batch.  Each row places EOS immediately after its
        # active prefix; padding after EOS is masked out.  ESM derives
        # position IDs from these input IDs, so the active tokens and EOS keep
        # exactly the same positions as a compact hard-tokenized sequence.
        if batch == 0:
            raise ValueError("MIC soft sequence batch cannot be empty")
        pad_id = int(vocab.get("<pad>", vocab.get("[PAD]", 1)))
        max_length = int(lengths.max().detach().cpu())
        full_length = max_length + 2
        cls_embedding = self.esm_embedding_weight[cls_id].view(1, -1)
        eos_embedding = self.esm_embedding_weight[eos_id].view(1, -1)
        embed_rows = []
        id_rows = []
        mask_rows = []
        for row, length_value in enumerate(lengths.detach().cpu().tolist()):
            length = int(length_value)
            body = residue_embeds[row, :length, :]
            row_embeds = torch.cat([cls_embedding, body, eos_embedding], dim=0)
            row_ids = torch.cat(
                [
                    torch.tensor([cls_id], dtype=torch.long, device=device),
                    torch.full((length,), aa_id, dtype=torch.long, device=device),
                    torch.tensor([eos_id], dtype=torch.long, device=device),
                ],
                dim=0,
            )
            row_mask = torch.ones(length + 2, dtype=torch.long, device=device)
            padding = full_length - row_embeds.shape[0]
            if padding:
                row_embeds = F.pad(row_embeds, (0, 0, 0, padding), value=0.0)
                row_ids = F.pad(row_ids, (0, padding), value=pad_id)
                row_mask = F.pad(row_mask, (0, padding), value=0)
            embed_rows.append(row_embeds)
            id_rows.append(row_ids)
            mask_rows.append(row_mask)
        full_embeds = torch.stack(embed_rows, dim=0)
        input_ids = torch.stack(id_rows, dim=0)
        attention_mask = torch.stack(mask_rows, dim=0)
        pooled = self._pooled_from_embeds(
            full_embeds,
            attention_mask,
            input_ids,
        )
        return pooled

    def score_soft_log_mic(
        self,
        soft_onehot: torch.Tensor,
        token_to_esm: torch.Tensor,
        position_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return differentiable ``[batch, species]`` log10-MIC predictions."""
        return self._head_outputs(self._soft_pooled(soft_onehot, token_to_esm, position_mask))

    def reward_from_log_mic(self, log_mic: torch.Tensor) -> torch.Tensor:
        """Map log10-MIC predictions to a bounded low-MIC reward."""
        return torch.sigmoid((float(self.reward_threshold) - log_mic) / float(self.reward_scale))

    def score_soft(
        self,
        soft_onehot: torch.Tensor,
        token_to_esm: torch.Tensor,
        position_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raw = self.score_soft_log_mic(soft_onehot, token_to_esm, position_mask)
        return self.reward_from_log_mic(raw).mean(dim=-1), raw

    @torch.no_grad()
    def score_sequences_log_mic(
        self,
        seq_list: List[str],
        *,
        batch_size: int = 64,
        max_length: int = 52,
    ) -> torch.Tensor:
        values: List[torch.Tensor] = []
        device = next(self.parameters()).device
        for start in range(0, len(seq_list), batch_size):
            batch = []
            for offset, sequence in enumerate(seq_list[start:start + batch_size]):
                cleaned = clean_sequence(
                    "" if sequence is None else str(sequence),
                    min_len=1,
                    max_len=max(1, max_length - 2),
                )
                if not cleaned:
                    index = start + offset
                    raise ValueError(
                        "MIC scoring requires a non-empty canonical amino-acid "
                        f"sequence (invalid item at index {index})"
                    )
                batch.append(cleaned)
            toks = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            toks = {key: value.to(device) for key, value in toks.items()}
            out = self.esm(**toks, return_dict=True)
            values.append(self._head_outputs(self._pool(out.last_hidden_state, toks["attention_mask"])))
        if not values:
            return torch.empty((0, len(self.species_names)), device=device)
        return torch.cat(values, dim=0)

    @torch.no_grad()
    def score_sequences(
        self,
        seq_list: List[str],
        *,
        batch_size: int = 64,
        max_length: int = 52,
    ) -> Tuple[List[float], List[List[float]]]:
        raw = self.score_sequences_log_mic(seq_list, batch_size=batch_size, max_length=max_length)
        rewards = self.reward_from_log_mic(raw).mean(dim=-1)
        return rewards.detach().cpu().tolist(), raw.detach().cpu().tolist()


def load_mic_regression_model(
    path: str,
    *,
    device: str = "cpu",
    reward_threshold: Optional[float] = None,
    reward_scale: Optional[float] = None,
) -> ESMMICRegressionModel:
    """Load a continuous AMP-MIC checkpoint and validate its metadata."""
    if not path:
        raise ValueError("--mic-regression-model-path is required for MIC reward")
    if not os.path.exists(path):
        raise FileNotFoundError(f"MIC regression checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device)
    kind = checkpoint.get("checkpoint_kind")
    if kind != ESMMICRegressionModel.CHECKPOINT_KIND:
        raise ValueError(
            f"Unsupported MIC checkpoint kind {kind!r}; expected "
            f"{ESMMICRegressionModel.CHECKPOINT_KIND!r}"
        )
    model = ESMMICRegressionModel(
        checkpoint.get("esm_model_path", "ESM2-8M"),
        hidden_dim=int(checkpoint.get("hidden_dim", 256)),
        species_names=checkpoint["species_names"],
        pooling=checkpoint.get("pooling", "cls"),
        target_mean=checkpoint.get("target_mean"),
        target_std=checkpoint.get("target_std"),
        reward_threshold=float(
            checkpoint.get("reward_threshold", math.log10(32.0))
            if reward_threshold is None else reward_threshold
        ),
        reward_scale=float(
            checkpoint.get("reward_scale", 0.5)
            if reward_scale is None else reward_scale
        ),
        device=device,
    )
    if "state_dict" not in checkpoint:
        raise ValueError("MIC checkpoint is missing state_dict")
    incompatible = model.load_state_dict(checkpoint["state_dict"], strict=False)
    missing_non_backbone = [
        key for key in incompatible.missing_keys
        if not key.startswith("esm.")
    ]
    if missing_non_backbone or incompatible.unexpected_keys:
        raise ValueError(
            "Invalid MIC regression checkpoint state: "
            f"missing={missing_non_backbone[:5]} "
            f"unexpected={incompatible.unexpected_keys[:5]}"
        )
    model.to(device).eval()
    return model


class PeptideClassPrior:
    """Data-trained AMP/AFP/AVP prior used as a TTGD reward.

    The prior is intentionally lightweight and dependency-free so it can be
    trained directly from ``data/train/mulit_peptide_train.csv`` on machines
    that do not yet have the full torch/transformers environment installed.
    It captures three data signals per class:
      1. length distribution
      2. residue composition
      3. position-specific residue preference
    During TTGD this becomes a differentiable token reward by mapping each
    ProGen token to a canonical amino acid score.
    """

    VERSION = 1

    def __init__(
        self,
        *,
        min_len: int = 5,
        max_len: int = 50,
        alpha: float = 0.5,
        length_weight: float = 0.35,
        residue_weight: float = 1.0,
        position_weight: float = 0.45,
        data: Optional[dict] = None,
    ):
        self.min_len = min_len
        self.max_len = max_len
        self.alpha = alpha
        self.length_weight = length_weight
        self.residue_weight = residue_weight
        self.position_weight = position_weight
        self._token_score_cache = {}
        if data is None:
            self.class_counts = {str(k): 0 for k in LABEL_TO_TYPE}
            self.length_counts = {str(k): Counter() for k in LABEL_TO_TYPE}
            self.residue_counts = {str(k): Counter() for k in LABEL_TO_TYPE}
            self.position_counts = {
                str(k): [Counter() for _ in range(max_len)] for k in LABEL_TO_TYPE
            }
            self.background_lengths = Counter()
            self.background_residues = Counter()
            self.background_positions = [Counter() for _ in range(max_len)]
        else:
            self.class_counts = {str(k): int(v) for k, v in data["class_counts"].items()}
            self.length_counts = {
                str(k): Counter({int(kk): int(vv) for kk, vv in v.items()})
                for k, v in data["length_counts"].items()
            }
            self.residue_counts = {
                str(k): Counter({kk: int(vv) for kk, vv in v.items()})
                for k, v in data["residue_counts"].items()
            }
            self.position_counts = {
                str(k): [Counter(pos) for pos in positions]
                for k, positions in data["position_counts"].items()
            }
            self.background_lengths = Counter({
                int(k): int(v) for k, v in data["background_lengths"].items()
            })
            self.background_residues = Counter({
                k: int(v) for k, v in data["background_residues"].items()
            })
            self.background_positions = [
                Counter(pos) for pos in data["background_positions"]
            ]

    @classmethod
    def train_from_csv(
        cls,
        csv_path: str,
        *,
        min_len: int = 5,
        max_len: int = 50,
        alpha: float = 0.5,
    ) -> "PeptideClassPrior":
        prior = cls(min_len=min_len, max_len=max_len, alpha=alpha)
        with open(csv_path, newline="") as handle:
            for row in csv.DictReader(handle):
                try:
                    label = str(canonical_target(row["label"]))
                except Exception:
                    continue
                seq = clean_sequence(row.get("sequence", ""), min_len=min_len, max_len=max_len)
                if not seq:
                    continue
                prior.class_counts[label] += 1
                prior.length_counts[label][len(seq)] += 1
                prior.background_lengths[len(seq)] += 1
                for i, aa in enumerate(seq[:max_len]):
                    prior.residue_counts[label][aa] += 1
                    prior.position_counts[label][i][aa] += 1
                    prior.background_residues[aa] += 1
                    prior.background_positions[i][aa] += 1
        return prior

    @classmethod
    def load(cls, path: str) -> "PeptideClassPrior":
        with open(path) as handle:
            payload = json.load(handle)
        kwargs = payload.get("params", {})
        return cls(data=payload["data"], **kwargs)

    def save(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {
            "version": self.VERSION,
            "label_to_type": LABEL_TO_TYPE,
            "params": {
                "min_len": self.min_len,
                "max_len": self.max_len,
                "alpha": self.alpha,
                "length_weight": self.length_weight,
                "residue_weight": self.residue_weight,
                "position_weight": self.position_weight,
            },
            "data": {
                "class_counts": self.class_counts,
                "length_counts": {
                    k: dict(v) for k, v in self.length_counts.items()
                },
                "residue_counts": {
                    k: dict(v) for k, v in self.residue_counts.items()
                },
                "position_counts": {
                    k: [dict(pos) for pos in positions]
                    for k, positions in self.position_counts.items()
                },
                "background_lengths": dict(self.background_lengths),
                "background_residues": dict(self.background_residues),
                "background_positions": [dict(pos) for pos in self.background_positions],
            },
        }
        with open(path, "w") as handle:
            json.dump(payload, handle, indent=2)

    def summary(self) -> str:
        chunks = []
        for label in sorted(self.class_counts, key=int):
            chunks.append(f"{LABEL_TO_TYPE[int(label)]}:{self.class_counts[label]}")
        return " ".join(chunks)

    def _log_prob(self, count: float, total: float, bins: int) -> float:
        return math.log((count + self.alpha) / (total + self.alpha * bins))

    def _length_log_odds(self, seq_len: int, label: str) -> float:
        bins = self.max_len - self.min_len + 1
        target_total = sum(self.length_counts[label].values())
        bg_total = sum(self.background_lengths.values())
        return (
            self._log_prob(self.length_counts[label][seq_len], target_total, bins)
            - self._log_prob(self.background_lengths[seq_len], bg_total, bins)
        )

    def _residue_log_odds(self, aa: str, label: str) -> float:
        target_total = sum(self.residue_counts[label].values())
        bg_total = sum(self.background_residues.values())
        return (
            self._log_prob(self.residue_counts[label][aa], target_total, len(AA_ALPHABET))
            - self._log_prob(self.background_residues[aa], bg_total, len(AA_ALPHABET))
        )

    def _position_log_odds(self, pos: int, aa: str, label: str) -> float:
        pos = min(pos, self.max_len - 1)
        target_total = sum(self.position_counts[label][pos].values())
        bg_total = sum(self.background_positions[pos].values())
        return (
            self._log_prob(self.position_counts[label][pos][aa], target_total, len(AA_ALPHABET))
            - self._log_prob(self.background_positions[pos][aa], bg_total, len(AA_ALPHABET))
        )

    def score_sequence(
        self,
        seq: str,
        target_label: int | str,
        *,
        length_weight_override: Optional[float] = None,
    ) -> float:
        target = str(canonical_target(target_label))
        seq = clean_sequence(seq, min_len=self.min_len, max_len=self.max_len)
        if not seq:
            return -5.0
        residue_score = sum(self._residue_log_odds(aa, target) for aa in seq) / len(seq)
        position_score = sum(
            self._position_log_odds(i, aa, target) for i, aa in enumerate(seq)
        ) / len(seq)
        length_score = self._length_log_odds(len(seq), target)
        length_weight = self.length_weight if length_weight_override is None else float(length_weight_override)
        return (
            self.residue_weight * residue_score
            + self.position_weight * position_score
            + length_weight * length_score
        )

    def token_scores_for_tokenizer(
        self,
        progen_tokenizer,
        vocab_size: int,
        target_label: int | str,
        *,
        device: str = "cpu",
    ):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required to build TTGD token rewards.")
        target = str(canonical_target(target_label))
        cache_key = (id(progen_tokenizer), vocab_size, target, str(device))
        if cache_key in self._token_score_cache:
            return self._token_score_cache[cache_key]

        scores = torch.full((vocab_size,), -1.0, dtype=torch.float32, device=device)
        vocab = (
            getattr(progen_tokenizer, "vocab", None)
            or (progen_tokenizer.get_vocab() if hasattr(progen_tokenizer, "get_vocab") else {})
        )
        for tok, idx in vocab.items():
            if idx >= vocab_size:
                continue
            token = str(tok).upper()
            if len(token) == 1 and token in AA_ALPHABET:
                scores[idx] = float(self._residue_log_odds(token, target))
        self._token_score_cache[cache_key] = scores
        return scores

# ─── Vocabulary alignment: ProGen2 → ESM-2 ───────────────────────────────────
def build_vocab_alignment(progen_tokenizer, esm_alphabet, progen_vocab_size: int) -> torch.Tensor:
    """Build a [progen_vocab_size, esm_vocab_size] one-hot alignment matrix.

    Row i selects the ESM-2 token whose string representation matches ProGen2
    token i.  Tokens with no ESM-2 counterpart map to ESM-2's <unk>.

    progen_vocab_size must equal the model's actual embedding / logit dimension,
    which may differ from len(progen_tokenizer) due to padding.
    """
    esm_tok_to_idx = esm_alphabet.tok_to_idx
    unk_esm = esm_tok_to_idx.get("<unk>", 0)
    esm_vocab_size = len(esm_alphabet.all_toks)

    align = torch.zeros(progen_vocab_size, esm_vocab_size)
    for tok, progen_idx in progen_tokenizer.vocab.items():
        if progen_idx < progen_vocab_size:          # guard against out-of-range
            esm_idx = esm_tok_to_idx.get(tok, unk_esm)
            align[progen_idx, esm_idx] = 1.0
    return align

# ─── Forward hook: inject soft embeddings into ESM-2 ─────────────────────────
class SoftEmbedHook:
    """Context manager that replaces ESM-2's embed_tokens output with
    pre-computed soft embeddings.  Gradients propagate back through the hook
    to the ProGen2 logits being optimised."""

    def __init__(self, esm_model: nn.Module, soft_embeds: torch.Tensor):
        self.esm_model = esm_model
        self.soft_embeds = soft_embeds   # [B, L+2, d_esm]  (incl. CLS / EOS)
        self._handle: Optional[torch.utils.hooks.RemovableHook] = None

    def _hook(self, module, input, output):
        return self.soft_embeds

    def __enter__(self):
        self._handle = self.esm_model.embed_tokens.register_forward_hook(self._hook)
        return self

    def __exit__(self, *_):
        if self._handle is not None:
            self._handle.remove()

# ─── Trainable soft embedder ──────────────────────────────────────────────────

class AMPSoftEmbedder(nn.Module):
    """Wraps per-token ProGen2 logits as a trainable nn.Parameter and provides
    differentiable embeddings for both ProGen2 (NLL loss) and ESM-2 (reward).
    """

    def __init__(
        self,
        progen_model: nn.Module,
        progen_tokenizer,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.temperature = temperature

        # ProGen2 input embedding matrix  [progen_vocab, d_progen]
        self.register_buffer(
            "progen_embed_weight",
            get_progen_input_embeddings(progen_model).weight.detach().clone(),
        )
        # Alignment matrix  [model_vocab_size, esm_vocab]
        # Use the embedding weight's row count (model vocabulary size), rather
        # than assuming that a tokenizer wrapper's length equals the model's
        # compiled embedding size.
        model_vocab_size = self.progen_embed_weight.shape[0]
        # The current reward backends build their alignment lazily in
        # ``TTGDOptimizer._token_to_esm_matrix``. Keep the legacy attributes
        # explicit so obsolete helper paths fail with a useful message instead
        # of an opaque AttributeError.
        self.progen_to_esm_align: Optional[torch.Tensor] = None
        self.esm_embed_weight: Optional[torch.Tensor] = None

        self.logits: Optional[nn.Parameter] = None

    # ── parameter lifecycle ───────────────────────────────────────────────────

    def initialize(self, init_logits: torch.Tensor):
        if self.logits is not None:
            del self.logits
        init_logits = init_logits.detach().clone().float()
        require_finite(init_logits, "soft_embedder_init_logits")
        self.logits = nn.Parameter(init_logits)

    def deconstruct(self):
        if self.logits is not None:
            del self.logits
        self.logits = None

    # ── differentiable projections ────────────────────────────────────────────

    def _soft_onehot(self) -> torch.Tensor:
        """[1, L, progen_vocab]  via softmax."""
        probabilities = F.softmax(self.logits / self.temperature, dim=-1)
        return require_finite(probabilities, "soft_onehot")

    def _progen_embeds(self, soft_onehot: torch.Tensor) -> torch.Tensor:
        """[1, L, d_progen]  soft input embeddings for ProGen2."""
        return torch.matmul(
            soft_onehot.to(self.progen_embed_weight.dtype),
            self.progen_embed_weight,
        )

    def _esm_soft_embeds(self, soft_onehot: torch.Tensor) -> torch.Tensor:
        """[1, L, d_esm]  vocab-aligned soft embeddings for ESM-2."""
        if self.progen_to_esm_align is None or self.esm_embed_weight is None:
            raise RuntimeError(
                "Legacy AMPSoftEmbedder ESM projection is unavailable; "
                "use the ESMMLP/MIC reward backend alignment helpers."
            )
        # [1,L,progen_vocab] × [progen_vocab,esm_vocab] → [1,L,esm_vocab]
        aligned = torch.matmul(
            soft_onehot.to(self.progen_to_esm_align.dtype),
            self.progen_to_esm_align,
        )
        # [1,L,esm_vocab] × [esm_vocab,d_esm] → [1,L,d_esm]
        return torch.matmul(aligned.to(self.esm_embed_weight.dtype), self.esm_embed_weight)

    def get_argmax_sequence(self, progen_tokenizer) -> str:
        token_ids = self.logits.argmax(dim=-1).squeeze(0).tolist()
        tokens = progen_tokenizer.convert_ids_to_tokens(token_ids)
        return "".join(t for t in tokens if isinstance(t, str) and t.isalpha()).upper()

    def forward(self):
        soft_onehot = self._soft_onehot()
        return dict(
            soft_onehot=soft_onehot,
            progen_embeds=self._progen_embeds(soft_onehot),
        )

# ─── Differentiable reward via ESM-2 hook ────────────────────────────────────

def differentiable_reward(
    esm_soft_embeds: torch.Tensor,     # [B, L, d_esm]
    predict_model: "PredictModel",
    esm_alphabet,
    argmax_residue_ids: torch.Tensor,  # [B, L] long – used only for padding mask
    representation_layer: int = 33,
) -> torch.Tensor:
    """Differentiable reward via PredictModel (ESM-2 650M + MLP).

    Supports batch size B ≥ 1: all B sequences are scored in a single ESM-2
    forward pass, giving ~B× speedup over sequential calls.
    Returns sigmoid(raw_logit) in [0, 1], shape [B].
    """
    esm_model = predict_model.submodel1
    mlp = predict_model.submodel2
    device = esm_soft_embeds.device
    B = esm_soft_embeds.shape[0]

    # Fixed CLS / EOS embeddings – one per sequence in the batch
    with torch.no_grad():
        cls_emb = esm_model.embed_tokens(
            torch.full((B, 1), esm_alphabet.cls_idx, device=device)
        )  # [B, 1, d_esm]
        eos_emb = esm_model.embed_tokens(
            torch.full((B, 1), esm_alphabet.eos_idx, device=device)
        )  # [B, 1, d_esm]

    # [B, L+2, d_esm] – injected via forward hook replacing embed_tokens output
    full_soft_embeds = torch.cat([cls_emb, esm_soft_embeds, eos_emb], dim=1)

    # Token IDs only determine padding masks inside ESM-2
    cls_tok = torch.full((B, 1), esm_alphabet.cls_idx, device=device, dtype=torch.long)
    eos_tok = torch.full((B, 1), esm_alphabet.eos_idx, device=device, dtype=torch.long)
    tokens  = torch.cat([cls_tok, argmax_residue_ids, eos_tok], dim=1)  # [B, L+2]

    with SoftEmbedHook(esm_model, full_soft_embeds):
        output = esm_model(tokens, repr_layers=[representation_layer])

    # CLS token representation (index 0)
    cls_repr  = output["representations"][representation_layer][:, 0, :]  # [B, d_esm]
    raw_logit = mlp(cls_repr.to(next(mlp.parameters()).dtype)).squeeze(-1)  # [B]
    return torch.sigmoid(raw_logit)  # [B], values in [0, 1]

# ─── TTGD optimiser ───────────────────────────────────────────────────────────

class TTGDOptimizer:
    """Per-sequence logit optimiser.

    Maximises the configured composite reward together with the NLL/KL
    regularizers.  When enabled, ``mic_reward_coeff`` adds a bounded mean
    low-MIC surrogate across the three organism-specific heads; it is an
    auxiliary predicted-MIC signal rather than an experimental measurement.
    """

    def __init__(
        self,
        progen_model: nn.Module,
        progen_tokenizer,
        *,
        max_iters: int = 30,
        learning_rate: float = 0.1,
        kl_coeff: float = 0.1,
        reward_coeff: float = 1.0,
        nll_coeff: float = 0.01,
        class_prior: Optional[PeptideClassPrior] = None,
        activity_classifier: Optional[ActivityClassifier] = None,
        esm_mlp_reward_model: Optional[ESMMLPRewardModel] = None,
        mic_regression_model: Optional[ESMMICRegressionModel] = None,
        mic_reward_coeff: float = 0.0,
        reward_source: str = "prior",
        target_label: int | str = 0,
        regularizer_ref_sequences: Optional[List[str]] = None,
        regularizer_target_length: float = 18.0,
        activity_coeff: float = 1.0,
        paper_activity_coeff: float = 1.0,
        esm_activity_coeff: Optional[float] = None,
        stability_coeff: float = 0.15,
        similarity_coeff: float = 0.10,
        similarity_mode: str = "novelty",
        length_coeff: float = 0.10,
        stability_target: float = 40.0,
        stability_scale: float = 10.0,
        length_scale: float = 4.0,
        temperature: float = 1.0,
        device: str = "cuda",
        verbose: bool = False,
    ):
        self.progen_model = progen_model
        self.progen_tokenizer = progen_tokenizer
        if max_iters < 0 or learning_rate <= 0:
            raise ValueError("max_iters must be non-negative and learning_rate positive")
        for name, value in (
            ("kl_coeff", kl_coeff),
            ("reward_coeff", reward_coeff),
            ("nll_coeff", nll_coeff),
            ("activity_coeff", activity_coeff),
            ("paper_activity_coeff", paper_activity_coeff),
            ("stability_coeff", stability_coeff),
            ("similarity_coeff", similarity_coeff),
            ("length_coeff", length_coeff),
        ):
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name, value in (("temperature", temperature), ("stability_scale", stability_scale), ("length_scale", length_scale)):
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        self.max_iters = max_iters
        self.lr = learning_rate
        self.kl_coeff = kl_coeff
        self.reward_coeff = reward_coeff
        self.nll_coeff = nll_coeff
        self.class_prior = class_prior
        self.activity_classifier = activity_classifier
        self.esm_mlp_reward_model = esm_mlp_reward_model
        self.mic_regression_model = mic_regression_model
        self.mic_reward_coeff = float(mic_reward_coeff)
        self.reward_source = canonical_reward_source(reward_source)
        self.target_label = canonical_target(target_label)
        self.device = device
        self.verbose = verbose
        self._prior_token_scores = None
        self._token_to_aa = None
        # ESM-backed rewards may use different checkpoints/tokenizers.  Keep
        # one alignment per backend instead of silently reusing the first one.
        self._token_to_esm_cache = {}
        self.regularizer_ref_sequences = regularizer_ref_sequences or []
        self.regularizer_target_length = float(regularizer_target_length)
        self.activity_coeff = float(activity_coeff)
        self.paper_activity_coeff = float(paper_activity_coeff)
        self.esm_activity_coeff = float(activity_coeff if esm_activity_coeff is None else esm_activity_coeff)
        self.stability_coeff = float(stability_coeff)
        self.similarity_coeff = float(similarity_coeff)
        if similarity_mode not in SIMILARITY_MODES:
            raise ValueError(
                "similarity_mode must be one of: " + ", ".join(sorted(SIMILARITY_MODES))
            )
        self.similarity_mode = similarity_mode
        self.length_coeff = float(length_coeff)
        self.stability_target = float(stability_target)
        self.stability_scale = float(stability_scale)
        self.length_scale = float(length_scale)
        self._regularizer_ref_cache = {}
        self._instability_weight_matrix = None
        self._regularizer_ref_esm_embeddings = None

        # TGLP optimizes only latent logits. Frozen model weights still propagate
        # gradients to soft input embeddings while avoiding parameter-grad buildup.
        for parameter in self.progen_model.parameters():
            parameter.requires_grad_(False)

        self.soft_embedder = AMPSoftEmbedder(
            progen_model, progen_tokenizer,
            temperature=temperature,
        ).to(device)

    def _data_prior_reward(
        self,
        soft_onehot: torch.Tensor,
        position_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Expected target-class log-odds under the current soft token distribution."""
        B = soft_onehot.shape[0]
        if self.class_prior is None or self.reward_coeff <= 0:
            return torch.zeros((B,), device=self.device)
        if self._prior_token_scores is None:
            vocab_size = soft_onehot.shape[-1]
            self._prior_token_scores = self.class_prior.token_scores_for_tokenizer(
                self.progen_tokenizer,
                vocab_size,
                self.target_label,
                device=self.device,
            )
        token_scores = (soft_onehot * self._prior_token_scores.view(1, 1, -1)).sum(dim=-1)
        if position_mask is None:
            return token_scores.mean(dim=-1)
        if position_mask.shape != token_scores.shape:
            raise ValueError("position_mask must match soft sequence shape for prior reward")
        mask = position_mask.to(device=token_scores.device, dtype=token_scores.dtype)
        return (token_scores * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)

    def _token_to_aa_matrix(self, vocab_size: int) -> torch.Tensor:
        if self._token_to_aa is not None:
            return self._token_to_aa
        matrix = torch.zeros((vocab_size, len(AA_ALPHABET)), dtype=torch.float32, device=self.device)
        vocab = (
            getattr(self.progen_tokenizer, "vocab", None)
            or (self.progen_tokenizer.get_vocab() if hasattr(self.progen_tokenizer, "get_vocab") else {})
        )
        for tok, idx in vocab.items():
            if idx >= vocab_size:
                continue
            token = str(tok).upper()
            if len(token) == 1 and token in AA_TO_INDEX:
                matrix[idx, AA_TO_INDEX[token]] = 1.0
        self._token_to_aa = matrix
        return matrix

    def _token_to_esm_matrix(
        self,
        vocab_size: int,
        *,
        backend: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """Return a ProGen-to-ESM alignment for one concrete ESM backend.

        Activity and MIC predictors are allowed to use different ESM
        checkpoints.  The alignment therefore keys on the backend identity,
        model vocabulary size, and device; sharing a single matrix can map
        soft tokens through the wrong tokenizer without raising an error.
        """
        esm_backend = (
            backend
            if backend is not None
            else self.esm_mlp_reward_model or self.mic_regression_model
        )
        if esm_backend is None:
            raise RuntimeError("No ESM-backed reward model is loaded.")
        esm_vocab = esm_backend.esm_vocab
        esm_vocab_size = int(esm_backend.esm_embedding_weight.shape[0])
        # Keep the backend object itself in the key.  Using only ``id`` would
        # permit an id-reuse collision after a predictor is garbage-collected.
        cache = getattr(self, "_token_to_esm_cache", None)
        if cache is None:
            cache = self._token_to_esm_cache = {}
        cache_key = (esm_backend, int(vocab_size), esm_vocab_size, str(self.device))
        cached = cache.get(cache_key)
        if cached is not None:
            return cached
        default_idx = esm_vocab.get("<pad>", esm_vocab.get("<unk>", 0))
        default_idx = min(int(default_idx), max(esm_vocab_size - 1, 0))
        matrix = torch.zeros((vocab_size, esm_vocab_size), dtype=torch.float32, device=self.device)
        vocab = (
            getattr(self.progen_tokenizer, "vocab", None)
            or (self.progen_tokenizer.get_vocab() if hasattr(self.progen_tokenizer, "get_vocab") else {})
        )
        for tok, idx in vocab.items():
            if idx >= vocab_size:
                continue
            token = str(tok).upper()
            esm_idx = esm_vocab.get(token, default_idx) if len(token) == 1 else default_idx
            if int(esm_idx) < esm_vocab_size:
                matrix[idx, esm_idx] = 1.0
        cache[cache_key] = matrix
        return matrix

    def _activity_reward(self, soft_onehot: torch.Tensor) -> torch.Tensor:
        B = soft_onehot.shape[0]
        if self.activity_classifier is None:
            return torch.zeros((B,), device=self.device)
        token_to_aa = self._token_to_aa_matrix(soft_onehot.shape[-1])
        features = soft_activity_features(soft_onehot, token_to_aa)
        logits = self.activity_classifier(features)
        return torch.sigmoid(logits)

    def _esm_mlp_reward(
        self,
        soft_onehot: torch.Tensor,
        position_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = soft_onehot.shape[0]
        if self.esm_mlp_reward_model is None:
            return torch.zeros((B,), device=self.device)
        token_to_esm = self._token_to_esm_matrix(
            soft_onehot.shape[-1], backend=self.esm_mlp_reward_model
        )
        return self.esm_mlp_reward_model.score_soft(
            soft_onehot,
            token_to_esm,
            self.target_label,
            position_mask=position_mask,
        )

    def _esm_mlp_reward_and_embedding(
        self,
        soft_onehot: torch.Tensor,
        position_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B = soft_onehot.shape[0]
        if self.esm_mlp_reward_model is None:
            return torch.zeros((B,), device=self.device), None
        token_to_esm = self._token_to_esm_matrix(
            soft_onehot.shape[-1], backend=self.esm_mlp_reward_model
        )
        return self.esm_mlp_reward_model.soft_score_and_embedding(
            soft_onehot,
            token_to_esm,
            self.target_label,
            position_mask=position_mask,
        )

    def _mic_reward(
        self,
        soft_onehot: torch.Tensor,
        position_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Differentiable bounded reward for low predicted MIC."""
        B = soft_onehot.shape[0]
        if self.mic_regression_model is None:
            return torch.zeros((B,), device=self.device)
        token_to_esm = self._token_to_esm_matrix(
            soft_onehot.shape[-1], backend=self.mic_regression_model
        )
        reward, _raw_log_mic = self.mic_regression_model.score_soft(
            soft_onehot,
            token_to_esm,
            position_mask=position_mask,
        )
        return require_finite(reward, "soft_mic_reward")

    def _soft_aa_probs(self, soft_onehot: torch.Tensor) -> torch.Tensor:
        token_to_aa = self._token_to_aa_matrix(soft_onehot.shape[-1])
        return torch.matmul(soft_onehot.to(token_to_aa.dtype), token_to_aa)

    def _instability_matrix(self) -> Optional[torch.Tensor]:
        if self._instability_weight_matrix is not None:
            return self._instability_weight_matrix
        if ProtParamData is None or not hasattr(ProtParamData, "DIWV"):
            return None
        diwv = ProtParamData.DIWV
        matrix = torch.zeros((len(AA_ALPHABET), len(AA_ALPHABET)), dtype=torch.float32, device=self.device)
        for a, i in AA_TO_INDEX.items():
            for b, j in AA_TO_INDEX.items():
                value = None
                if isinstance(diwv.get(a), dict):
                    value = diwv.get(a, {}).get(b)
                if value is None:
                    value = diwv.get(a + b, 0.0)
                matrix[i, j] = float(value)
        self._instability_weight_matrix = matrix
        return matrix

    def _soft_stability_reward(self, aa_probs: torch.Tensor) -> torch.Tensor:
        B = aa_probs.shape[0]
        matrix = self._instability_matrix()
        if matrix is None or aa_probs.shape[1] < 2:
            return torch.zeros((B,), device=self.device)
        pair_scores = torch.einsum("bli,ij,blj->bl", aa_probs[:, :-1, :], matrix, aa_probs[:, 1:, :])
        soft_length = aa_probs.sum(dim=(1, 2)).clamp_min(1.0)
        instability = 10.0 * pair_scores.sum(dim=1) / soft_length
        return torch.sigmoid((self.stability_target - instability) / self.stability_scale)

    def _reference_profiles(self, length: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        cache_key = (length, str(self.device))
        if cache_key in self._regularizer_ref_cache:
            return self._regularizer_ref_cache[cache_key]
        if not self.regularizer_ref_sequences:
            value = (None, None)
            self._regularizer_ref_cache[cache_key] = value
            return value
        profiles = []
        lengths = []
        for seq in self.regularizer_ref_sequences:
            seq = clean_sequence(seq, max_len=max(length, 50))
            if not seq:
                continue
            profile = torch.zeros((length, len(AA_ALPHABET)), dtype=torch.float32, device=self.device)
            for pos, aa in enumerate(seq[:length]):
                profile[pos, AA_TO_INDEX[aa]] = 1.0
            profiles.append(profile)
            lengths.append(min(len(seq), length))
        if not profiles:
            value = (None, None)
        else:
            value = (
                torch.stack(profiles, dim=0),
                torch.tensor(lengths, dtype=torch.float32, device=self.device).clamp_min(1.0),
            )
        self._regularizer_ref_cache[cache_key] = value
        return value

    def _soft_reference_identity_reward(self, aa_probs: torch.Tensor) -> torch.Tensor:
        B, length, _ = aa_probs.shape
        profiles, ref_lengths = self._reference_profiles(length)
        if profiles is None or ref_lengths is None:
            return torch.zeros((B,), device=self.device)
        ref_mask = profiles.sum(dim=-1)
        matches = torch.einsum("bla,rla->brl", aa_probs, profiles)
        identity = (matches * ref_mask.view(1, profiles.shape[0], length)).sum(dim=-1)
        identity = identity / ref_lengths.view(1, -1)
        max_identity = identity.max(dim=1).values
        return max_identity.clamp(0.0, 1.0)

    def _soft_novelty_reward(self, aa_probs: torch.Tensor) -> torch.Tensor:
        return 1.0 - self._soft_reference_identity_reward(aa_probs)

    def _soft_similarity_reward(self, aa_probs: torch.Tensor) -> torch.Tensor:
        return self._soft_reference_identity_reward(aa_probs)

    def _reference_esm_embeddings(self) -> Optional[torch.Tensor]:
        if self._regularizer_ref_esm_embeddings is not None:
            return self._regularizer_ref_esm_embeddings
        if self.esm_mlp_reward_model is None or not self.regularizer_ref_sequences:
            return None
        refs = [seq for seq in self.regularizer_ref_sequences if clean_sequence(seq)]
        if not refs:
            return None
        self._regularizer_ref_esm_embeddings = self.esm_mlp_reward_model.embed_sequences(
            refs,
            batch_size=64,
        ).to(self.device)
        return self._regularizer_ref_esm_embeddings

    def _soft_esm_similarity_reward(self, soft_embedding: Optional[torch.Tensor]) -> torch.Tensor:
        if soft_embedding is None:
            return torch.zeros((1,), device=self.device)
        ref_embeddings = self._reference_esm_embeddings()
        if ref_embeddings is None or ref_embeddings.numel() == 0:
            return torch.zeros((soft_embedding.shape[0],), device=self.device)
        cosine = torch.matmul(F.normalize(soft_embedding, dim=-1), ref_embeddings.T)
        return cosine.max(dim=1).values.clamp(-1.0, 1.0)

    def _soft_length_reward(self, aa_probs: torch.Tensor) -> torch.Tensor:
        soft_length = aa_probs.sum(dim=(1, 2))
        z = (soft_length - self.regularizer_target_length) / self.length_scale
        return torch.exp(-(z ** 2))

    def _regularized_esm_reward(
        self,
        soft_onehot: torch.Tensor,
        esm_mlp_reward: torch.Tensor,
        soft_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        aa_probs = self._soft_aa_probs(soft_onehot)
        reward = self.esm_activity_coeff * esm_mlp_reward
        if self.stability_coeff:
            reward = reward + self.stability_coeff * self._soft_stability_reward(aa_probs)
        if self.similarity_coeff:
            if self.similarity_mode == "esm_similarity":
                reward = reward + self.similarity_coeff * self._soft_esm_similarity_reward(soft_embedding)
            elif self.similarity_mode == "similarity":
                reward = reward + self.similarity_coeff * self._soft_similarity_reward(aa_probs)
            else:
                reward = reward + self.similarity_coeff * self._soft_novelty_reward(aa_probs)
        if self.length_coeff:
            reward = reward + self.length_coeff * self._soft_length_reward(aa_probs)
        return reward

    def _optimisation_reward(
        self,
        soft_onehot: torch.Tensor,
        position_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = soft_onehot.shape[0]
        prior_reward = self._data_prior_reward(soft_onehot, position_mask=position_mask)
        activity_reward = self._activity_reward(soft_onehot)
        weighted_activity_reward = self.paper_activity_coeff * activity_reward
        soft_embedding = None
        if self.reward_source in REGULARIZED_REWARD_SOURCES and self.similarity_mode == "esm_similarity":
            esm_mlp_reward, soft_embedding = self._esm_mlp_reward_and_embedding(
                soft_onehot, position_mask=position_mask
            )
        else:
            if self.esm_mlp_reward_model is None:
                esm_mlp_reward = torch.zeros((B,), device=self.device)
            else:
                token_to_esm = self._token_to_esm_matrix(
                    soft_onehot.shape[-1], backend=self.esm_mlp_reward_model
                )
                esm_mlp_reward = self.esm_mlp_reward_model.score_soft(
                    soft_onehot,
                    token_to_esm,
                    self.target_label,
                    position_mask=position_mask,
                )
        regularized_esm_reward = None
        if self.reward_source == "activity":
            reward = activity_reward
        elif self.reward_source == "prior+activity":
            reward = prior_reward + activity_reward
        elif self.reward_source == "esm_mlp":
            reward = esm_mlp_reward
        elif self.reward_source == "prior+esm_mlp":
            reward = prior_reward + esm_mlp_reward
        elif self.reward_source == "activity+esm_mlp":
            reward = weighted_activity_reward + self.esm_activity_coeff * esm_mlp_reward
        elif self.reward_source == "prior+activity+esm_mlp":
            reward = prior_reward + weighted_activity_reward + self.esm_activity_coeff * esm_mlp_reward
        elif self.reward_source in REGULARIZED_REWARD_SOURCES:
            regularized_esm_reward = self._regularized_esm_reward(
                soft_onehot,
                esm_mlp_reward,
                soft_embedding,
            )
            if self.reward_source == "esm_mlp_reg":
                reward = regularized_esm_reward
            elif self.reward_source == "prior+esm_mlp_reg":
                reward = prior_reward + regularized_esm_reward
            elif self.reward_source == "activity+esm_mlp_reg":
                reward = weighted_activity_reward + regularized_esm_reward
            else:
                reward = prior_reward + weighted_activity_reward + regularized_esm_reward
        else:
            reward = prior_reward
        if self.mic_regression_model is not None:
            reward = reward + self.mic_reward_coeff * self._mic_reward(
                soft_onehot,
                position_mask=position_mask,
            )
        return reward

    def _argmax_esm_ids(self, logits: torch.Tensor) -> torch.Tensor:
        """Map argmax ProGen2 logits → ESM-2 vocab indices.

        Supports both [B, L, V] (batched) and [1, L, V] inputs.
        Returns [B, L].
        """
        if self.soft_embedder.progen_to_esm_align is None:
            raise RuntimeError(
                "Legacy ESM-id projection is unavailable; use the backend-specific "
                "token alignment created by _token_to_esm_matrix."
            )
        progen_ids = logits.argmax(dim=-1)         # [B, L]
        B, L = progen_ids.shape
        flat = progen_ids.view(-1)               # [B*L]
        esm_ids = self.soft_embedder.progen_to_esm_align[flat].argmax(dim=-1)
        return esm_ids.view(B, L)                  # [B, L]

    def optimize(
        self,
        prefix_ids: torch.Tensor,    # [B, prefix_len]
        init_logits: torch.Tensor,   # [B, L, progen_vocab_size]
        position_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """Run max_iters gradient steps on B sequences simultaneously.

        All B sequences share a single Adam step and a single ESM-2 forward
        pass per iteration, giving ~B× throughput on the reward model.

        Returns:
            logits:       [B, L, V]  best logits found per sequence
            best_rewards: [B]        corresponding reward values (sigmoid)
        """
        init_logits = init_logits.to(self.device).float()
        require_finite(init_logits, "optimizer_init_logits")
        B = init_logits.shape[0]
        self.soft_embedder.initialize(init_logits)

        optimizer = torch.optim.Adam([self.soft_embedder.logits], lr=self.lr)
        best_logits = init_logits.detach().clone().to(self.device) # [B, L, V], on device
        init_log_probs = F.log_softmax(init_logits, dim=-1)
        require_finite(init_log_probs, "optimizer_init_log_probs")
        best_rewards = torch.full((B,), -float("inf"), device=self.device)  # [B]

        for step in range(self.max_iters):
            optimizer.zero_grad(set_to_none=True)

            out = self.soft_embedder()
            soft_onehot  = out["soft_onehot"]    # [B, L, V_progen]
            progen_embeds = out["progen_embeds"]  # [B, L, d_progen]
            require_finite(progen_embeds, f"progen_embeds_step_{step}")

            # NLL: log P_ProGen2(y_soft | prefix)
            nll = torch.tensor(0.0, device=self.device)
            if self.nll_coeff > 0:
                prefix_emb = get_progen_input_embeddings(self.progen_model)(prefix_ids)
                full_emb = torch.cat([prefix_emb, progen_embeds], dim=1)
                lm_out = self.progen_model(inputs_embeds=full_emb)
                resp_logits = lm_out.logits[:, prefix_ids.shape[1] - 1: -1, :]
                require_finite(resp_logits, f"nll_response_logits_step_{step}")
                log_probs_lm = F.log_softmax(resp_logits, dim=-1)
                token_nll = -(log_probs_lm * soft_onehot).sum(dim=-1)
                if position_mask is None:
                    active_mask = torch.ones_like(token_nll, dtype=token_nll.dtype)
                else:
                    if position_mask.shape != token_nll.shape:
                        raise ValueError(
                            "position_mask must match the response tensor shape for NLL"
                        )
                    active_mask = position_mask.to(
                        device=token_nll.device, dtype=token_nll.dtype
                    )
                nll = (token_nll * active_mask).sum() / active_mask.sum().clamp_min(1.0)
                require_finite(nll, f"nll_step_{step}")

            kl = torch.tensor(0.0, device=self.device)
            if self.kl_coeff > 0:
                kl = categorical_reverse_kl(self.soft_embedder.logits, init_log_probs)

            reward = self._optimisation_reward(soft_onehot, position_mask=position_mask)
            require_finite(reward, f"reward_step_{step}")
            loss = self.nll_coeff * nll + self.kl_coeff * kl - self.reward_coeff * reward.mean()
            require_finite(loss, f"loss_step_{step}")

            # The proxy describes the current logits, so save them before Adam moves.
            proxy = reward.detach() - self.kl_coeff * kl.detach()
            require_finite(proxy, f"proxy_step_{step}")
            improved = proxy > best_rewards
            best_rewards[improved] = proxy[improved]
            if improved.any():
                best_logits[improved] = self.soft_embedder.logits.detach()[improved]

            loss.backward()
            require_finite(self.soft_embedder.logits.grad, f"logit_gradient_step_{step}")
            torch.nn.utils.clip_grad_norm_([self.soft_embedder.logits], max_norm=10.0)
            optimizer.step()
            require_finite(self.soft_embedder.logits, f"optimized_logits_step_{step}")

        # The final Adam update has not yet been scored inside the loop.
        with torch.no_grad():
            final_out = self.soft_embedder()
            final_reward = self._optimisation_reward(
                final_out["soft_onehot"],
                position_mask=position_mask,
            )
            final_kl = categorical_reverse_kl(self.soft_embedder.logits, init_log_probs)
            final_proxy = final_reward - self.kl_coeff * final_kl
            require_finite(final_proxy, "final_proxy")
            improved = final_proxy > best_rewards
            best_rewards[improved] = final_proxy[improved]
            if improved.any():
                best_logits[improved] = self.soft_embedder.logits.detach()[improved]

        require_finite(best_logits, "best_logits")
        self.soft_embedder.deconstruct()
        del optimizer
        torch.cuda.empty_cache()
        return {
            "logits": best_logits.to(self.device),
            "best_rewards": best_rewards,
            "finite_steps": self.max_iters,
        }

# ─── Sampling helpers ─────────────────────────────────────────────────────────

@torch.no_grad()
def sample_from_logits(
    logits: torch.Tensor,
    temperature: float = 0.8,
    top_p: float = 0.95,
    top_k: int = 50,
) -> torch.Tensor:
    """Sample token IDs from [B, L, V] logits.  Returns [B, L].

    Fully vectorised: no Python loop over positions or batch items.
    """
    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, length, vocabulary]")
    if temperature <= 0 or not math.isfinite(float(temperature)):
        raise ValueError("temperature must be finite and positive")
    if not 0 < float(top_p) <= 1:
        raise ValueError("top_p must lie in (0, 1]")
    if top_k < 0:
        raise ValueError("top_k must be non-negative")
    require_finite(logits, "resampling_input_logits")
    B, L, V = logits.shape
    scaled = logits / max(temperature, 1e-6)   # [B, L, V]
    if 0 < top_k < V:
        threshold = torch.topk(scaled, top_k, dim=-1).values[..., -1:]
        scaled = scaled.masked_fill(scaled < threshold, -float("inf"))
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(scaled, dim=-1, descending=True)
        cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)  # [B, L, V]
        # Remove tokens where cumulative prob exceeds top_p (shift by 1 to keep the first)
        remove = cum > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        # Map back to original vocab order
        remove_orig = torch.zeros_like(scaled, dtype=torch.bool).scatter_(-1, sorted_idx, remove)
        scaled = scaled.masked_fill(remove_orig, -float("inf"))
    probs = F.softmax(scaled, dim=-1)          # [B, L, V]
    require_finite(probs, "resampling_probabilities")
    ids = torch.multinomial(probs.view(B * L, V), 1).view(B, L)
    return ids                                  # [B, L]

# ─── Main generator ───────────────────────────────────────────────────────────

class TTGDPeptideGenerator:
    """Test-Time Gradient Descent generator for AMP/AFP/AVP targets.

    Per sequence:
      1. Sample initial rollout from ProGen2
      2. Run n_ttgd_rounds of: optimise logits → resample → accept/reject
    """

    def __init__(
        self,
        progen_model: nn.Module,
        progen_tokenizer,
        *,
        min_length: int = 5,
        max_length: int = 50,
        n_ttgd_iters: int = 30,
        learning_rate: float = 0.1,
        kl_coeff: float = 0.1,
        reward_coeff: float = 1.0,
        nll_coeff: float = 0.01,
        resample_temperature: float = 0.8,
        resample_top_p: float = 0.95,
        resample_top_k: int = 50,
        rollout_temperature: float = 0.8,
        rollout_top_p: float = 0.95,
        rollout_top_k: int = 50,
        max_attempt_factor: int = 20,
        seed: int = 20260725,
        use_rejection_sampling: bool = True,
        gen_batch_size: int = 8,
        device: str = "cuda",
        verbose: int = 1,
        target_type: str | int = "amp",
        class_prior: Optional[PeptideClassPrior] = None,
        reward_source: str = "prior",
        activity_classifier_path: Optional[str] = None,
        esm_reward_model_path: Optional[str] = None,
        regularizer_ref_sequences: Optional[List[str]] = None,
        regularizer_target_length: float = 18.0,
        activity_coeff: float = 1.0,
        paper_activity_coeff: float = 1.0,
        esm_activity_coeff: Optional[float] = None,
        stability_coeff: float = 0.15,
        similarity_coeff: float = 0.10,
        similarity_mode: str = "novelty",
        length_coeff: float = 0.10,
        stability_target: float = 40.0,
        stability_scale: float = 10.0,
        length_scale: float = 4.0,
        use_external_amp_rewards: bool = False,
        mic_model_path: str = "MIC_predictor/esm_mic_predictor/mic_predictor.pt",
        macrel_model_path: Optional[str] = None,
        mic_regression_model_path: Optional[str] = None,
        mic_reward_coeff: float = 0.0,
        mic_reward_threshold: Optional[float] = None,
        mic_reward_scale: Optional[float] = None,
        hard_prior_length_weight: Optional[float] = None,
        hard_charge_coeff: float = 0.2,
    ):
        if min_length < 1 or max_length < min_length:
            raise ValueError("require 1 <= min_length <= max_length")
        if n_ttgd_iters < 0 or learning_rate <= 0:
            raise ValueError("n_ttgd_iters must be non-negative and learning_rate positive")
        for name, value in (
            ("kl_coeff", kl_coeff),
            ("reward_coeff", reward_coeff),
            ("nll_coeff", nll_coeff),
            ("activity_coeff", activity_coeff),
            ("paper_activity_coeff", paper_activity_coeff),
            ("stability_coeff", stability_coeff),
            ("similarity_coeff", similarity_coeff),
            ("length_coeff", length_coeff),
        ):
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name, value in (
            ("resample_temperature", resample_temperature),
            ("rollout_temperature", rollout_temperature),
            ("stability_scale", stability_scale),
            ("length_scale", length_scale),
        ):
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name, value in (("resample_top_p", resample_top_p), ("rollout_top_p", rollout_top_p)):
            if not 0 < float(value) <= 1:
                raise ValueError(f"{name} must lie in (0, 1]")
        if resample_top_k < 0 or rollout_top_k < 0:
            raise ValueError("top-k values must be non-negative")
        if max_attempt_factor < 1 or gen_batch_size < 1:
            raise ValueError("max_attempt_factor and gen_batch_size must be positive")
        self.progen_model = progen_model.to(device)
        self.progen_tokenizer = progen_tokenizer
        self.min_length = int(min_length)
        self.max_length = max_length
        self.resample_temperature = resample_temperature
        self.resample_top_p = resample_top_p
        self.resample_top_k = int(resample_top_k)
        self.rollout_temperature = float(rollout_temperature)
        self.rollout_top_p = float(rollout_top_p)
        self.rollout_top_k = int(rollout_top_k)
        self.max_attempt_factor = int(max_attempt_factor)
        self.seed = int(seed)
        self.use_rejection_sampling = use_rejection_sampling
        self.gen_batch_size = gen_batch_size
        self.device = device
        self.verbose = verbose
        self.target_label = canonical_target(target_type)
        self.target_type = LABEL_TO_TYPE[self.target_label]
        self.class_prior = class_prior
        reward_source = canonical_reward_source(reward_source)
        if reward_source not in REWARD_SOURCES:
            raise ValueError(
                "reward_source must be one of: prior, activity, prior+activity, "
                "esm_mlp, prior+esm_mlp, activity+esm_mlp, prior+activity+esm_mlp, "
                "esm_mlp_reg, prior+esm_mlp_reg, activity+esm_mlp_reg, "
                "prior+activity+esm_mlp_reg"
            )
        if reward_source in ACTIVITY_REWARD_SOURCES and self.target_label != 0:
            raise ValueError("Activity-classifier reward is currently defined for AMP only.")
        self.reward_source = reward_source
        self.regularizer_ref_sequences = regularizer_ref_sequences or []
        self.regularizer_target_length = float(regularizer_target_length)
        self.activity_coeff = float(activity_coeff)
        self.paper_activity_coeff = float(paper_activity_coeff)
        self.esm_activity_coeff = float(activity_coeff if esm_activity_coeff is None else esm_activity_coeff)
        self.stability_coeff = float(stability_coeff)
        self.similarity_coeff = float(similarity_coeff)
        if similarity_mode not in SIMILARITY_MODES:
            raise ValueError(
                "similarity_mode must be one of: " + ", ".join(sorted(SIMILARITY_MODES))
            )
        self.similarity_mode = similarity_mode
        self.length_coeff = float(length_coeff)
        self.stability_target = float(stability_target)
        self.stability_scale = float(stability_scale)
        self.length_scale = float(length_scale)
        self.hard_prior_length_weight = (
            None if hard_prior_length_weight is None else float(hard_prior_length_weight)
        )
        self.hard_charge_coeff = float(hard_charge_coeff)
        self._regularizer_ref_esm_embeddings = None
        self.mic_reward_coeff = float(mic_reward_coeff)
        if self.mic_reward_coeff < 0:
            raise ValueError("mic_reward_coeff must be non-negative")
        self.activity_classifier = None
        if reward_source in ACTIVITY_REWARD_SOURCES:
            self.activity_classifier = load_activity_classifier(
                activity_classifier_path or "save_model/amp_activity_classifier.pt",
                device=device,
            )
        self.esm_mlp_reward_model = None
        if reward_source in ESM_REWARD_SOURCES:
            self.esm_mlp_reward_model = load_esm_mlp_reward_model(
                esm_reward_model_path or "save_model/esm_mlp_reward.pt",
                device=device,
            )
        self.mic_regression_model = None
        if self.mic_reward_coeff > 0 or mic_regression_model_path:
            if self.target_label != 0:
                raise ValueError("Continuous MIC reward is defined for AMP generation only.")
            if not mic_regression_model_path:
                raise ValueError(
                    "mic_regression_model_path is required when mic_reward_coeff is positive"
                )
            self.mic_regression_model = load_mic_regression_model(
                mic_regression_model_path,
                device=device,
                reward_threshold=mic_reward_threshold,
                reward_scale=mic_reward_scale,
            )
        self.use_external_amp_rewards = use_external_amp_rewards and self.target_label == 0
        self.mic_model = None
        self.macrel_model = None
        if self.use_external_amp_rewards:
            if ApexMIC is None or macrel_predictor is None:
                raise RuntimeError(
                    "AMP external rewards requested, but MIC_predictor/macrel_predictor "
                    "could not be imported."
                )
            self.mic_model = ApexMIC()
            self.mic_model.load_state_dict(torch.load(mic_model_path, map_location=device))
            self.mic_model.to(device).eval()
            self.macrel_model = _load_macrel_model_compat(macrel_model_path)

        self.ttgd = TTGDOptimizer(
            progen_model, progen_tokenizer,
            max_iters=n_ttgd_iters,
            learning_rate=learning_rate,
            kl_coeff=kl_coeff,
            reward_coeff=reward_coeff,
            nll_coeff=nll_coeff,
            class_prior=class_prior,
            activity_classifier=self.activity_classifier,
            esm_mlp_reward_model=self.esm_mlp_reward_model,
            mic_regression_model=self.mic_regression_model,
            mic_reward_coeff=self.mic_reward_coeff,
            reward_source=reward_source,
            target_label=self.target_label,
            regularizer_ref_sequences=self.regularizer_ref_sequences,
            regularizer_target_length=self.regularizer_target_length,
            activity_coeff=self.activity_coeff,
            paper_activity_coeff=self.paper_activity_coeff,
            esm_activity_coeff=self.esm_activity_coeff,
            stability_coeff=self.stability_coeff,
            similarity_coeff=self.similarity_coeff,
            similarity_mode=self.similarity_mode,
            length_coeff=self.length_coeff,
            stability_target=self.stability_target,
            stability_scale=self.stability_scale,
            length_scale=self.length_scale,
            device=device,
            verbose=(verbose >= 2),
        )

    def _rollout_batch(self, prefix_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample B sequences from ProGen2 in a single batched call.

        Sampling uses top-p only to choose token IDs. A separate teacher-forced
        forward pass obtains raw autoregressive logits before any sampling
        processor or warper. Those finite raw logits initialize TGLP.

        Args:
            prefix_ids: [B, prefix_len]

        Returns:
            response_ids:    [B, max_length]  – generated tokens (padded to max_length)
            response_logits: [B, max_length, vocab_size]
        """
        self.progen_model.eval()
        B = prefix_ids.shape[0]
        pad_id = self.progen_tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.progen_tokenizer.eos_token_id
        if pad_id is None:
            pad_id = 0
        attention_mask = torch.ones_like(prefix_ids, device=self.device)
        with torch.no_grad():
            bad_words_ids = []
            for token in ("B", "O", "U", "X", "Z"):
                ids = self.progen_tokenizer.encode(token, add_special_tokens=False)
                if ids and ids not in bad_words_ids:
                    bad_words_ids.append(ids)
            gen_ids = self.progen_model.generate(
                input_ids=prefix_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_length,
                do_sample=True,
                temperature=self.rollout_temperature,
                top_p=self.rollout_top_p,
                top_k=self.rollout_top_k,
                bad_words_ids=bad_words_ids or None,
                pad_token_id=pad_id,
                use_cache=True,
            )
            response_ids = gen_ids[:, prefix_ids.shape[1]:]  # [B, L_actual]

            # Causal logits at index prefix_len - 1 + j predict response token j.
            full_attention_mask = (gen_ids != pad_id).long()
            raw_out = self.progen_model(
                input_ids=gen_ids,
                attention_mask=full_attention_mask,
                use_cache=False,
            )
            start = prefix_ids.shape[1] - 1
            response_logits = raw_out.logits[
                :, start : start + response_ids.shape[1], :
            ].float()
            require_finite(response_logits, "raw_rollout_logits")

            # Pad / truncate to exactly max_length so TTGD sees a fixed shape
            L_actual = response_ids.shape[1]
            if L_actual < self.max_length:
                pad_tok = torch.full(
                    (B, self.max_length - L_actual), pad_id,
                    device=self.device, dtype=torch.long,
                )
                pad_lgts = torch.zeros(
                    (B, self.max_length - L_actual, response_logits.shape[-1]),
                    device=self.device,
                )
                response_ids = torch.cat([response_ids, pad_tok], dim=1)
                response_logits = torch.cat([response_logits, pad_lgts], dim=1)
            else:
                response_ids = response_ids[:, :self.max_length]
                response_logits = response_logits[:, :self.max_length]

        require_finite(response_logits, "padded_raw_rollout_logits")
        return response_ids, response_logits  # [B, max_length], [B, max_length, V]

    def _decode(self, token_ids: torch.Tensor) -> str:
        tokens = self.progen_tokenizer.convert_ids_to_tokens(
            token_ids.squeeze(0).tolist()
        )
        residues: List[str] = []
        for token in tokens:
            token = str(token).upper() if isinstance(token, str) else ""
            if len(token) != 1 or token not in AA_ALPHABET:
                break
            residues.append(token)
        raw = "".join(residues)
        return clean_sequence(raw, min_len=0, max_len=self.max_length)

    def _position_mask(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Mark the amino-acid prefix before the first special/pad token.

        TTGD keeps a fixed response tensor for batching.  The mask prevents
        post-EOS padding logits from contributing to the continuous MIC
        predictor while retaining a static shape for autograd.
        """
        masks = torch.zeros(
            token_ids.shape,
            dtype=torch.long,
            device=token_ids.device,
        )
        vocab = (
            getattr(self.progen_tokenizer, "vocab", None)
            or self.progen_tokenizer.get_vocab()
        )
        aa_ids = {
            int(index)
            for token, index in vocab.items()
            if len(str(token)) == 1 and str(token).upper() in AA_ALPHABET
        }
        for row in range(token_ids.shape[0]):
            active = True
            for col in range(token_ids.shape[1]):
                token_id = int(token_ids[row, col].item())
                if active and token_id in aa_ids:
                    masks[row, col] = 1
                else:
                    active = False
        return masks

    @torch.no_grad()
    def _hard_stability_reward(self, seq: str) -> float:
        instability = sequence_instability_index(seq)
        return 1.0 / (1.0 + math.exp(-(self.stability_target - instability) / self.stability_scale))

    @torch.no_grad()
    def _hard_reference_identity_reward(self, seq: str) -> float:
        seq = clean_sequence(seq, max_len=self.max_length)
        if not seq or not self.regularizer_ref_sequences:
            return 0.0
        max_identity = 0.0
        for ref in self.regularizer_ref_sequences:
            ref = clean_sequence(ref, max_len=max(self.max_length, len(ref)))
            if not ref:
                continue
            denom = max(min(len(ref), self.max_length), 1)
            matches = 0
            for pos, aa in enumerate(ref[:self.max_length]):
                if pos < len(seq) and seq[pos] == aa:
                    matches += 1
            max_identity = max(max_identity, matches / denom)
        return max(0.0, min(1.0, max_identity))

    @torch.no_grad()
    def _hard_novelty_reward(self, seq: str) -> float:
        return 1.0 - self._hard_reference_identity_reward(seq)

    @torch.no_grad()
    def _hard_similarity_reward(self, seq: str) -> float:
        return self._hard_reference_identity_reward(seq)

    @torch.no_grad()
    def _reference_esm_embeddings(self) -> Optional[torch.Tensor]:
        if self._regularizer_ref_esm_embeddings is not None:
            return self._regularizer_ref_esm_embeddings
        if self.esm_mlp_reward_model is None or not self.regularizer_ref_sequences:
            return None
        refs = [seq for seq in self.regularizer_ref_sequences if clean_sequence(seq)]
        if not refs:
            return None
        self._regularizer_ref_esm_embeddings = self.esm_mlp_reward_model.embed_sequences(
            refs,
            batch_size=64,
        ).to(self.device)
        return self._regularizer_ref_esm_embeddings

    @torch.no_grad()
    def _hard_esm_similarity_rewards(self, seqs: List[str]) -> List[float]:
        if self.esm_mlp_reward_model is None or not seqs:
            return [0.0] * len(seqs)
        ref_embeddings = self._reference_esm_embeddings()
        if ref_embeddings is None or ref_embeddings.numel() == 0:
            return [0.0] * len(seqs)
        seq_embeddings = self.esm_mlp_reward_model.embed_sequences(seqs, batch_size=64).to(self.device)
        cosine = torch.matmul(F.normalize(seq_embeddings, dim=-1), ref_embeddings.T)
        values = cosine.max(dim=1).values.clamp(-1.0, 1.0).detach().cpu().tolist()
        return [float(value) for value in values]

    @torch.no_grad()
    def _hard_length_reward(self, seq: str) -> float:
        z = (len(seq) - self.regularizer_target_length) / self.length_scale
        return math.exp(-(z ** 2))

    @torch.no_grad()
    def _regularized_hard_reward(
        self,
        seq: str,
        class_prob: float,
        esm_similarity: Optional[float] = None,
    ) -> float:
        reward = self.esm_activity_coeff * float(class_prob)
        if self.stability_coeff:
            reward += self.stability_coeff * self._hard_stability_reward(seq)
        if self.similarity_coeff:
            if self.similarity_mode == "esm_similarity":
                reward += self.similarity_coeff * float(esm_similarity or 0.0)
            elif self.similarity_mode == "similarity":
                reward += self.similarity_coeff * self._hard_similarity_reward(seq)
            else:
                reward += self.similarity_coeff * self._hard_novelty_reward(seq)
        if self.length_coeff:
            reward += self.length_coeff * self._hard_length_reward(seq)
        return reward

    @torch.no_grad()
    def _reward_batch(self, sequences: List[str]) -> List[float]:
        """Hard-sequence reward for accept/reject after resampling."""
        rewards_final = [-1.0] * len(sequences)
        valid = [(i, s) for i, s in enumerate(sequences) if s]
        if not valid:
            return rewards_final
        idxs, seqs = zip(*valid)

        rewards = []
        charge = compute_charge(list(seqs))
        activity_scores = None
        if self.activity_classifier is not None:
            features = sequence_activity_features(list(seqs), device=self.device)
            activity_scores = torch.sigmoid(self.activity_classifier(features)).detach().cpu().tolist()
        esm_mlp_scores = None
        if self.esm_mlp_reward_model is not None:
            esm_mlp_scores = self.esm_mlp_reward_model.score_sequences(
                list(seqs), self.target_label
            )
        esm_similarity_scores = None
        if self.similarity_mode == "esm_similarity" and self.esm_mlp_reward_model is not None:
            esm_similarity_scores = self._hard_esm_similarity_rewards(list(seqs))
        mic_scores = None
        mic_log10_values = None
        if self.mic_regression_model is not None:
            mic_scores, mic_log10_values = self.mic_regression_model.score_sequences(
                list(seqs),
                batch_size=max(1, min(64, len(seqs))),
                max_length=self.max_length + 2,
            )
        for i, seq in enumerate(seqs):
            prior_score = (
                self.class_prior.score_sequence(
                    seq,
                    self.target_label,
                    length_weight_override=self.hard_prior_length_weight,
                )
                if self.class_prior is not None
                else 0.0
            )
            # Mild charge bonus is useful for AMP-like peptides and harmlessly
            # small for AFP/AVP compared with class-prior reward.
            prior_reward = float(prior_score + self.hard_charge_coeff * charge[i])
            activity_reward = float(activity_scores[i]) if activity_scores is not None else 0.0
            esm_mlp_reward = float(esm_mlp_scores[i]) if esm_mlp_scores is not None else 0.0
            weighted_activity_reward = self.paper_activity_coeff * activity_reward
            if self.reward_source == "activity":
                rewards.append(activity_reward)
            elif self.reward_source == "prior+activity":
                rewards.append(prior_reward + activity_reward)
            elif self.reward_source == "esm_mlp":
                rewards.append(esm_mlp_reward)
            elif self.reward_source == "prior+esm_mlp":
                rewards.append(prior_reward + esm_mlp_reward)
            elif self.reward_source == "activity+esm_mlp":
                rewards.append(weighted_activity_reward + self.esm_activity_coeff * esm_mlp_reward)
            elif self.reward_source == "prior+activity+esm_mlp":
                rewards.append(prior_reward + weighted_activity_reward + self.esm_activity_coeff * esm_mlp_reward)
            elif self.reward_source == "esm_mlp_reg":
                rewards.append(self._regularized_hard_reward(
                    seq,
                    esm_mlp_reward,
                    esm_similarity_scores[i] if esm_similarity_scores is not None else None,
                ))
            elif self.reward_source == "prior+esm_mlp_reg":
                rewards.append(prior_reward + self._regularized_hard_reward(
                    seq,
                    esm_mlp_reward,
                    esm_similarity_scores[i] if esm_similarity_scores is not None else None,
                ))
            elif self.reward_source == "activity+esm_mlp_reg":
                rewards.append(weighted_activity_reward + self._regularized_hard_reward(
                    seq,
                    esm_mlp_reward,
                    esm_similarity_scores[i] if esm_similarity_scores is not None else None,
                ))
            elif self.reward_source == "prior+activity+esm_mlp_reg":
                rewards.append(prior_reward + weighted_activity_reward + self._regularized_hard_reward(
                    seq,
                    esm_mlp_reward,
                    esm_similarity_scores[i] if esm_similarity_scores is not None else None,
                ))
            else:
                rewards.append(prior_reward)

            if mic_scores is not None:
                rewards[-1] += self.mic_reward_coeff * float(mic_scores[i])

        if self.use_external_amp_rewards:
            macrel_reward = macrel_predictor(list(seqs), self.macrel_model)
            with torch.no_grad():
                mic_reward = self.mic_model(list(seqs)).detach().cpu().numpy()
            if getattr(mic_reward, "ndim", 1) == 0:
                mic_reward = mic_reward.reshape(1)
            for i in range(len(rewards)):
                rewards[i] += float(macrel_reward[i]) * 0.1 + float(mic_reward[i])

        for i, orig_idx in enumerate(idxs):
            rewards_final[orig_idx] = float(rewards[i])
        if self.verbose >= 2:
            print("rewards_final=", rewards_final)
        return rewards_final

    def generate(
        self,
        prompt: str,
        n_sequences: int = 10,
        n_ttgd_rounds: int = 3,
    ) -> List[dict]:
        """Generate `n_sequences` target-class peptide candidates using batched TTGD.

        Processes `gen_batch_size` sequences in parallel:
          - One batched ProGen2 generate call  (rollout)
          - One batched ESM-2 forward per TTGD step  (reward)
          - One batched ProGen2 forward per TTGD step if nll_coeff > 0

        Returns a list of dicts with keys:
            sequence, reward, accepted_rounds, total_rounds
        """
        if n_sequences < 1:
            raise ValueError("n_sequences must be positive")
        if n_ttgd_rounds < 0:
            raise ValueError("n_ttgd_rounds must be non-negative")
        prefix_ids = self.progen_tokenizer.encode(
            prompt, add_special_tokens=True, return_tensors="pt"
        ).to(self.device)  # [1, prefix_len]; prompt is '1' (ProGen2 family token)

        set_generation_seed(self.seed)
        B = self.gen_batch_size
        results: List[dict] = []
        attempted = 0
        max_attempted = max(n_sequences, n_sequences * self.max_attempt_factor)

        desc = f"TTGD {self.target_type} generation"
        progress = tqdm(total=n_sequences, desc=desc, disable=(self.verbose == 0))
        batch_idx = 0
        while len(results) < n_sequences and attempted < max_attempted:
            b = min(B, n_sequences - len(results))
            attempted += b
            batch_idx += 1
            batch_prefix = prefix_ids.expand(b, -1)   # [b, prefix_len]

            if self.verbose >= 1:
                lo = len(results) + 1
                hi = lo + b - 1
                print(f"\n[batch {batch_idx}  requested seqs {lo}-{hi}] rollout...",
                      flush=True)

            # ── Step 1: batched rollout ───────────────────────────────────────
            response_ids, init_logits = self._rollout_batch(batch_prefix)  # [b,L], [b,L,V]
            current_seqs = [self._decode(response_ids[i:i+1]) for i in range(b)]
            current_logits = init_logits                                    # [b, L, V]
            current_position_mask = self._position_mask(response_ids)
            current_rewards = self._reward_batch(current_seqs)             # [b]

            if self.verbose >= 1:
                for i, (s, r) in enumerate(zip(current_seqs, current_rewards)):
                    print(f"seq {len(results)+i+1}: init reward={r:+.4f}  "
                          f"len={len(s):3d}  {s[:20]}", flush=True)

            n_accepted = [0] * b

            # ── Steps 2-4: TTGD rounds ────────────────────────────────────────
            for rnd in range(n_ttgd_rounds):
                if self.verbose >= 1:
                    print(f"[round {rnd+1}/{n_ttgd_rounds}] optimising {b} seqs...",
                          flush=True)

                opt = self.ttgd.optimize(
                    batch_prefix,
                    current_logits,
                    position_mask=current_position_mask,
                )
                optimised_logits = opt["logits"]                            # [b, L, V]

                new_ids = sample_from_logits(
                    optimised_logits,
                    temperature=self.resample_temperature,
                    top_p=self.resample_top_p,
                    top_k=self.resample_top_k,
                )                                                           # [b, L]
                new_seqs = [self._decode(new_ids[i:i+1]) for i in range(b)]
                new_position_mask = self._position_mask(new_ids)
                new_rewards = self._reward_batch(new_seqs)                  # [b]

                # Per-sequence accept/reject
                for i in range(b):
                    improved = new_rewards[i] > current_rewards[i]
                    accept = bool(new_seqs[i]) and (
                        not self.use_rejection_sampling or improved
                    )
                    if accept:
                        current_seqs[i] = new_seqs[i]
                        current_rewards[i] = new_rewards[i]
                        current_logits = current_logits.clone()
                        current_logits[i] = optimised_logits[i]
                        current_position_mask = current_position_mask.clone()
                        current_position_mask[i] = new_position_mask[i]
                        n_accepted[i] += 1
                    if self.verbose >= 1:
                        flag = "acc" if accept else "rej"
                        print(f" seq {len(results)+i+1} [{flag}] "
                              f"reward={new_rewards[i]:+.4f}  {new_seqs[i][:20]}",
                              flush=True)

            for i in range(b):
                sequence = clean_sequence(
                    current_seqs[i],
                    min_len=self.min_length,
                    max_len=self.max_length,
                )
                if not sequence:
                    continue
                results.append({
                    "implementation": "PeptideDTO_raw_logits_strict_finite",
                    "target_type": self.target_type,
                    "target_label": self.target_label,
                    "sequence": sequence,
                    "reward": current_rewards[i],
                    "accepted_rounds": n_accepted[i],
                    "total_rounds": n_ttgd_rounds,
                    "seed": self.seed,
                })
                progress.update(1)
                if len(results) >= n_sequences:
                    break

        progress.close()
        if len(results) != n_sequences:
            raise RuntimeError(
                f"Generated {len(results)}/{n_sequences} valid sequences in "
                f"{attempted} attempts; required length range is "
                f"{self.min_length}-{self.max_length}."
            )

        return results


# Public PeptideDTO names plus backwards-compatible historical aliases.
PeptideDTOOptimizer = TTGDOptimizer
PeptideDTOGenerator = TTGDPeptideGenerator
TTGDAMPGenerator = TTGDPeptideGenerator

def main():
    parser = argparse.ArgumentParser(
        description="PeptideDTO with raw autoregressive logits and strict finite checks"
    )

    # -- data prior training -------------------------------------------------
    parser.add_argument("--train-prior", action="store_true",
                        help="Train the lightweight AMP/AFP/AVP data prior and exit.")
    parser.add_argument("--data-csv", default="data/train/mulit_peptide_train.csv",
                        help="CSV with sequence,label columns used for prior training.")
    parser.add_argument("--prior-output", default="save_model/ttgd_peptide_prior.json",
                        help="Where --train-prior writes the trained prior.")
    parser.add_argument("--prior-path", default="save_model/ttgd_peptide_prior.json",
                        help="Trained prior JSON used during generation.")
    parser.add_argument("--activity-classifier-path", default="save_model/amp_activity_classifier.pt",
                        help="AMP activity classifier checkpoint used when --reward-source includes activity.")
    parser.add_argument("--esm-reward-model-path", default="save_model/esm_mlp_reward.pt",
                        help="ESM+MLP reward checkpoint used when --reward-source includes esm_mlp.")
    parser.add_argument("--target-type", default="amp",
                        choices=sorted(TYPE_TO_LABEL),
                        help="Target peptide type for generation.")

    # ── models ──────────────────────────────────────────────────────────────
    parser.add_argument("--base-model", required=True,
                        help="Local ProGen2-xlarge model directory or Hugging Face model ID")
    parser.add_argument(
        "--model-dtype",
        default="float32",
        choices=("float32", "float16", "bfloat16"),
        help="Frozen ProGen2 weight dtype; optimized latent logits always remain float32.",
    )
    parser.add_argument(
        "--tokenizer-path",
        required=True,
        help=(
            "Path to ProGen2 tokenizer.json (preferred); a legacy vocab.txt "
            "is accepted only as a compatibility fallback"
        ),
    )
    parser.add_argument("--lora-checkpoint", default=None,
                        help="LoRA checkpoint .pt (optional)")
    parser.add_argument("--mic-model-path", default=None,
                        help="MIC predictor checkpoint (.pt)")
    parser.add_argument("--macrel-model-path", default=None,
                        help="Macrel AMP.pkl.gz path")
    parser.add_argument("--use-external-amp-rewards", action="store_true",
                        help="Also use MIC/macrel AMP rewards when target-type is AMP.")
    parser.add_argument(
        "--mic-regression-model-path",
        default=None,
        help="Continuous AMP-MIC ESM+MLP checkpoint predicting log10(MIC in micromolar).",
    )
    parser.add_argument(
        "--mic-reward-coeff",
        type=float,
        default=0.0,
        help="Coefficient for the differentiable low-MIC reward (AMP only).",
    )
    parser.add_argument(
        "--mic-reward-threshold",
        type=float,
        default=None,
        help="Optional log10(MIC) reward threshold; otherwise use checkpoint metadata.",
    )
    parser.add_argument(
        "--mic-reward-scale",
        type=float,
        default=None,
        help="Optional log10(MIC) reward temperature; otherwise use checkpoint metadata.",
    )

    # ── generation ──────────────────────────────────────────────────────────
    parser.add_argument("--prompt", default="1",
                        help="ProGen2 generation prompt")
    parser.add_argument("--n-sequences", type=int, default=10000)
    parser.add_argument("--n-ttgd-rounds", "--n-optimization-rounds", type=int, default=3,
                        dest="n_ttgd_rounds",
                        help="Optimise-resample rounds per sequence")
    parser.add_argument("--min-length", type=int, default=5,
                        help="Minimum valid peptide length")
    parser.add_argument("--max-length", type=int, default=50,
                        help="Max generated tokens per sequence")
    parser.add_argument("--max-attempt-factor", type=int, default=20,
                        help="Maximum candidates attempted per requested valid peptide")

    # ── TTGD hyper-parameters ───────────────────────────────────────────────
    parser.add_argument("--n-ttgd-iters", "--n-gradient-steps", type=int, default=20,
                        dest="n_ttgd_iters",
                        help="Gradient steps per TTGD round")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--kl-coeff", type=float, default=0.1,
                        help="KL regularisation weight")
    parser.add_argument("--reward-coeff", type=float, default=2.0,
                        help="Reward loss weight")
    parser.add_argument("--reward-source", default="prior",
                        choices=sorted(REWARD_SOURCES | set(REWARD_SOURCE_ALIASES)),
                        help="Reward backend used by TTGD optimisation and accept/reject.")
    parser.add_argument("--activity-coeff", type=float, default=1.0,
                        help="Backward-compatible alias for --esm-activity-coeff.")
    parser.add_argument("--paper-activity-coeff", type=float, default=1.0,
                        help="Weight for the paper AMP activity-classifier probability in combined reward sources.")
    parser.add_argument("--esm-activity-coeff", type=float, default=None,
                        help="Weight for the ESM+MLP target-class probability; defaults to --activity-coeff.")
    parser.add_argument("--stability-coeff", type=float, default=0.15,
                        help="Weight for stability reward in esm_mlp_reg reward sources.")
    parser.add_argument("--similarity-coeff", type=float, default=0.10,
                        help="Weight for reference-sequence distribution regularization in esm_mlp_reg reward sources.")
    parser.add_argument("--similarity-mode", choices=sorted(SIMILARITY_MODES), default="novelty",
                        help="Use novelty, position-wise similarity, or ESM embedding similarity against real same-class references.")
    parser.add_argument("--length-coeff", type=float, default=0.10,
                        help="Weight for target-length reward in esm_mlp_reg reward sources.")
    parser.add_argument("--stability-target", type=float, default=40.0,
                        help="Instability-index target; lower-instability sequences score higher.")
    parser.add_argument("--stability-scale", type=float, default=10.0,
                        help="Temperature for converting instability index to a stability reward.")
    parser.add_argument("--target-length", type=float, default=None,
                        help="Preferred peptide length for length regularisation; defaults to target-class median.")
    parser.add_argument("--length-scale", type=float, default=4.0,
                        help="Tolerance around --target-length for length regularisation.")
    parser.add_argument("--regularizer-reference-csv", default=None,
                        help="CSV used for similarity/length regularizers; defaults to --data-csv.")
    parser.add_argument("--regularizer-max-references", type=int, default=256,
                        help="Maximum same-class training sequences used by the similarity regularizer.")
    parser.add_argument("--nll-coeff", type=float, default=0.01,
                        help="ProGen2 NLL loss weight")
    parser.add_argument("--resample-temperature", type=float, default=0.8)
    parser.add_argument("--resample-top-p", type=float, default=0.95)
    parser.add_argument("--resample-top-k", type=int, default=50)
    parser.add_argument("--rollout-temperature", type=float, default=0.8)
    parser.add_argument("--rollout-top-p", type=float, default=0.95)
    parser.add_argument("--rollout-top-k", type=int, default=50)
    parser.add_argument("--no-rejection-sampling", action="store_true",
                        help="Always accept resampled sequence regardless of reward")
    parser.add_argument("--gen-batch-size", type=int, default=8,
                        help="Sequences processed in parallel per TTGD step (default 8). "
                             "Higher = faster but more GPU memory.")

    # ── misc ────────────────────────────────────────────────────────────────
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--output", default="ttgd_results.json")
    parser.add_argument("--verbose", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument(
        "--hard-prior-length-weight",
        type=float,
        default=None,
        help="Optional override for the length-prior coefficient in hard acceptance.",
    )
    parser.add_argument(
        "--hard-charge-coeff",
        type=float,
        default=0.2,
        help="Charge coefficient in the composite hard acceptance score.",
    )

    args = parser.parse_args()
    if not 1 <= args.min_length <= args.max_length:
        parser.error("Require 1 <= --min-length <= --max-length")
    esm_activity_coeff = (
        args.activity_coeff if args.esm_activity_coeff is None else args.esm_activity_coeff
    )

    if args.train_prior:
        prior = PeptideClassPrior.train_from_csv(args.data_csv)
        prior.save(args.prior_output)
        print(f"Trained TTGD peptide prior from {args.data_csv}")
        print(f"Class counts: {prior.summary()}")
        print(f"Saved prior -> {args.prior_output}")
        return

    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch is not installed in this Python environment. "
            "You can still run --train-prior, but TTGD generation requires "
            "the project's torch/transformers environment."
        )
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    set_generation_seed(args.seed)
    print(f"Device: {device}", flush=True)

    class_prior = None
    if args.prior_path:
        
        if not os.path.exists(args.prior_path):
            print(f"Prior not found at {args.prior_path}; training it from {args.data_csv}.")
            class_prior = PeptideClassPrior.train_from_csv(args.data_csv)
            class_prior.save(args.prior_path)
        else:
            class_prior = PeptideClassPrior.load(args.prior_path)
        print(f"Loaded TTGD prior ({class_prior.summary()}) from {args.prior_path}")
    target_label = canonical_target(args.target_type)
    print(f"Target: {LABEL_TO_TYPE[target_label]} (label {target_label})", flush=True)
    reward_source = canonical_reward_source(args.reward_source)

    regularizer_ref_sequences: List[str] = []
    regularizer_target_length = args.target_length
    if reward_source in REGULARIZED_REWARD_SOURCES:
        ref_csv = args.regularizer_reference_csv or args.data_csv
        all_ref_sequences = load_target_sequences_from_csv(
            ref_csv,
            target_label,
            max_len=max(args.max_length, 50),
        )
        if regularizer_target_length is None:
            regularizer_target_length = min(median_length(all_ref_sequences, fallback=18.0), args.max_length)
        regularizer_ref_sequences = select_reference_sequences(
            all_ref_sequences,
            args.regularizer_max_references,
        )
        print(
            "Regularized reward: "
            f"{esm_activity_coeff:g}*class_prob "
            f"+ {args.stability_coeff:g}*stability "
            f"+ {args.similarity_coeff:g}*{args.similarity_mode} "
            f"+ {args.length_coeff:g}*length | "
            f"target_length={regularizer_target_length:.2f} | "
            f"references={len(regularizer_ref_sequences)} from {ref_csv}",
            flush=True,
        )
    if reward_source in ACTIVITY_REWARD_SOURCES and reward_source in ESM_REWARD_SOURCES:
        print(
            "Combined activity reward: "
            f"{args.paper_activity_coeff:g}*paper_activity "
            f"+ {esm_activity_coeff:g}*esm_mlp_activity",
            flush=True,
        )

    print("Loading ProGen2...", flush=True)
    model_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.model_dtype]
    progen_tokenizer, progen_model = load_pretrained_progen_model(
        args.base_model,
        args.tokenizer_path,
        device_map=device,
        torch_dtype=model_dtype,
        lora_checkpoint=args.lora_checkpoint,
        inference_mode=False,   # gradients required for TTGD
    )#progen_model: PeftModelForCausalLM
    progen_model = progen_model.to(device)

    generator = TTGDPeptideGenerator(
        progen_model, progen_tokenizer,
        min_length=args.min_length,
        max_length=args.max_length,
        n_ttgd_iters=args.n_ttgd_iters,
        learning_rate=args.lr,
        kl_coeff=args.kl_coeff,
        reward_coeff=args.reward_coeff,
        nll_coeff=args.nll_coeff,
        resample_temperature=args.resample_temperature,
        resample_top_p=args.resample_top_p,
        resample_top_k=args.resample_top_k,
        rollout_temperature=args.rollout_temperature,
        rollout_top_p=args.rollout_top_p,
        rollout_top_k=args.rollout_top_k,
        max_attempt_factor=args.max_attempt_factor,
        seed=args.seed,
        use_rejection_sampling=not args.no_rejection_sampling,
        gen_batch_size=args.gen_batch_size,
        device=device,
        verbose=args.verbose,
        target_type=args.target_type,
        class_prior=class_prior,
        reward_source=reward_source,
        activity_classifier_path=args.activity_classifier_path,
        esm_reward_model_path=args.esm_reward_model_path,
        regularizer_ref_sequences=regularizer_ref_sequences,
        regularizer_target_length=regularizer_target_length or 18.0,
        activity_coeff=args.activity_coeff,
        paper_activity_coeff=args.paper_activity_coeff,
        esm_activity_coeff=esm_activity_coeff,
        stability_coeff=args.stability_coeff,
        similarity_coeff=args.similarity_coeff,
        similarity_mode=args.similarity_mode,
        length_coeff=args.length_coeff,
        stability_target=args.stability_target,
        stability_scale=args.stability_scale,
        length_scale=args.length_scale,
        use_external_amp_rewards=args.use_external_amp_rewards,
        mic_model_path=args.mic_model_path,
        macrel_model_path=args.macrel_model_path,
        mic_regression_model_path=args.mic_regression_model_path,
        mic_reward_coeff=args.mic_reward_coeff,
        mic_reward_threshold=args.mic_reward_threshold,
        mic_reward_scale=args.mic_reward_scale,
        hard_prior_length_weight=args.hard_prior_length_weight,
        hard_charge_coeff=args.hard_charge_coeff,
    )

    results = generator.generate(
        prompt=args.prompt,
        n_sequences=args.n_sequences,
        n_ttgd_rounds=args.n_ttgd_rounds,
    )

    # ── summary ───────────────────────────────────────────────────────────────
    PASS_THRESHOLD = 0.0
    passed = [r for r in results if r["reward"] > PASS_THRESHOLD]
    rewards = [r["reward"] for r in results]
    print(f"\n{'='*65}", flush=True)
    print(f"Generated {len(results)} sequences | {len(passed)} passed (reward > {PASS_THRESHOLD})")
    if rewards:
        print(
            f"Reward:  mean={sum(rewards)/len(rewards):.4f}  "
            f"max={max(rewards):.4f}  min={min(rewards):.4f}"
        )
    print("=" * 65)
    for i, r in enumerate(results):
        flag = "+" if r["reward"] > 0 else "-"
        print(
            f"[{i+1:3d}] [{flag}] reward={r['reward']:+.4f}  "
            f"acc={r['accepted_rounds']}/{r['total_rounds']}  "
            f"{r['sequence']}"
        )

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
