"""Model-loading helpers used by PeptideDTO training and generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Sequence, Tuple

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM, BertTokenizer


class ProGen2Tokenizer:
    """Small, explicit adapter for the tokenizer shipped with ProGen2.

    The released ProGen2 checkpoint contains a ``tokenizer.json`` with a
    30-token byte-level BPE vocabulary.  The historical PeptideDTO
    ``assets/vocab.txt`` is a 25-token BERT vocabulary and therefore cannot be
    used with that checkpoint: all control and residue IDs after ``[PAD]`` are
    shifted.  This adapter keeps the checkpoint's IDs authoritative while
    exposing the subset of the Hugging Face tokenizer API used by generation.

    ``add_special_tokens`` is deliberately opt-in.  ProGen2's family prompt
    (normally ``"1"``) is a real vocabulary token, not a BOS marker, so adding
    BOS/EOS implicitly would change the autoregressive conditioning sequence.
    """

    def __init__(self, tokenizer_json: str | Path):
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:  # pragma: no cover - dependency error path
            raise ImportError(
                "The ProGen2 tokenizer adapter requires the `tokenizers` package."
            ) from exc

        path = Path(tokenizer_json)
        if not path.is_file():
            raise FileNotFoundError(f"ProGen2 tokenizer.json not found: {path}")
        self._tokenizer = Tokenizer.from_file(str(path))
        self.vocab = dict(self._tokenizer.get_vocab())
        self._id_to_token = {idx: tok for tok, idx in self.vocab.items()}

        self.pad_token = "<|pad|>"
        self.bos_token = "<|bos|>"
        self.eos_token = "<|eos|>"
        missing = [tok for tok in (self.pad_token, self.bos_token, self.eos_token)
                   if tok not in self.vocab]
        if missing:
            raise ValueError(
                f"ProGen2 tokenizer is missing required special tokens: {missing}"
            )
        self.pad_token_id = int(self.vocab[self.pad_token])
        self.bos_token_id = int(self.vocab[self.bos_token])
        self.eos_token_id = int(self.vocab[self.eos_token])
        self.unk_token = "<|unk|>" if "<|unk|>" in self.vocab else None
        self.unk_token_id = (
            int(self.vocab[self.unk_token]) if self.unk_token is not None else None
        )

    def __len__(self) -> int:
        return max(self._id_to_token, default=-1) + 1

    def get_vocab(self) -> Dict[str, int]:
        return dict(self.vocab)

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        return_tensors: str | None = None,
    ):
        encoding = self._tokenizer.encode(
            str(text), add_special_tokens=add_special_tokens
        )
        ids = list(encoding.ids)
        if return_tensors is None:
            return ids
        if return_tensors != "pt":
            raise ValueError(f"Unsupported return_tensors={return_tensors!r}; use 'pt'.")
        return torch.tensor([ids], dtype=torch.long)

    def batch_encode_plus(
        self,
        texts: Iterable[str],
        *,
        add_special_tokens: bool = False,
        padding: bool = False,
        return_tensors: str | None = None,
        **_: Any,
    ):
        encoded = [
            self.encode(text, add_special_tokens=add_special_tokens)
            for text in texts
        ]
        if padding:
            width = max((len(ids) for ids in encoded), default=0)
            encoded = [
                ids + [self.pad_token_id] * (width - len(ids)) for ids in encoded
            ]
        if return_tensors is None:
            return {"input_ids": encoded}
        if return_tensors != "pt":
            raise ValueError(f"Unsupported return_tensors={return_tensors!r}; use 'pt'.")
        input_ids = torch.tensor(encoded, dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": (input_ids != self.pad_token_id).long(),
        }

    def __call__(self, texts, **kwargs):
        if isinstance(texts, str):
            return self.batch_encode_plus([texts], **kwargs)
        return self.batch_encode_plus(texts, **kwargs)

    def convert_ids_to_tokens(self, ids):
        if isinstance(ids, int):
            return self._id_to_token.get(ids, self.unk_token or "<|unk|>")
        return [self.convert_ids_to_tokens(int(idx)) for idx in ids]

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            if tokens in self.vocab:
                return int(self.vocab[tokens])
            if self.unk_token_id is None:
                raise KeyError(f"Unknown ProGen2 token: {tokens!r}")
            return self.unk_token_id
        return [self.convert_tokens_to_ids(token) for token in tokens]


def resolve_progen2_tokenizer_path(tokenizer_path: str | Path) -> Tuple[Path, bool]:
    """Return a tokenizer file and whether it is the checkpoint-native format."""
    path = Path(tokenizer_path)
    if path.is_dir():
        json_path = path / "tokenizer.json"
        vocab_path = path / "vocab.txt"
    elif path.name == "tokenizer.json":
        json_path = path
        vocab_path = path.with_name("vocab.txt")
    else:
        json_path = path.with_name("tokenizer.json")
        vocab_path = path
    if json_path.is_file():
        return json_path, True
    if vocab_path.is_file():
        return vocab_path, False
    raise FileNotFoundError(
        f"No ProGen2 tokenizer found at {tokenizer_path}; expected tokenizer.json "
        "or a legacy vocab.txt."
    )


def _unwrap_progen_model(model: torch.nn.Module) -> torch.nn.Module:
    """Find the underlying ProGen2 module through optional PEFT wrappers."""
    seen = set()
    current = model
    while id(current) not in seen:
        seen.add(id(current))
        if hasattr(current, "transformer") and hasattr(current.transformer, "wte"):
            return current
        if hasattr(current, "wte"):
            return current
        next_model = None
        for attr in ("base_model", "model", "module"):
            candidate = getattr(current, attr, None)
            if candidate is not None and candidate is not current:
                next_model = candidate
                break
        if next_model is None:
            break
        current = next_model
    raise AttributeError("Could not locate ProGen2 transformer.wte through model wrappers")


def get_progen_input_embeddings(model: torch.nn.Module) -> torch.nn.Module:
    """Unified input embedding accessor for native and PEFT-wrapped ProGen2."""
    try:
        embeddings = model.get_input_embeddings()
        if embeddings is not None:
            return embeddings
    except (AttributeError, NotImplementedError):
        pass
    base = _unwrap_progen_model(model)
    return base.transformer.wte if hasattr(base, "transformer") else base.wte


def get_progen_output_embeddings(model: torch.nn.Module) -> torch.nn.Module:
    """Unified LM-head accessor for native and PEFT-wrapped ProGen2."""
    try:
        embeddings = model.get_output_embeddings()
        if embeddings is not None:
            return embeddings
    except (AttributeError, NotImplementedError):
        pass
    base = _unwrap_progen_model(model)
    if not hasattr(base, "lm_head"):
        raise AttributeError("Could not locate ProGen2 lm_head through model wrappers")
    return base.lm_head


def _progen_classes():
    try:
        from progen2hf.models import ProGenConfig, ProGenForCausalLM
    except ImportError:
        # The pinned RL-PLM checkout exposes ``progen2hf`` while the
        # standalone ProGen2 reference checkout used on lab149 exposes the
        # same classes as ``progen2.models.progen``.  Both implementations
        # share the checkpoint config/model contract.
        try:
            from progen2.models.progen.configuration_progen import ProGenConfig
            from progen2.models.progen.modeling_progen import ProGenForCausalLM
        except ImportError as exc:
            raise ImportError(
                "PeptideDTO requires a ProGen2 implementation. Add the pinned "
                "RL-PLM amp_design path or the standalone progen2 checkout to "
                "PYTHONPATH."
            ) from exc
    return ProGenConfig, ProGenForCausalLM


def rename_legacy_lora_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Map historical ProGen2 LoRA keys to current PEFT names."""
    renamed: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("base_model.model.transformer.h.") and (
            key.endswith(".attn.qkv_proj.weight")
            or key.endswith(".attn.out_proj.weight")
        ):
            renamed[key.replace(".weight", ".base_layer.weight")] = value
        else:
            renamed[key] = value
    return renamed


def load_pretrained_progen_model(
    base_model_path: str,
    tokenizer_path: str,
    *,
    device_map: str | None = None,
    torch_dtype: torch.dtype | None = None,
    lora_r: int = 32,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    target_modules: Sequence[str] = ("qkv_proj", "out_proj"),
    lora_checkpoint: str | Path | None = None,
    inference_mode: bool = False,
) -> Tuple[object, torch.nn.Module]:
    """Load ProGen2-xlarge and attach the paper's LoRA architecture."""
    tokenizer_file, is_native_json = resolve_progen2_tokenizer_path(tokenizer_path)
    # Existing configs still point at ``assets/vocab.txt``.  When the base
    # checkpoint carries its own tokenizer.json, prefer that file automatically
    # so a legacy config cannot silently recreate the historical ID mismatch.
    checkpoint_tokenizer = Path(base_model_path) / "tokenizer.json"
    if checkpoint_tokenizer.is_file():
        tokenizer_file, is_native_json = checkpoint_tokenizer, True
    ProGenConfig, ProGenForCausalLM = _progen_classes()
    try:
        AutoConfig.register("progen", ProGenConfig)
    except ValueError:
        pass
    try:
        AutoModelForCausalLM.register(ProGenConfig, ProGenForCausalLM)
    except ValueError:
        pass

    tokenizer = (
        ProGen2Tokenizer(tokenizer_file)
        if is_native_json
        else BertTokenizer(vocab_file=str(tokenizer_file))
    )
    # Transformers accepts a mapping for a single-device placement.  Passing
    # ``"cuda:0"`` directly is version-dependent and is rejected by several
    # Accelerate releases, so normalize the public CLI form here.
    normalized_device_map = device_map
    if isinstance(device_map, str) and (
        device_map == "cpu" or device_map.startswith("cuda")
    ):
        normalized_device_map = {"": device_map}
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        local_files_only=Path(base_model_path).exists(),
        trust_remote_code=False,
        torch_dtype=torch_dtype,
        device_map=normalized_device_map,
    )
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=inference_mode,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=list(target_modules),
        ),
    )
    # The historical ProGen2 class predates the Transformers embedding API.
    # Install the standard accessors on both the base and PEFT wrapper so all
    # downstream code follows one interface, while retaining helper fallbacks
    # for third-party wrappers that reject instance attributes.
    base_model = _unwrap_progen_model(model)
    if not hasattr(base_model, "get_input_embeddings"):
        base_model.get_input_embeddings = lambda: base_model.transformer.wte
    if not hasattr(base_model, "get_output_embeddings"):
        base_model.get_output_embeddings = lambda: base_model.lm_head
    if lora_checkpoint:
        checkpoint = Path(lora_checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"LoRA checkpoint not found: {checkpoint}")
        state = torch.load(checkpoint, map_location="cpu")
        incompatible = model.load_state_dict(rename_legacy_lora_keys(state), strict=False)
        if not any("lora_" in key for key in state):
            raise ValueError(f"Checkpoint contains no LoRA weights: {checkpoint}")
        missing_lora = [
            key for key in incompatible.missing_keys if "lora_" in key
        ]
        if missing_lora:
            raise ValueError(
                f"Checkpoint is missing LoRA weights in {checkpoint}: "
                f"{missing_lora[:5]}"
            )
        unexpected_lora = [key for key in incompatible.unexpected_keys if "lora_" in key]
        if unexpected_lora:
            raise ValueError(f"Unexpected LoRA keys in {checkpoint}: {unexpected_lora[:5]}")
    model.eval()
    return tokenizer, model
