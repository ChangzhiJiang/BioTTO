"""Minimal LoRA support for Evo 2's custom Vortex projection modules."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn
from torch.nn import functional as F


DEFAULT_TARGET_SUFFIXES = (
    "projections",
    "out_filter_dense",
    "inner_mha_cls.Wqkv",
    "inner_mha_cls.out_proj",
)


@dataclass(frozen=True)
class Evo2LoraConfig:
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.05
    target_suffixes: tuple[str, ...] = DEFAULT_TARGET_SUFFIXES
    block_indices: tuple[int, ...] | None = None


class LoraProjection(nn.Module):
    """Add a trainable low-rank branch to any 2-D Vortex projection."""

    def __init__(
        self,
        base: nn.Module,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        weight = getattr(base, "weight", None)
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            raise TypeError("LoRA base module must expose a 2-D weight tensor")
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        output_features, input_features = weight.shape
        self.lora_a = nn.Parameter(
            torch.empty(rank, input_features, device=weight.device, dtype=torch.float32)
        )
        self.lora_b = nn.Parameter(
            torch.zeros(output_features, rank, device=weight.device, dtype=torch.float32)
        )
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def forward(self, inputs: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        base_output = self.base(inputs, *args, **kwargs)
        adapter = F.linear(
            F.linear(self.dropout(inputs.float()), self.lora_a), self.lora_b
        ) * self.scaling
        if isinstance(base_output, tuple):
            first, *rest = base_output
            return (first + adapter.to(first.dtype), *rest)
        return base_output + adapter.to(base_output.dtype)


def _parent_and_attribute(model: nn.Module, name: str) -> tuple[nn.Module, str]:
    parent_name, _, attribute = name.rpartition(".")
    return (model.get_submodule(parent_name) if parent_name else model), attribute


def inject_lora(model: nn.Module, config: Evo2LoraConfig) -> list[str]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    def block_is_selected(name: str) -> bool:
        if config.block_indices is None:
            return True
        parts = name.split(".")
        return (
            len(parts) >= 2
            and parts[0] == "blocks"
            and parts[1].isdigit()
            and int(parts[1]) in config.block_indices
        )

    targets = [
        name
        for name, module in model.named_modules()
        if not isinstance(module, LoraProjection)
        and block_is_selected(name)
        and any(name.endswith(suffix) for suffix in config.target_suffixes)
        and isinstance(getattr(module, "weight", None), torch.Tensor)
        and module.weight.ndim == 2
    ]
    if not targets:
        raise RuntimeError("no Evo 2 projection modules matched the LoRA target suffixes")
    for name in targets:
        parent, attribute = _parent_and_attribute(model, name)
        setattr(
            parent,
            attribute,
            LoraProjection(
                getattr(parent, attribute),
                rank=config.rank,
                alpha=config.alpha,
                dropout=config.dropout,
            ),
        )
    return targets


def iter_lora_modules(model: nn.Module) -> Iterable[tuple[str, LoraProjection]]:
    for name, module in model.named_modules():
        if isinstance(module, LoraProjection):
            yield name, module


def lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [
        parameter
        for _, module in iter_lora_modules(model)
        for parameter in (module.lora_a, module.lora_b)
    ]


def materialize_autograd_tensors(model: nn.Module) -> None:
    """Replace inference-mode checkpoint tensors before LoRA backpropagation."""
    with torch.inference_mode(False):
        for module in model.modules():
            for name, parameter in list(module._parameters.items()):
                if parameter is None or not parameter.is_inference():
                    continue
                module._parameters[name] = nn.Parameter(
                    parameter.detach().clone(), requires_grad=False
                )
            for name, buffer in list(module._buffers.items()):
                if buffer is not None and buffer.is_inference():
                    module._buffers[name] = buffer.detach().clone()
            for name, value in list(vars(module).items()):
                if isinstance(value, torch.Tensor) and value.is_inference():
                    setattr(module, name, value.detach().clone())


def set_lora_training(model: nn.Module, training: bool) -> None:
    """Toggle adapter dropout without enabling stochastic frozen-model layers."""
    model.eval()
    for _, module in iter_lora_modules(model):
        module.train(training)


def trainable_parameter_summary(model: nn.Module) -> dict[str, int | float]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_fraction": trainable / total,
        "lora_modules": sum(1 for _ in iter_lora_modules(model)),
    }


def adapter_state(model: nn.Module) -> dict[str, torch.Tensor]:
    state = {}
    for name, module in iter_lora_modules(model):
        state[f"{name}.lora_a"] = module.lora_a.detach().cpu()
        state[f"{name}.lora_b"] = module.lora_b.detach().cpu()
    return state


def restore_adapter_state(
    model: nn.Module, state: dict[str, torch.Tensor]
) -> None:
    """Load tensors into LoRA modules that have already been injected."""
    modules = dict(iter_lora_modules(model))
    expected = {
        f"{name}.{parameter_name}"
        for name in modules
        for parameter_name in ("lora_a", "lora_b")
    }
    if set(state) != expected:
        missing = sorted(expected - set(state))
        unexpected = sorted(set(state) - expected)
        raise ValueError(
            f"adapter keys do not match injected modules; "
            f"missing={missing[:3]} unexpected={unexpected[:3]}"
        )
    for key, value in state.items():
        module_name, parameter_name = key.rsplit(".", 1)
        parameter = getattr(modules[module_name], parameter_name)
        if parameter.shape != value.shape:
            raise ValueError(f"adapter shape mismatch for {key}")
        parameter.data.copy_(value.to(parameter.device, parameter.dtype))


def save_lora_adapter(
    model: nn.Module,
    path: Path,
    config: Evo2LoraConfig,
    metadata: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "evo2_vortex_lora_v1",
            "config": asdict(config),
            "metadata": metadata or {},
            "state_dict": adapter_state(model),
        },
        path,
    )


def load_lora_adapter(
    model: nn.Module, path: Path
) -> tuple[Evo2LoraConfig, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    raw_config = payload["config"]
    raw_config["target_suffixes"] = tuple(raw_config["target_suffixes"])
    if raw_config.get("block_indices") is not None:
        raw_config["block_indices"] = tuple(raw_config["block_indices"])
    config = Evo2LoraConfig(**raw_config)
    inject_lora(model, config)
    restore_adapter_state(model, payload["state_dict"])
    return config, payload.get("metadata", {})
