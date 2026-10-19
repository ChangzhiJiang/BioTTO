"""Differentiable activity predictors for the Gosai three-cell benchmark."""

from __future__ import annotations

import torch
from torch import nn


class ShortEnformerActivity(nn.Module):
    """Differentiable three-cell activity predictor for 200 bp sequences."""

    def __init__(
        self,
        dim: int = 192,
        depth: int = 4,
        heads: int = 4,
        num_downsamples: int = 3,
        target_length: int = 1,
    ) -> None:
        super().__init__()
        from enformer_pytorch import Enformer

        if dim % heads or (dim // heads) % 6:
            raise ValueError("dim / heads must be an integer divisible by 6")
        self.config = {
            "dim": dim,
            "depth": depth,
            "heads": heads,
            "num_downsamples": num_downsamples,
            "target_length": target_length,
        }
        enformer_config = {
            **self.config,
            "output_heads": {"activity": 3},
            "attn_dim_key": max(dim // heads // 2, 8),
            "dropout_rate": 0.1,
            "attn_dropout": 0.05,
            "pos_dropout": 0.01,
            "dim_divisible_by": max(32, dim // 3),
        }
        self.backbone = Enformer.from_hparams(**enformer_config)
        self.activity_head = nn.Linear(dim * 2, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != 4:
            raise ValueError("x must have shape [batch, length, 4]")
        embeddings = self.backbone(x, return_only_embeddings=True)
        return self.activity_head(embeddings.mean(dim=1))


class GreluEnformerActivity(nn.Module):
    """Frozen full-size gReLU Enformer with gradients to soft DNA inputs."""

    def __init__(self, lightning_model: nn.Module) -> None:
        super().__init__()
        self.lightning_model = lightning_model
        self.lightning_model.eval()
        for parameter in self.lightning_model.parameters():
            parameter.requires_grad_(False)

    @classmethod
    def from_checkpoint(
        cls, path: str, device: torch.device | str = "cpu"
    ) -> "GreluEnformerActivity":
        from grelu.lightning import LightningModel

        # The checkpoint records EnformerPretrainedModel, whose constructor
        # downloads generic weights before restoring the checkpoint. The plain
        # EnformerModel has the identical final architecture and restores all
        # checkpoint tensors without a network dependency.
        model = LightningModel.load_from_checkpoint(
            path,
            model_params={"model_type": "EnformerModel", "n_tasks": 3},
            map_location=device,
            weights_only=False,
        )
        model.to(device)
        return cls(model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != 4:
            raise ValueError("x must have shape [batch, length, 4]")
        output = self.lightning_model(x.transpose(1, 2).contiguous())
        if output.ndim == 3 and output.shape[-1] == 1:
            output = output[..., 0]
        if output.ndim != 2 or output.shape[-1] != 3:
            raise ValueError(f"expected activity shape [batch, 3], got {output.shape}")
        return output


def load_activity_checkpoint(
    path: str, device: torch.device | str = "cpu"
) -> nn.Module:
    if path.lower().endswith(".ckpt"):
        return GreluEnformerActivity.from_checkpoint(path, device=device)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = ShortEnformerActivity(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model
