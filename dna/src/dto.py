"""Inference-time four-base logit optimization on top of Evo 2.

The adapter intentionally uses Evo 2's public model object without modifying the
upstream package. The model is frozen; only the per-sequence logits are optimized.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Sequence

import torch
import torch.nn.functional as F


BASES = "ACGT"


def validate_sequence(sequence: str) -> str:
    sequence = sequence.upper().replace(" ", "").replace("\n", "")
    invalid = sorted(set(sequence) - set(BASES))
    if invalid:
        raise ValueError(f"DNA contains unsupported bases: {invalid}")
    if not sequence:
        raise ValueError("DNA sequence must not be empty")
    return sequence


@dataclass
class DtoConfig:
    steps: int = 5
    learning_rate: float = 0.01
    temperature: float = 0.7
    reward_weight: float = 1.0
    prior_weight: float = 0.20
    gradient_clip_norm: float = 5.0
    initial_sequence_margin: float = 0.10
    trust_region_weight: float = 0.0
    edit_penalty_weight: float = 0.0
    max_edits: int | None = None
    # Optional, deterministic per-trajectory perturbation.  The default keeps
    # historical searches bit-for-bit compatible; scalability studies can use
    # a seeded perturbation to make repeated starts genuine, auditable paths.
    initial_logit_noise_std: float = 0.0
    initial_logit_noise_seed: int | None = None
    audit_hashes: bool = False


@dataclass
class DtoResult:
    initial_sequence: str
    optimized_sequence: str
    initial_reward: float
    final_reward: float
    initial_prior_nll: float
    final_prior_nll: float
    history: list[Dict[str, float]]
    final_trust_kl: float
    final_expected_edit_fraction: float
    hamming_edits: int
    initial_logits_sha256: str | None = None
    final_logits_sha256: str | None = None
    initial_noise_sha256: str | None = None


def _tensor_sha256(value: torch.Tensor) -> str:
    """Hash a canonical CPU float32 view for replay auditing."""
    data = value.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


RewardFn = Callable[[torch.Tensor], torch.Tensor]


class Evo2Dto:
    """Frozen Evo 2 prior plus differentiable test-time sequence optimization.

    The reward callback receives a straight-through tensor with shape
    ``[batch, length, 4]``. Its forward values are hard one-hot bases while its
    backward path uses the softmax relaxation, matching the DTO idea.
    """

    def __init__(
        self,
        model_name: str = "evo2_7b",
        lora_adapter: str | Path | None = None,
    ) -> None:
        from evo2 import Evo2

        # The upstream Evo2 API accepts ``local_path`` for an offline
        # checkpoint.  Keep the historical model-name path as the default, but
        # allow the reproducibility package to point at its bundled weight
        # without relying on a HuggingFace cache.
        local_path = os.environ.get("BIOTTO_EVO2_LOCAL_PATH")
        self.evo2 = (
            Evo2(model_name, local_path=local_path)
            if local_path
            else Evo2(model_name)
        )
        self.model = self.evo2.model
        self._materialize_autograd_tensors(self.model)
        self.lora_adapter_path = str(lora_adapter) if lora_adapter is not None else None
        self.lora_config = None
        self.lora_metadata: dict[str, object] = {}
        if lora_adapter is not None:
            from .lora import load_lora_adapter

            self.lora_config, self.lora_metadata = load_lora_adapter(
                self.model, Path(lora_adapter)
            )
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        self.device = self.model.block_idx_to_device[0]
        self.base_ids = torch.tensor(
            [ord(base) for base in BASES], dtype=torch.long, device=self.device
        )

    @staticmethod
    def _materialize_autograd_tensors(model: torch.nn.Module) -> None:
        """Convert Vortex inference tensors into normal frozen tensors.

        Evo 2's checkpoint loader intentionally uses ``torch.inference_mode`` for
        fast generation. Autograd cannot save those tensors during a backward
        pass, even when all model parameters are frozen. Cloning parameters and
        buffers once keeps the weights unchanged while enabling input gradients.
        """
        with torch.inference_mode(False):
            for module in model.modules():
                for name, parameter in list(module._parameters.items()):
                    if parameter is None:
                        continue
                    if not parameter.is_inference():
                        continue
                    materialized = parameter.detach().clone()
                    module._parameters[name] = torch.nn.Parameter(
                        materialized, requires_grad=False
                    )
                for name, buffer in list(module._buffers.items()):
                    if buffer is None:
                        continue
                    if not buffer.is_inference():
                        continue
                    module._buffers[name] = buffer.detach().clone()
                # Vortex keeps rotary cos/sin caches as plain tensor attributes
                # rather than registered buffers. Cached generation creates them
                # in inference mode, so clone those attributes as well.
                for name, value in list(vars(module).items()):
                    if isinstance(value, torch.Tensor) and value.is_inference():
                        setattr(module, name, value.detach().clone())

    def _token_ids(self, sequence: str) -> torch.Tensor:
        sequence = validate_sequence(sequence)
        return torch.tensor(
            [ord(base) for base in sequence], dtype=torch.long, device=self.device
        ).unsqueeze(0)

    def disable_flash_attention(self) -> None:
        """Use PyTorch SDPA for short-sequence Vortex generation.

        Some Torch/Vortex combinations fail in the FlashAttention KV-cache
        operator around 200 bp. Replacing only the attention kernels preserves
        Evo 2 weights and rotary/KV-cache semantics.
        """
        from vortex.model.attention import CrossAttention, SelfAttention

        replaced = 0
        for module in self.model.modules():
            mha = getattr(module, "inner_mha_cls", None)
            if mha is None or not hasattr(mha, "inner_attn"):
                continue
            self_attention = mha.inner_attn
            cross_attention = mha.inner_cross_attn
            mha.use_flash_attn = False
            mha.inner_attn = SelfAttention(
                causal=self_attention.causal,
                softmax_scale=self_attention.softmax_scale,
                attention_dropout=self_attention.drop.p,
            )
            mha.inner_cross_attn = CrossAttention(
                causal=cross_attention.causal,
                softmax_scale=cross_attention.softmax_scale,
                attention_dropout=cross_attention.drop.p,
            )
            replaced += 1
        if replaced == 0:
            raise RuntimeError("no Vortex attention blocks were found")

    def _hard_softmax(self, logits: torch.Tensor, temperature: float) -> torch.Tensor:
        soft = F.softmax(logits / temperature, dim=-1)
        hard = F.one_hot(soft.argmax(dim=-1), num_classes=4).to(soft.dtype)
        return hard + soft - soft.detach()

    @staticmethod
    def _soft_distribution(logits: torch.Tensor, temperature: float) -> torch.Tensor:
        return F.softmax(logits / temperature, dim=-1)

    def _reference_indices(self, sequence: str) -> torch.Tensor:
        return torch.tensor(
            [BASES.index(base) for base in validate_sequence(sequence)],
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(0)

    @staticmethod
    def _trust_region_kl(
        distribution: torch.Tensor, reference_distribution: torch.Tensor
    ) -> torch.Tensor:
        """Mean KL(q || q_init) over candidate positions."""
        eps = torch.finfo(distribution.dtype).eps
        distribution = distribution.clamp_min(eps)
        reference_distribution = reference_distribution.clamp_min(eps)
        return (
            distribution * (distribution.log() - reference_distribution.log())
        ).sum(dim=-1).mean()

    @staticmethod
    def _expected_edit_fraction(
        distribution: torch.Tensor, reference_indices: torch.Tensor
    ) -> torch.Tensor:
        keep_probability = distribution.gather(-1, reference_indices.unsqueeze(-1))
        return (1.0 - keep_probability.squeeze(-1)).mean()

    def logits_for_sequence(
        self, sequence: str, initial_sequence_margin: float = 0.10
    ) -> torch.Tensor:
        """Initialize four-base logits from Evo 2's teacher-forced predictions."""
        sequence = validate_sequence(sequence)
        token_ids = self._token_ids(sequence)
        with torch.no_grad():
            outputs, _ = self.model(token_ids)
            base_logits = outputs[..., self.base_ids]

        # The logit at position i predicts position i+1. The first base has no
        # left context, so initialize it with a neutral prior.
        logits = torch.zeros(
            1, token_ids.shape[1], 4, device=self.device, dtype=base_logits.dtype
        )
        if token_ids.shape[1] > 1:
            logits[:, 1:] = base_logits[:, :-1]

        # Preserve the actual rollout as the initial hard sequence. Evo 2's
        # logits describe a distribution and the sampled base is not always its
        # argmax, especially at nonzero generation temperature.
        chosen = torch.tensor(
            [BASES.index(base) for base in sequence],
            device=self.device,
            dtype=torch.long,
        ).view(1, -1, 1)
        chosen_values = logits.gather(-1, chosen)
        boost = (
            logits.max(dim=-1, keepdim=True).values
            - chosen_values
            + initial_sequence_margin
        ).clamp_min(0)
        logits.scatter_add_(-1, chosen, boost)
        return logits.detach().float()

    def soft_forward(self, base_distribution: torch.Tensor) -> torch.Tensor:
        """Run Evo 2 from differentiable four-base embeddings.

        Evo 2's public forward accepts integer token IDs. The underlying Vortex
        model also accepts embedded activations, so we form a weighted sum of the
        four nucleotide embedding rows and invoke its stateless forward path.
        """
        if base_distribution.ndim != 3 or base_distribution.shape[-1] != 4:
            raise ValueError("base_distribution must have shape [batch, length, 4]")

        embedding_weight = self.model.embedding_layer.weight[self.base_ids]
        embeddings = torch.einsum(
            "blc,ch->blh",
            base_distribution.to(embedding_weight.dtype),
            embedding_weight,
        )
        hidden, _ = self.model.stateless_forward(embeddings)
        hidden = hidden.to(self.device)
        hidden = self.model.norm(hidden)
        return self.model.unembed(hidden)

    def prior_nll(
        self, output_logits: torch.Tensor, base_distribution: torch.Tensor
    ) -> torch.Tensor:
        """Causal Evo 2 NLL over the soft next-base targets."""
        log_probs = F.log_softmax(output_logits[..., self.base_ids], dim=-1)
        if base_distribution.shape[1] <= 1:
            return output_logits.new_zeros(())
        next_targets = base_distribution[:, 1:]
        next_log_probs = log_probs[:, :-1]
        return -(next_targets * next_log_probs).sum(dim=-1).mean()

    def decode(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        reference_sequence: str | None = None,
        max_edits: int | None = None,
    ) -> str:
        indices = (logits / temperature).argmax(dim=-1)
        if max_edits is not None:
            if max_edits < 0:
                raise ValueError("max_edits must be non-negative")
            if reference_sequence is None:
                raise ValueError("reference_sequence is required when max_edits is set")
            reference_indices = self._reference_indices(reference_sequence)
            if reference_indices.shape != indices.shape:
                raise ValueError("reference sequence length does not match logits")
            changed = indices != reference_indices
            if int(changed.sum()) > max_edits:
                candidate_logits = logits.gather(-1, indices.unsqueeze(-1)).squeeze(-1)
                reference_logits = logits.gather(
                    -1, reference_indices.unsqueeze(-1)
                ).squeeze(-1)
                gains = (candidate_logits - reference_logits).masked_fill(
                    ~changed, float("-inf")
                )
                keep = torch.zeros_like(changed)
                keep_indices = gains.flatten().topk(max_edits).indices
                keep.flatten()[keep_indices] = True
                indices = torch.where(keep, indices, reference_indices)
        indices = indices[0].tolist()
        return "".join(BASES[index] for index in indices)

    def one_hot_for_sequence(self, sequence: str) -> torch.Tensor:
        sequence = validate_sequence(sequence)
        indices = torch.tensor(
            [BASES.index(base) for base in sequence],
            dtype=torch.long,
            device=self.device,
        )
        return F.one_hot(indices, num_classes=4).to(torch.float32).unsqueeze(0)

    def score_sequence(self, sequence: str) -> float:
        """Return mean causal log-probability over the four DNA bases."""
        token_ids = self._token_ids(sequence)
        with torch.no_grad():
            outputs, _ = self.model(token_ids)
            log_probs = F.log_softmax(outputs[..., self.base_ids], dim=-1)
            # ASCII IDs are not contiguous, so map target IDs explicitly.
            target_indices = torch.zeros_like(token_ids[:, 1:])
            for index, token_id in enumerate(self.base_ids):
                target_indices[token_ids[:, 1:] == token_id] = index
            return float(log_probs[:, :-1].gather(-1, target_indices.unsqueeze(-1)).mean())

    def optimize(
        self,
        sequence: str,
        reward_fn: RewardFn,
        config: Optional[DtoConfig] = None,
    ) -> DtoResult:
        config = config or DtoConfig()
        sequence = validate_sequence(sequence)
        logits = self.logits_for_sequence(
            sequence, config.initial_sequence_margin
        )
        initial_logits_sha256 = None
        final_logits_sha256 = None
        initial_noise_sha256 = None
        if config.initial_logit_noise_std < 0:
            raise ValueError("initial_logit_noise_std must be non-negative")
        if config.initial_logit_noise_std:
            if config.initial_logit_noise_seed is None:
                raise ValueError(
                    "initial_logit_noise_seed is required when logit noise is enabled"
                )
            # Draw on CPU so the perturbation is independent of CUDA's
            # nondeterministic RNG implementation, then transfer it to the
            # model device.  Re-boost the observed bases so the trajectory
            # still starts from the locked sequence.
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(config.initial_logit_noise_seed) % (2**63 - 1))
            noise = torch.randn(
                logits.shape, generator=generator, dtype=logits.dtype
            ).to(logits.device)
            if config.audit_hashes:
                initial_noise_sha256 = _tensor_sha256(noise)
            logits = logits + float(config.initial_logit_noise_std) * noise
            chosen = self._reference_indices(sequence).unsqueeze(-1)
            chosen_values = logits.gather(-1, chosen)
            non_chosen = torch.zeros_like(logits, dtype=torch.bool)
            non_chosen.scatter_(-1, chosen, True)
            max_other = logits.masked_fill(non_chosen, float("-inf")).max(
                dim=-1, keepdim=True
            ).values
            boost = (
                max_other - chosen_values + config.initial_sequence_margin
            ).clamp_min(0)
            logits = logits.scatter_add(-1, chosen, boost)
        if config.audit_hashes:
            initial_logits_sha256 = _tensor_sha256(logits)
        logits = logits.detach().requires_grad_(True)
        optimizer = torch.optim.Adam([logits], lr=config.learning_rate)
        reference_logits = logits.detach().clone()
        reference_distribution = self._soft_distribution(
            reference_logits, config.temperature
        )
        reference_indices = self._reference_indices(sequence)

        with torch.no_grad():
            initial_q = self._hard_softmax(logits, config.temperature)
            initial_output = self.soft_forward(initial_q)
            initial_prior = self.prior_nll(initial_output, initial_q)
            initial_reward = reward_fn(initial_q).detach()

        history: list[Dict[str, float]] = []
        for step in range(config.steps):
            optimizer.zero_grad(set_to_none=True)
            soft_q = self._soft_distribution(logits, config.temperature)
            q = self._hard_softmax(logits, config.temperature)
            output_logits = self.soft_forward(q)
            prior = self.prior_nll(output_logits, q)
            reward = reward_fn(q)
            trust_kl = self._trust_region_kl(soft_q, reference_distribution)
            expected_edit_fraction = self._expected_edit_fraction(
                soft_q, reference_indices
            )
            loss = (
                config.prior_weight * prior
                - config.reward_weight * reward
                + config.trust_region_weight * trust_kl
                + config.edit_penalty_weight * expected_edit_fraction
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_([logits], config.gradient_clip_norm)
            optimizer.step()
            history.append(
                {
                    "step": float(step),
                    "loss": float(loss.detach()),
                    "reward": float(reward.detach()),
                    "prior_nll": float(prior.detach()),
                    "trust_kl": float(trust_kl.detach()),
                    "expected_edit_fraction": float(expected_edit_fraction.detach()),
                }
            )

        with torch.no_grad():
            if config.audit_hashes:
                final_logits_sha256 = _tensor_sha256(logits)
            final_soft_q = self._soft_distribution(logits, config.temperature)
            final_trust_kl = self._trust_region_kl(
                final_soft_q, reference_distribution
            )
            final_expected_edit_fraction = self._expected_edit_fraction(
                final_soft_q, reference_indices
            )
            optimized_sequence = self.decode(
                logits,
                config.temperature,
                reference_sequence=sequence,
                max_edits=config.max_edits,
            )
            final_q = self.one_hot_for_sequence(optimized_sequence)
            final_output = self.soft_forward(final_q)
            final_prior = self.prior_nll(final_output, final_q)
            final_reward = reward_fn(final_q).detach()

        return DtoResult(
            initial_sequence=sequence,
            optimized_sequence=optimized_sequence,
            initial_reward=float(initial_reward),
            final_reward=float(final_reward),
            initial_prior_nll=float(initial_prior),
            final_prior_nll=float(final_prior),
            history=history,
            final_trust_kl=float(final_trust_kl),
            final_expected_edit_fraction=float(final_expected_edit_fraction),
            hamming_edits=sum(
                before != after
                for before, after in zip(sequence, optimized_sequence)
            ),
            initial_logits_sha256=initial_logits_sha256,
            final_logits_sha256=final_logits_sha256,
            initial_noise_sha256=initial_noise_sha256,
        )


class MotifReward:
    """Small differentiable reward used only for pipeline smoke tests."""

    def __init__(self, motifs: Sequence[str]) -> None:
        self.motifs = [validate_sequence(motif) for motif in motifs]

    def __call__(self, q: torch.Tensor) -> torch.Tensor:
        total = q.new_zeros(())
        for motif in self.motifs:
            target = torch.tensor(
                [BASES.index(base) for base in motif], device=q.device, dtype=torch.long
            )
            motif_q = q[:, : len(motif), :].gather(
                -1, target.view(1, -1, 1).expand(q.shape[0], -1, 1)
            )
            total = total + motif_q.mean()
        return total / max(len(self.motifs), 1)
