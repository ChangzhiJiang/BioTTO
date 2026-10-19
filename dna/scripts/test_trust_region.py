#!/usr/bin/env python3
"""Unit checks for DTO trust-region and edit-budget helpers."""

import torch

from src.dto import Evo2Dto


def main() -> None:
    dto = object.__new__(Evo2Dto)
    dto.device = torch.device("cpu")
    logits = torch.tensor(
        [[[0.0, 4.0, 0.0, 0.0], [0.0, 0.0, 3.0, 0.0], [0.0, 0.0, 0.0, 2.0]]],
        requires_grad=True,
    )
    assert dto.decode(logits, reference_sequence="AAA", max_edits=1) == "CAA"
    assert dto.decode(logits, reference_sequence="AAA", max_edits=0) == "AAA"

    reference = Evo2Dto._soft_distribution(logits.detach(), temperature=0.7)
    distribution = Evo2Dto._soft_distribution(logits, temperature=0.7)
    indices = dto._reference_indices("AAA")
    loss = Evo2Dto._trust_region_kl(distribution, reference)
    loss = loss + Evo2Dto._expected_edit_fraction(distribution, indices)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    print("trust/edit unit checks passed")


if __name__ == "__main__":
    main()
