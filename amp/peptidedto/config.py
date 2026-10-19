"""Configuration helpers for the paper-faithful 5--50 benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import math

import yaml


REQUIRED_SECTIONS = {"target", "sft", "tgl", "reward", "sampling", "benchmark"}


def load_target_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open() as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")
    missing = REQUIRED_SECTIONS - set(config)
    if missing:
        raise ValueError(f"Missing configuration sections: {sorted(missing)}")
    if config["target"]["name"] not in {"amp", "afp", "avp"}:
        raise ValueError("target.name must be amp, afp, or avp")
    if [config["benchmark"]["min_length"], config["benchmark"]["max_length"]] != [5, 50]:
        raise ValueError("Paper reproduction requires the 5--50 residue benchmark")
    benchmark = config["benchmark"]
    if int(benchmark["n_sequences"]) <= 0:
        raise ValueError("benchmark.n_sequences must be positive")
    if int(benchmark["max_attempt_factor"]) < 1:
        raise ValueError("benchmark.max_attempt_factor must be positive")
    if int(benchmark["generation_batch_size"]) < 1:
        raise ValueError("benchmark.generation_batch_size must be positive")
    tgl = config["tgl"]
    if int(tgl["rounds"]) < 1 or int(tgl["iterations"]) < 0:
        raise ValueError("tgl.rounds must be positive and tgl.iterations non-negative")
    for key in ("learning_rate",):
        if not math.isfinite(float(tgl[key])) or float(tgl[key]) <= 0:
            raise ValueError(f"tgl.{key} must be finite and positive")
    for key in ("reward_coeff", "nll_coeff", "kl_coeff"):
        if not math.isfinite(float(tgl[key])) or float(tgl[key]) < 0:
            raise ValueError(f"tgl.{key} must be finite and non-negative")
    sampling = config["sampling"]
    if float(sampling["temperature"]) <= 0 or not math.isfinite(float(sampling["temperature"])):
        raise ValueError("sampling.temperature must be finite and positive")
    if not 0 < float(sampling["top_p"]) <= 1:
        raise ValueError("sampling.top_p must lie in (0, 1]")
    if int(sampling["top_k"]) < 0:
        raise ValueError("sampling.top_k must be non-negative")
    reward = config["reward"]
    for key in ("activity_coeff", "stability_coeff", "similarity_coeff", "length_coeff"):
        if not math.isfinite(float(reward[key])) or float(reward[key]) < 0:
            raise ValueError(f"reward.{key} must be finite and non-negative")
    for key in ("stability_scale", "length_scale"):
        if not math.isfinite(float(reward[key])) or float(reward[key]) <= 0:
            raise ValueError(f"reward.{key} must be finite and positive")
    if not 5 <= float(reward["target_length"]) <= 50:
        raise ValueError("reward.target_length must lie within the benchmark length range")
    if int(reward["max_references"]) < 1:
        raise ValueError("reward.max_references must be positive")
    mic = config.get("mic")
    if mic and mic.get("enabled"):
        if float(mic.get("reward_coeff", 0)) < 0:
            raise ValueError("mic.reward_coeff must be non-negative")
        if float(mic.get("reward_scale", 1)) <= 0:
            raise ValueError("mic.reward_scale must be positive")
    return config
