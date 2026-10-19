"""PeptideDTO: test-time optimization for therapeutic peptide design."""

from importlib import import_module

__all__ = [
    "ESMMLPClassifier",
    "ESMMICRegressionModel",
    "PeptideDTOGenerator",
    "PeptideDTOOptimizer",
    "PeptideClassPrior",
    "TTGDOptimizer",
    "TTGDPeptideGenerator",
    "categorical_reverse_kl",
    "clean_sequence",
    "load_esm_mlp_reward_model",
    "load_mic_regression_model",
    "HuangLSTMRegressor",
    "HuangLSTMMICModel",
    "load_huang_lstm_mic_model",
    "require_finite",
]

__version__ = "0.1.0"


def __getattr__(name: str):
    if name in __all__:
        if name in {"HuangLSTMRegressor", "HuangLSTMMICModel", "load_huang_lstm_mic_model"}:
            return getattr(import_module(".mic_lstm", __name__), name)
        return getattr(import_module(".core", __name__), name)
    raise AttributeError(name)
