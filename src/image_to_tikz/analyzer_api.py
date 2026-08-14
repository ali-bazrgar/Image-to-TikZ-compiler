from __future__ import annotations

from .analyzer_model_free import ModelFreeImageAnalyzer

# Stable public analyzer entry point. The runtime is strictly model-free.
ImageAnalyzer = ModelFreeImageAnalyzer

__all__ = ["ImageAnalyzer"]
