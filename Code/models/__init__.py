"""
features/__init__.py

Initializes the feature extraction submodules:
- BLOSUM embeddings
- ESM model embeddings
- Structure-based features
- Classic sequence features
"""

from .train_model import SolubilityPredictor
__all__ = [
    "SolubilityPredictor",
]
