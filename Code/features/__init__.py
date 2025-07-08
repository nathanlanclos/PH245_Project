"""
features/__init__.py

Initializes the feature extraction submodules:
- BLOSUM embeddings
- ESM model embeddings
- Structure-based features
- Classic sequence features
"""

from .blosum_features import BlosumFeatureExtractor
from .esm_features import ESMFeatureExtractor
from .structure_features import StructureFeatureExtractor
from .sequence_features import SequenceFeatureExtractor
from .generate_structures import StructureGenerator
from .esm3_features import ESM3FeatureExtractor

__all__ = [
    "BlosumFeatureExtractor",
    "ESMFeatureExtractor",
    "StructureFeatureExtractor",
    "SequenceFeatureExtractor",
    "StructureGenerator"
    "ESM3FeatureExtractor"
]
