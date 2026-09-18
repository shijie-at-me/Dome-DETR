"""
Dome-DETR: DETR with Density-Oriented Feature-Query Manipulation for Efficient Tiny Object Detection
Copyright (c) 2025 The Dome-DETR Authors. All Rights Reserved.

The Dome-DETR model: ``DOME`` wires a backbone, an encoder (``HybridEncoder``, the D-FINE
baseline, or ``DomeHybridEncoder`` with DeFE and MWAS) and a decoder (``DFINETransformer``, fixed top-k
queries, or ``DomeTransformer`` with PAQI); ``DomeCriterion`` and ``HungarianMatcher`` train it
and ``DomePostProcessor`` turns its outputs into detections. Importing the package registers
all of them for the configs.
"""

from ...solver.matcher import HungarianMatcher
from .dfine_decoder import DFINETransformer
from .dome import DOME
from .dome_criterion import DomeCriterion
from .dome_decoder import DomeTransformer
from .dome_encoder import DomeHybridEncoder
from .hybrid_encoder import HybridEncoder
from .postprocessor import DomePostProcessor

__all__ = [
    "DFINETransformer",
    "DOME",
    "DomeCriterion",
    "DomeHybridEncoder",
    "DomePostProcessor",
    "DomeTransformer",
    "HungarianMatcher",
    "HybridEncoder",
]
