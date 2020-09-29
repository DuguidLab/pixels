"""
This library provides tools that can be used to manipulate and analyse neuropixels data
aligned to trial-organised behavioural data.
"""
from pixels.experiment import Experiment
from pixels.error import PixelsError

__all__ = [
    "Experiment",
    "PixelsError",
]
