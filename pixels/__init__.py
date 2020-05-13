"""
This library provides tools that can be used to manipulate and analyse neuropixels data
aligned to trial-organised behavioural data.
"""


import os
import json

from pixels.experiment import Experiment


class PixelsError(Exception):
    """
    Error type used for user errors.
    """
    pass
