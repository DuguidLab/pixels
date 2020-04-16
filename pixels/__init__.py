"""
This library provides tools that can be used to manipulate and analyse neuropixels data
aligned to trial-organised behavioural data.
"""


import os
import json

from pixels.experiment import Experiment


def select_mice(**kwargs):
    """
    Stuff to select mice.
    """
    # do something with kwargs to select animals

    return mouse_ids


def get_sessions(mouse_ids, meta_dir):
    if not instance(mouse_ids, (List, Tuple, Set)):
        mouse_ids = List(mouse_ids)

    sessions = {}

    for mouse_id in mouse_ids:
        mouse_file = os.path.join(meta_dir, mouse_id + '.json')
        with open(mouse_file, 'r') as fd:
            mouse_data = json.load(fd)
