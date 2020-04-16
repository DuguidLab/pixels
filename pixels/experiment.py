"""
This module provides an Experiment class which serves as the main interface to process
data and run subsequent analyses.
"""


import os
from pathlib import Path

from pixels import ioutils


class Experiment:
    def __init__(self, mouse_ids, behaviour, data_dir, meta_dir):
        """
        This represents an experiment and can be used to process or analyse data for a
        group of mice.

        Parameters
        ----------
        mouse_ids : list of strs
            List of IDs of the mice to be included in this experiment.

        behaviour : class
            Class definition subclassing from behaviours.Behaviour.

        data_dir : str
            Path to the top-level folder containing data for these mice.

        meta_dir : str
            Path to the folder containing training metadata JSON files.

        """
        if not isinstance(mouse_ids, (list, tuple, set)):
            mouse_ids = [mouse_ids]

        self.behaviour = behaviour
        self.mouse_ids = mouse_ids
        self.data_dir = Path(data_dir)
        self.meta_dir = Path(meta_dir)

        self.sessions = []
        for session in ioutils.get_sessions(mouse_ids, self.data_dir, self.meta_dir):
            self.sessions.append(
                behaviour(
                    session['sessions'],
                    metadata=session['metadata'],
                    data_dir=session['data_dir'],
                )
            )

    def extract_spikes(self, resample=True):
        """
        Extract the spikes from raw spike data for all sessions.
        """
        for session in self.sessions:
            session.extract_spikes(resample=resample)

    def process_lfp(self, resample=True):
        """
        Process the LFP data from the raw neural recording data for all sessions.
        """
        for session in self.sessions:
            session.process_lfp(resample=resample)

    def process_behaviour(self):
        """
        Process behavioural data from raw tdms files for all sessions.
        """
        for session in self.sessions:
            session.process_behaviour()

    def process_motion_tracking(self):
        """
        Process motion tracking data either from raw camera data, or from
        previously-generated deeplabcut coordinate data, for all sessions.
        """
        for session in self.sessions:
            session.process_motion_tracking()
