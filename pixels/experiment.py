"""
This module provides an Experiment class which serves as the main interface to process
data and run subsequent analyses.
"""


import os
from pathlib import Path

import pandas as pd

from pixels import ioutils
from pixels.error import PixelsError


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
            Class definition subclassing from pixels.behaviours.Behaviour.

        data_dir : str
            Path to the top-level folder containing data for these mice. This folder
            should contain these folders: raw, interim, processed

        meta_dir : str
            Path to the folder containing training metadata JSON files.

        """
        if not isinstance(mouse_ids, (list, tuple, set)):
            mouse_ids = [mouse_ids]

        self.behaviour = behaviour
        self.mouse_ids = mouse_ids

        self.data_dir = Path(data_dir).expanduser()
        self.meta_dir = Path(meta_dir).expanduser()
        if not self.data_dir.exists():
            raise PixelsError(f"Directory not found: {data_dir}")
        if not self.meta_dir.exists():
            raise PixelsError(f"Directory not found: {meta_dir}")

        self.raw = self.data_dir / 'raw'
        self.processed = self.data_dir / 'processed'
        self.interim = self.data_dir / 'interim'

        self.sessions = []
        for session in ioutils.get_sessions(mouse_ids, self.data_dir, self.meta_dir):
            self.sessions.append(
                behaviour(
                    session['sessions'],
                    metadata=session['metadata'],
                    data_dir=session['data_dir'],
                )
            )

    def __getitem__(self, index):
        """
        Allow indexing directly of sessions with myexp[X].
        """
        return self.sessions[index]

    def __len__(self):
        """
        Length of experiment is the number of sessions.
        """
        return len(self.sessions)

    def __repr__(self):
        rep = "Experiment with sessions:"
        for session in self.sessions:
            rep += "\n\t" + session.name
        return rep

    def process_spikes(self):
        """
        Process the spike data from the raw neural recording data for all sessions.
        """
        for session in self.sessions:
            session.process_spikes()

    def extract_spikes(self):
        """
        Extract the spikes from raw spike data for all sessions.
        """
        for session in self.sessions:
            session.extract_spikes()

    def process_lfp(self):
        """
        Process the LFP data from the raw neural recording data for all sessions.
        """
        for session in self.sessions:
            session.process_lfp()

    def process_behaviour(self):
        """
        Process behavioural data from raw tdms files for all sessions.
        """
        for session in self.sessions:
            session.process_behaviour()

    def extract_videos(self):
        """
        Extract videos from TDMS in the raw folder to avi files in the interim folder.
        """
        for session in self.sessions:
            session.extract_videos()

    def process_motion_tracking(self, config, create_labelled_video=False):
        """
        Process motion tracking data either from raw camera data, or from
        previously-generated deeplabcut coordinate data, for all sessions.
        """
        for session in self.sessions:
            session.process_motion_tracking(config, create_labelled_video=False)

    def align_trials(self, label, event, data, raw=False, duration=1):
        """
        Get trials aligned to an event. Check behaviours.base.Behaviour.align_trials for
        usage information.
        """
        trials = []
        for session in self.sessions:
            trials.append(session.align_trials(label, event, data, raw, duration))

        df = pd.concat(
            trials, axis=1, copy=False,
            keys=range(len(trials)),
            names=["session", "unit", "trial"]
        )
        return df
