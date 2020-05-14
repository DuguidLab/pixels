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

        self._trial_duration = 6  # number of seconds in which to extract trials

    @property
    def trial_duration(self):
        return self._trial_duration

    @trial_duration.setter
    def trial_duration(self, secs):
        """
        By default, when we get data aligned to trials we get the event of interest in
        the centre of 6 secs. We can change this by setting experiment.trial_duration.
        """
        self._trial_duration = secs
        for session in self.sessions:
            session.trial_duration = secs

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

    def process_motion_tracking(self):
        """
        Process motion tracking data either from raw camera data, or from
        previously-generated deeplabcut coordinate data, for all sessions.
        """
        for session in self.sessions:
            session.process_motion_tracking()

    def align_trials(self, label, event, data):
        """
        Get trials aligned to an event.

        Parameters
        ----------
        label : int
            An action label value to specify which trial types are desired.

        event : int
            An event type value to specify which event to align the trials to.

        data : str
            One of 'behaviour', 'spikes' or 'lfp'.

        """
        df = []
        for session in self.sessions:
            df.append(session.align_trials(label, event, data))
        return df
