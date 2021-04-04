"""
This module provides an Experiment class which serves as the main interface to process
data and run subsequent analyses.
"""


from pathlib import Path

import pandas as pd

from pixels import ioutils
from pixels.error import PixelsError


class Experiment:
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
    def __init__(self, mouse_ids, behaviour, data_dir, meta_dir):
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

    def set_cache(self, on):
        for session in self.sessions:
            session.set_cache(on)

    def process_spikes(self):
        """
        Process the spike data from the raw neural recording data for all sessions.
        """
        for i, session in enumerate(self.sessions):
            print(">>>>> Processing spikes for session {} ({} / {})"
                   .format(session.name, i + 1, len(self.sessions)))
            session.process_spikes()

    def sort_spikes(self):
        """
        Extract the spikes from raw spike data for all sessions.
        """
        for i, session in enumerate(self.sessions):
            print(">>>>> Sorting spikes for session {} ({} / {})"
                   .format(session.name, i + 1, len(self.sessions)))
            session.sort_spikes()

    def process_lfp(self):
        """
        Process the LFP data from the raw neural recording data for all sessions.
        """
        for i, session in enumerate(self.sessions):
            print(">>>>> Processing LFP data for session {} ({} / {})"
                   .format(session.name, i + 1, len(self.sessions)))
            session.process_lfp()

    def process_behaviour(self):
        """
        Process behavioural data from raw tdms files for all sessions.
        """
        for i, session in enumerate(self.sessions):
            print(">>>>> Processing behaviour for session {} ({} / {})"
                   .format(session.name, i + 1, len(self.sessions)))
            session.process_behaviour()

    def extract_videos(self):
        """
        Extract videos from TDMS in the raw folder to avi files in the interim folder.
        """
        for i, session in enumerate(self.sessions):
            print(">>>>> Extracting videos for session {} ({} / {})"
                   .format(session.name, i + 1, len(self.sessions)))
            session.extract_videos()

    def process_motion_tracking(self, config, create_labelled_video=True):
        """
        Process motion tracking data either from raw camera data, or from
        previously-generated deeplabcut coordinate data, for all sessions.
        """
        for i, session in enumerate(self.sessions):
            print(">>>>> Processing motion tracking for session {} ({} / {})"
                   .format(session.name, i + 1, len(self.sessions)))
            session.process_motion_tracking(config, create_labelled_video)

    def align_trials(self, *args, **kwargs):
        """
        Get trials aligned to an event. Check behaviours.base.Behaviour.align_trials for
        usage information.
        """
        trials = []
        for session in self.sessions:
            trials.append(session.align_trials(*args, **kwargs))

        df = pd.concat(
            trials, axis=1, copy=False,
            keys=range(len(trials)),
            names=["session", "rec_num", "unit", "trial"]
        )
        return df

    def get_cluster_info(self):
        """
        Get some basic high-level information for each cluster. This is mostly just the
        information seen in the table in phy.
        """
        return [s.get_cluster_info() for s in self.sessions]

    def get_spike_widths(self, group='good', min_depth=0, max_depth=None):
        """
        Get the widths of spikes for units matching the specified criteria.
        """
        widths = [
            s.get_spike_widths(group, min_depth, max_depth) for s in self.sessions
        ]

        df = pd.concat(
            widths, axis=1, copy=False,
            keys=range(len(widths)),
            names=["session"]
        )
        return df

    def get_spike_waveforms(
        self, group='good', min_depth=0, max_depth=None, min_spike_width=None,
        max_spike_width=None
    ):
        """
        Get the waveforms of spikes for units matching the specified criteria.
        """
        waveforms = [
            s.get_spike_waveforms(
                group=group,
                min_depth=min_depth, max_depth=max_depth,
                min_spike_width=min_spike_width, max_spike_width=max_spike_width,
            ) for s in self.sessions
        ]

        df = pd.concat(
            waveforms, axis=1, copy=False,
            keys=range(len(waveforms)),
            names=["session"]
        )
        return df

    def get_aligned_spike_rate_CI(self, *args, **kwargs):
        CIs = [
            s.get_aligned_spike_rate_CI(*args, **kwargs) for s in self.sessions
        ]

        df = pd.concat(
            CIs, axis=1, copy=False,
            keys=range(len(CIs)),
            names=["session"]
        )
        return df
