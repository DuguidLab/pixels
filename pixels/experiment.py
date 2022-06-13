"""
This module provides an Experiment class which serves as the main interface to process
data and run subsequent analyses.
"""


from operator import attrgetter, itemgetter
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

    interim_dir : str
        Path to the folder to use for interim files. If not passed, this will default to
        a folder inside the data_dir called 'interim'.

    session_date_fmt : str
        A format string used to parse the date from the name of session folders. By
        default this is "%y%m%d" which will capture YYMMDD formats.

    """
    def __init__(
        self,
        mouse_ids,
        behaviour,
        data_dir,
        meta_dir=None,
        interim_dir=None,
        session_date_fmt="%y%m%d",
    ):
        if not isinstance(mouse_ids, (list, tuple, set)):
            mouse_ids = [mouse_ids]

        self.behaviour = behaviour
        self.mouse_ids = mouse_ids

        self.data_dir = Path(data_dir).expanduser()
        if not self.data_dir.exists():
            raise PixelsError(f"Directory not found: {data_dir}")

        if meta_dir:
            self.meta_dir = Path(meta_dir).expanduser()
            if not self.meta_dir.exists():
                raise PixelsError(f"Directory not found: {meta_dir}")
        else:
            self.meta_dir = None

        self.sessions = []
        sessions = ioutils.get_sessions(mouse_ids, self.data_dir, self.meta_dir, session_date_fmt)

        for name, metadata in sessions.items():
            assert len(set(s['data_dir'] for s in metadata)) == 1, "All JSON items with same day must use same data folder."
            self.sessions.append(
                behaviour(
                    name,
                    metadata=[s['metadata'] for s in metadata],
                    data_dir=metadata[0]['data_dir'],
                    interim_dir=interim_dir,
                )
            )

        self.sessions.sort(key=attrgetter("name"))

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
        """ Extract the spikes from raw spike data for all sessions.  """
        for i, session in enumerate(self.sessions):
            print(">>>>> Sorting spikes for session {} ({} / {})"
                   .format(session.name, i + 1, len(self.sessions)))
            session.sort_spikes()

    def assess_noise(self):
        """
        Assess the noise for the raw AP data.
        """
        for i, session in enumerate(self.sessions):
            print(">>>>> Assessing noise for session {} ({} / {})"
                   .format(session.name, i + 1, len(self.sessions)))
            session.assess_noise()

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

    def extract_videos(self, force=False):
        """
        Extract videos from TDMS in the raw folder to avi files in the interim folder.
        """
        for i, session in enumerate(self.sessions):
            print(">>>>> Extracting videos for session {} ({} / {})"
                   .format(session.name, i + 1, len(self.sessions)))
            session.extract_videos(force=force)

    def configure_motion_tracking(self, project):
        """
        Process motion tracking data either from raw camera data, or from
        previously-generated deeplabcut coordinate data, for all sessions.
        """
        for i, session in enumerate(self.sessions):
            print(">>>>> Configuring motion tracking for session {} ({} / {})"
                   .format(session.name, i + 1, len(self.sessions)))
            session.configure_motion_tracking(project)

    def run_motion_tracking(self, *args, **kwargs):
        """
        Run motion tracking on camera data for all sessions.
        """
        for i, session in enumerate(self.sessions):
            print(">>>>> Running motion tracking for session {} ({} / {})"
                   .format(session.name, i + 1, len(self.sessions)))
            session.run_motion_tracking(*args, **kwargs)

    def draw_motion_index_rois(self, video_match, num_rois=1):
        """
        Draw motion index ROIs using EasyROI. If ROIs already exist, skip.
        """
        for i, session in enumerate(self.sessions):
            print(">>>>> Drawing motion index ROIs for session {} ({} / {})"
                   .format(session.name, i + 1, len(self.sessions)))
            session.draw_motion_index_rois(video_match, num_rois=num_rois)

    def process_motion_index(self, video_match, num_rois=1):
        """
        Extract motion indexes from videos for all sessions.
        """
        for session in self.sessions:
            session.draw_motion_index_rois(video_match, num_rois=num_rois)

        for i, session in enumerate(self.sessions):
            print(">>>>> Processing motion index for session {} ({} / {})"
                   .format(session.name, i + 1, len(self.sessions)))
            session.process_motion_index(video_match)

    def select_units(self, *args, **kwargs):
        """
        Select units based on specified criteria. The output of this can be passed to
        some other methods to apply those methods only to these units.
        """
        units = []

        for i, session in enumerate(self.sessions):
            units.append(session.select_units(*args, **kwargs))

        return units

    def align_trials(self, *args, units=None, **kwargs):
        """
        Get trials aligned to an event. Check behaviours.base.Behaviour.align_trials for
        usage information.
        """
        trials = {}
        for i, session in enumerate(self.sessions):
            if units:
                if units[i]:
                    trials[i] = session.align_trials(*args, units=units[i], **kwargs)
            else:
                trials[i] = session.align_trials(*args, **kwargs)

        if "motion_tracking" in args:
            df = pd.concat(
                trials.values(), axis=1, copy=False,
                keys=trials.keys(),
                names=["session", "trial", "scorer", "bodyparts", "coords"]
            )

        else:
            df = pd.concat(
                trials.values(), axis=1, copy=False,
                keys=trials.keys(),
                names=["session"] + trials[0].columns.names,
            )

        return df

    def align_clips(self, label, event, video_match, duration=1):
        trials = []
        for session in self.sessions:
            trials.append(session.align_clips(label, event, video_match, duration))

        df = pd.concat(trials, axis=1, copy=False, names=["Session"], keys=range(len(trials)))
        return df

    def get_cluster_info(self):
        """
        Get some basic high-level information for each cluster. This is mostly just the
        information seen in the table in phy.
        """
        return [s.get_cluster_info() for s in self.sessions]

    def get_spike_widths(self, units=None):
        """
        Get the widths of spikes for units matching the specified criteria.
        """
        widths = {}

        for i, session in enumerate(self.sessions):
            if units:
                if units[i]:
                    widths[i] = session.get_spike_widths(units=units[i])
            else:
                widths[i] = session.get_spike_widths()

        df = pd.concat(
            widths.values(), axis=1, copy=False,
            keys=widths.keys(),
            names=["session"]
        )
        return df

    def get_spike_waveforms(self, units=None):
        """
        Get the waveforms of spikes for units matching the specified criteria.
        """
        waveforms = {}

        for i, session in enumerate(self.sessions):
            if units:
                if units[i]:
                    waveforms[i] = session.get_spike_waveforms(units=units[i])
            else:
                waveforms[i] = session.get_spike_waveforms()

        df = pd.concat(
            waveforms.values(), axis=1, copy=False,
            keys=waveforms.keys(),
            names=["session"]
        )
        return df

    def get_aligned_spike_rate_CI(self, *args, units=None, **kwargs):
        """
        Get the confidence intervals of the mean firing rates within a window aligned to
        a specified action label and event.
        """
        CIs = []

        for i, session in enumerate(self.sessions):
            if units:
                ses_cis = session.get_aligned_spike_rate_CI(*args, units=units[i], **kwargs)
            else:
                ses_cis = session.get_aligned_spike_rate_CI(*args, **kwargs)
            CIs.append(ses_cis)

        df = pd.concat(
            CIs, axis=1, copy=False,
            keys=range(len(CIs)),
            names=["session"]
        )
        return df

    def get_session_by_name(self, name: str):
        for session in self.sessions:
            if session.name == name:
                return session
        raise PixelsError
