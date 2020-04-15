"""
This module manipulates files and analyses data at the session level.
"""


import datetime
import json
import numpy as np
import os
import scipy.signal
import time
import matplotlib.pyplot as plt
from pathlib import Path

from pixels import ioutils
from pixels import signal


class Session:

    sample_rate = 1000

    def __init__(self, name, metadata=None, data_dir=None):
        """
        This class represents a single individual recording session.

        Parameters
        ----------
        name : str
            The name of the session in the form YYMMDD_mouseID.

        metadata : dict (optional)
            A dictionary of metadata for this session. This is typically taken from the
            session's JSON file.

        data_dir : str (optional)
            The folder in which data for this session is stored.

        """
        self.name = name
        self.data_dir = data_dir
        self.metadata = metadata
        self.files = ioutils.get_data_files(data_dir, name)

        self.spike_meta = [ioutils.read_meta(f['spike_meta']) for f in self.files]
        self.lfp_meta = [ioutils.read_meta(f['lfp_meta']) for f in self.files]

    def extract_spikes(self):
        """
        Extract the spikes from raw spike data.
        """

    def process_lfp(self):
        """
        Process the LFP data from the raw neural recording data.
        """

    def process_behaviour(self):
        """
        Process behavioural data from raw tdms and align to neuropixels data.
        """
        for rec_num, recording in enumerate(self.files):
            print(f">>>>> Processing behaviour for recording {rec_num + 1} of {len(self.files)}")

            print("[1/9] Loading behaviour TDMS")
            behavioural_data = ioutils.read_tdms(recording['behaviour'])

            print("[2/9] Loading neuropixels sync channel")
            sync_pixels = ioutils.read_bin(
                recording['spike_data'],
                self.spike_meta[rec_num]['nSavedChans'],
                channel=384,
            )
            sync_pixels = syncpixels[:15000000]

            print(f"[3/9] Downsampling to {self.sample_rate} Hz")
            sync_behav = signal.resample(behavioural_data["/'NpxlSync_Signal'/'0'"], 25000)
            sync_behav = (sync_behav > 2.5).astype(np.int8)
            sync_pixels = signal.resample(sync_pixels, 30000)
            sync_pixels = (sync_pixels > 35).astype(np.int8).squeeze()

            print(f"[4/9] Finding lag between sync channels")
            plot_path = Path(recording['spike_data'])
            plot_path = plot_path.with_name(plot_path.stem + '.png')
            lag, match = signal.find_sync_lag(
                sync_behav, sync_pixels, length=60000, plot=plot_path
            )
            raise Exception

    def process_motion_tracking(self):
        """
        Process motion tracking data either from raw camera data, or from
        previously-generated deeplabcut coordinate data.
        """


def get_sessions(mouse_ids, data_dir, meta_dir):
    """
    Get a list of recording sessions for the specified mice, excluding those whose
    metadata contain '"exclude" = True'.

    Parameters
    ----------
    mouse_ids : list of strs
        List of mouse IDs.

    data_dir : str
        The path to the folder containing data for all sessions. This is searched for
        available sessions.

    meta_dir : str
        The path to the folder containing training metadata JSON files.

    """
    if not isinstance(mouse_ids, (list, tuple, set)):
        mouse_ids = [mouse_ids]
    available_sessions = os.listdir(data_dir)
    sessions = []

    for mouse in mouse_ids:
        mouse_sessions = []
        for session in available_sessions:
            if mouse in session:
                mouse_sessions.append(session)

        if mouse_sessions:
            with open(os.path.join(meta_dir, mouse + '.json'), 'r') as fd:
                mouse_meta = json.load(fd)
            session_dates = [
                datetime.datetime.strptime(s[0:6], '%y%m%d') for s in mouse_sessions
            ]
            for session in mouse_meta:
                meta_date = datetime.datetime.strptime(session['date'], '%Y-%m-%d')
                for index, ses_date in enumerate(session_dates):
                    if ses_date == meta_date:
                        if session.get('exclude', False):
                            continue
                        sessions.append(Session(
                            mouse_sessions[index],
                            metadata=session,
                            data_dir=data_dir,
                        ))
        else:
            print(f'Found no sessions for: {mouse}')

    return sessions
