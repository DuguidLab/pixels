"""
This module manipulates files and analyses data at the session level.
"""


import datetime
import json
import os

from pixels import ioutils


_SAMPLE_RATE = 1000


class Session:
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
        self.metadata = metadata
        self.data_dir = data_dir
        self.recordings = ioutils.get_data_files(data_dir, name)

    def extract_spikes(self, resample=True):
        """
        Extract the spikes from raw spike data.
        """

    def process_lfp(self, resample=True):
        """
        Process the LFP data from the raw neural recording data.
        """

    def process_behaviour(self, resample=True):
        """
        Process behavioural data from raw tdms files.
        """
        for recording in self.recordings:
            behavioural_data = ioutils.read_tdms(recording['behaviour'])
            sync_channel = ioutils.read_tdms(
                recording['spike_data'], "/'NpxlSync_Signal'/'0'"
            )

    def process_motion_tracking(self, resample=True):
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
