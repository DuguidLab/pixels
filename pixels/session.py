"""
This module manipulates files and analyses at the session level.
"""


import datetime
import json
import os

from pixels import ioutils


_SAMPLE_RATE = 1000


class Session:
    def __init__(self, metadata=None):
        self.metadata = metadata

    def extract_spikes(self, resample=True):
        """
        Extract the spikes from raw spike data.
        """

    def process_lfp(self):
        """
        Process the LFP data from the raw neural recording data.
        """

    def process_behaviour(self):
        """
        Process behavioural data from raw tdms files.
        """

    def process_motion_tracking(self):
        """
        Process motion tracking data either from raw camera data, or from
        previously-generated deeplabcut coordinate data.
        """


def list_sessions(mouse_ids, data_dir, meta_dir):
    """
    Get a list of recording sessions for the specified mice, excluding those whose
    metadata contain '"exclude" = True'.
    """
    meta_dir = os.path.expanduser(meta_dir)
    available_sessions = os.listdir(os.path.expanduser(data_dir))
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
                        mouse_sessions[index] = Session(metadata=session)
        else:
            print(f'Found no sessions for: {mouse}')

        sessions.extend(mouse_sessions)

    return sessions
