"""
This module contains helper functions for reading and writing files.
"""


import datetime
import glob
import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from nptdms import TdmsFile


def get_data_files(data_dir, session_name):
    """
    Get the file names of raw data for a session.

    Parameters
    ----------
    data_dir : str
        The directory containing the data.

    session_name : str
        The name of the session for which to get file names.

    Returns
    -------
    A list of dicts, where each dict corresponds to one recording. The dict will contain
    these keys to identify data files:

        - spike_data
        - spike_meta
        - lfp_data
        - lfp_meta
        - behaviour
        - camera_data
        - camera_meta

    """
    if session_name != data_dir.stem:
        data_dir = list(data_dir.glob(f'{session_name}*'))[0]
    files = []

    spike_data = glob.glob(f'{data_dir}/{session_name}_g[0-9]_t0.imec0.ap.bin')
    spike_data = glob.glob(f'{data_dir}/{session_name}_g[0-9]_t0.imec0.ap.bin')
    spike_meta = glob.glob(f'{data_dir}/{session_name}_g[0-9]_t0.imec0.ap.meta')
    lfp_data = glob.glob(f'{data_dir}/{session_name}_g[0-9]_t0.imec0.lf.bin')
    lfp_meta = glob.glob(f'{data_dir}/{session_name}_g[0-9]_t0.imec0.lf.meta')
    behaviour = glob.glob(f'{data_dir}/NeuropixelBehaviour([0-9]).tdms')
    camera = glob.glob(f'{data_dir}/USB_Camera*.tdms')
    camera_data = []
    camera_meta = []
    for match in camera:
        if 'meta' in match:
            camera_meta.append(match)
        else:
            camera_data.append(match)

    for num, spike_recording in enumerate(spike_data):
        recording = {}
        recording['spike_data'] = Path(spike_recording)
        recording['spike_meta'] = Path(spike_meta[num])
        recording['lfp_data'] = Path(lfp_data[num])
        recording['lfp_meta'] = Path(lfp_meta[num])
        if len(behaviour) == len(spike_data):
            recording['behaviour'] = Path(behaviour[num])
        else:
            recording['behaviour'] = Path(behaviour[0])
        recording['camera_data'] = Path(camera_data[num])
        recording['camera_meta'] = Path(camera_meta[num])
        recording['action_labels'] = data_dir / f'action_labels_{num}.npy'
        files.append(recording)

    return files


def read_meta(path):
    """
    Read metadata from a .meta file.

    Parameters
    ----------
    path : pathlib.Path
        Path to the meta file to be read.

    """
    metadata = {}
    for entry in path.read_text().split("\n"):
        if entry:
            key, value = entry.split("=")
            metadata[key] = value
    return metadata


def read_bin(path, num_chans, channel=None):
    """
    Read data from a bin file.

    Parameters
    ----------
    path : str
        Path to the bin file to be read.

    num_chans : int
        The number of channels of data present in the file.

    channel : int, optional
        The channel to read. If None (default), all channels are read.

    """
    if not isinstance(num_chans, int):
        num_chans = int(num_chans)

    mapping = np.memmap(path, np.int16, mode='r').reshape((-1, num_chans))

    if channel is not None:
        mapping = mapping[:, channel]

    return pd.DataFrame(data=mapping)


def read_tdms(path, groups=None):
    """
    Read data from a TDMS file.

    Parameters
    ----------
    path : str
        Path to the TDMS file to be read.

    groups : list of strs (optional)
        Names of groups stored inside the TDMS file that should be loaded. By default,
        all groups are loaded, so specifying the groups you want explicitly can avoid
        loading the entire file from disk.

    """
    with TdmsFile.read(path) as tdms_file:
        if groups is None:
            df = tdms_file.as_dataframe()
        else:
            data = []
            for group in groups:
                channel = tdms_file[group].channels()[0]
                group_data = tdms_file[group].as_dataframe()
                group_data = group_data.rename(columns={channel.name:channel.path})
                data.append(group_data)
            df = pd.concat(data, axis=1)
    return df


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
    sessions = []

    for mouse in mouse_ids:
        mouse_sessions = list(data_dir.glob(f'*{mouse}*'))

        if mouse_sessions:
            meta_file = meta_dir / (mouse + '.json') 
            with meta_file.open() as fd:
                mouse_meta = json.load(fd)
            session_dates = [
                datetime.datetime.strptime(s.stem[0:6], '%y%m%d') for s in mouse_sessions
            ]
            for session in mouse_meta:
                meta_date = datetime.datetime.strptime(session['date'], '%Y-%m-%d')
                for index, ses_date in enumerate(session_dates):
                    if ses_date == meta_date and not session.get('exclude', False):
                        sessions.append(dict(
                            sessions=mouse_sessions[index].stem,
                            metadata=session,
                            data_dir=data_dir,
                        ))
        else:
            print(f'Found no sessions for: {mouse}')

    return sessions
