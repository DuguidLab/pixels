"""
This module contains helper functions for reading and writing files.
"""


import glob
import os
from pathlib import Path
from nptdms import TdmsFile


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
    with TdmsFile.open(path) as tdms_file:
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


def read_meta(path):
    """
    Read metadata from a .meta file file.

    Parameters
    ----------
    path : str
        Path to the meta file to be read.

    """
    metadata = {}
    for entry in Path(path).read_text().split("\n"):
        if entry:
            key, value = entry.split("=")
            metadata[key] = value
    return metadata


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
    if session_name not in data_dir:
        data_dir = glob.glob(os.path.join(data_dir, f'{session_name}*'))[0]
    files = []

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
        recording['spike_data'] = spike_recording
        recording['spike_meta'] = spike_meta[num]
        recording['lfp_data'] = lfp_data[num]
        recording['lfp_meta'] = lfp_meta[num]
        if len(behaviour) == len(spike_data):
            recording['behaviour'] = behaviour[num]
        else:
            recording['behaviour'] = behaviour
        recording['camera_data'] = camera_data[num]
        recording['camera_meta'] = camera_meta[num]
        files.append(recording)

    return files
