"""
This module contains helper functions for reading and writing files.
"""


import datetime
import glob
import json
import os
from pathlib import Path
from tempfile import gettempdir

import ffmpeg
import numpy as np
import pandas as pd
from nptdms import TdmsFile

from pixels.error import PixelsError


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

    spike_data = glob.glob(f'{data_dir}/{session_name}_g[0-9]_t0.imec0.ap.bin*')
    spike_meta = glob.glob(f'{data_dir}/{session_name}_g[0-9]_t0.imec0.ap.meta*')
    lfp_data = glob.glob(f'{data_dir}/{session_name}_g[0-9]_t0.imec0.lf.bin*')
    lfp_meta = glob.glob(f'{data_dir}/{session_name}_g[0-9]_t0.imec0.lf.meta*')
    behaviour = glob.glob(f'{data_dir}/[0-9a-zA-Z_-]*([0-9]).tdms*')
    camera = glob.glob(f'{data_dir}/*Camera*.tdms*')
    camera_data = []
    camera_meta = []
    for match in camera:
        if 'meta' in match:
            camera_meta.append(match)
        else:
            camera_data.append(match)

    if not (spike_data and spike_meta and lfp_data and lfp_meta):
        raise PixelsError(f"{session_name}: raw files not correctly named.")

    for num, spike_recording in enumerate(spike_data):
        recording = {}
        recording['spike_data'] = original_name(spike_recording)
        recording['spike_meta'] = original_name(spike_meta[num])
        recording['lfp_data'] = original_name(lfp_data[num])
        recording['lfp_meta'] = original_name(lfp_meta[num])
        if len(behaviour) == len(spike_data):
            recording['behaviour'] = original_name(behaviour[num])
        else:
            recording['behaviour'] = original_name(behaviour[0])
        recording['camera_data'] = original_name(camera_data[num])
        recording['camera_meta'] = original_name(camera_meta[num])
        recording['action_labels'] = Path(f'action_labels_{num}.npy')
        recording['behaviour_processed'] = recording['behaviour'].with_name(
            recording['behaviour'].stem + '_processed.h5'
        )
        recording['spike_processed'] = recording['spike_data'].with_name(
            recording['spike_data'].stem + '_processed.h5'
        )
        recording['lfp_processed'] = recording['lfp_data'].with_name(
            recording['lfp_data'].stem + '_processed.h5'
        )
        files.append(recording)

    return files


def original_name(path):
    """
    Get the original name of a file, uncompressed, as a pathlib.Path.
    """
    name = os.path.basename(path)
    if name.endswith('.tar.gz'):
        name = name[:-7]
    return Path(name)


def read_meta(path):
    """
    Read metadata from a .meta file.

    Parameters
    ----------
    path : pathlib.Path
        Path to the meta file to be read.

    Returns
    -------
    dict : A dictionary containing the metadata from the specified file.

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

    channel : int or slice, optional
        The channel to read. If None (default), all channels are read.

    Returns
    -------
    numpy.memmap array : A 2D memory-mapped array containing containing the binary
        file's data.

    """
    if not isinstance(num_chans, int):
        num_chans = int(num_chans)

    mapping = np.memmap(path, np.int16, mode='r').reshape((-1, num_chans))

    if channel is not None:
        mapping = mapping[:, channel]

    return mapping


def read_tdms(path, groups=None):
    """
    Read data from a TDMS file.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the TDMS file to be read.

    groups : list of strs (optional)
        Names of groups stored inside the TDMS file that should be loaded. By default,
        all groups are loaded, so specifying the groups you want explicitly can avoid
        loading the entire file from disk.

    Returns
    -------
    pandas.DataFrame : A dataframe containing the data from the TDMS file.

    """
    with TdmsFile.read(path, memmap_dir=gettempdir()) as tdms_file:
        if groups is None:
            df = tdms_file.as_dataframe()
        else:
            data = []
            for group in groups:
                channel = tdms_file[group].channels()[0]
                group_data = tdms_file[group].as_dataframe()
                group_data = group_data.rename(columns={channel.name: channel.path})
                data.append(group_data)
            df = pd.concat(data, axis=1)
    return df


def save_ndarray_as_avi(video, path, frame_rate):
    """
    Save a numpy.ndarray as an .avi video.

    Parameters
    ----------
    video : numpy.ndarray
        Video data to save to file. It's dimensions should be (duration, height, width)
        and data should be of uint8 type.

    path : string / pathlib.Path object
        File to which the video will be saved.

    frame_rate : int
        The frame rate of the output video.

    """
    _, height, width = video.shape
    path = Path(path)

    process = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{width}x{height}')
        .output(path.as_posix(), pix_fmt='yuv420p', vcodec='libx264', r=frame_rate)
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )

    for frame in video:
        process.stdin.write(
            np.stack([frame, frame, frame], axis=2)
            .astype(np.uint8)
            .tobytes()
        )

    process.stdin.close()
    process.wait()
    if not path.exists():
        raise PixelsError(f"AVI creation failed: {path}")


def read_hdf5(path):
    """
    Read a dataframe from a h5 file.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the h5 file to read.

    Returns
    -------
    pandas.DataFrame : The dataframe stored within the hdf5 file under the name 'df'.

    """
    df = pd.read_hdf(path, 'df')
    return df


def write_hdf5(path, df):
    """
    Write a dataframe to an h5 file.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the h5 file to write to.

    df : pd.DataFrame
        Dataframe to save to h5.

    """
    df.to_hdf(path, 'df', mode='w')
    return


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

    Returns
    -------
    list of dicts : Dictionaries containing the values that can be used to create new
        Behaviour subclass instances.

    """
    if not isinstance(mouse_ids, (list, tuple, set)):
        mouse_ids = [mouse_ids]
    sessions = []
    raw_dir = data_dir / 'raw'

    for mouse in mouse_ids:
        mouse_sessions = list(raw_dir.glob(f'*{mouse}*'))

        if mouse_sessions:
            meta_file = meta_dir / (mouse + '.json')
            with meta_file.open() as fd:
                mouse_meta = json.load(fd)
            session_dates = [
                datetime.datetime.strptime(s.stem[0:6], '%y%m%d') for s in mouse_sessions
            ]

            s = 0
            for i, session in enumerate(mouse_meta):
                try:
                    meta_date = datetime.datetime.strptime(session['date'], '%Y-%m-%d')
                except TypeError:
                    raise PixelsError(f"{mouse} session #{i}: 'date' not found in JSON.")

                for index, ses_date in enumerate(session_dates):
                    if ses_date == meta_date and not session.get('exclude', False):
                        s += 1
                        sessions.append(dict(
                            sessions=mouse_sessions[index].stem,
                            metadata=session,
                            data_dir=data_dir,
                        ))
            if s == 0:
                print(f'No session dates match between folders and metadata for: {mouse}')

        else:
            print(f'Found no sessions for: {mouse}')

    return sessions
