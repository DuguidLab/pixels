"""
This module contains helper functions for reading and writing files.
"""


import datetime
import glob
import json
import os
from pathlib import Path
from tempfile import gettempdir

import cv2
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

    spike_data = sorted(glob.glob(f'{data_dir}/{session_name}_g[0-9]_t0.imec[0-9].ap.bin*'))
    spike_meta = sorted(glob.glob(f'{data_dir}/{session_name}_g[0-9]_t0.imec[0-9].ap.meta*'))
    lfp_data = sorted(glob.glob(f'{data_dir}/{session_name}_g[0-9]_t0.imec[0-9].lf.bin*'))
    lfp_meta = sorted(glob.glob(f'{data_dir}/{session_name}_g[0-9]_t0.imec[0-9].lf.meta*'))
    behaviour = sorted(glob.glob(f'{data_dir}/[0-9a-zA-Z_-]*([0-9]).tdms*'))

    if not spike_data:
        raise PixelsError(f"{session_name}: could not find raw AP data file.")
    if not spike_meta:
        raise PixelsError(f"{session_name}: could not find raw AP metadata file.")
    if not lfp_data:
        raise PixelsError(f"{session_name}: could not find raw LFP data file.")
    if not lfp_meta:
        raise PixelsError(f"{session_name}: could not find raw LFP metadata file.")

    camera_data = []
    camera_meta = []
    for rec in behaviour:
        name = Path(rec).stem
        rec_vids = sorted(glob.glob(f'{data_dir}/*{name}-*.tdms*'))
        vids = [v for v in rec_vids if 'meta' not in v]
        camera_data.append(vids)
        meta = [v for v in rec_vids if 'meta' in v]
        camera_meta.append(meta)

    for num, spike_recording in enumerate(spike_data):
        recording = {}
        recording['spike_data'] = original_name(spike_recording)
        recording['spike_meta'] = original_name(spike_meta[num])
        recording['lfp_data'] = original_name(lfp_data[num])
        recording['lfp_meta'] = original_name(lfp_meta[num])

        if behaviour:
            if len(behaviour) == len(spike_data):
                recording['behaviour'] = original_name(behaviour[num])
            else:
                recording['behaviour'] = original_name(behaviour[0])
            recording['behaviour_processed'] = recording['behaviour'].with_name(
                recording['behaviour'].stem + '_processed.h5'
            )

            # We only have videos if we also have behavioural TDMS data
            if len(camera_data) > num:
                recording['camera_data'] = [original_name(d) for d in camera_data[num]]
                recording['camera_meta'] = [original_name(d) for d in camera_meta[num]]
                recording['motion_index'] = Path(f'motion_index_{num}.npy')
                recording['motion_tracking'] = Path(f'motion_tracking_{num}.h5')
        else:
            recording['behaviour'] = None
            recording['behaviour_processed'] = None

        recording['action_labels'] = Path(f'action_labels_{num}.npy')
        recording['spike_processed'] = recording['spike_data'].with_name(
            recording['spike_data'].stem + '_processed.h5'
        )
        recording['spike_rate_processed'] = Path(f'spike_rate_{num}.h5')
        recording['lfp_processed'] = recording['lfp_data'].with_name(
            recording['lfp_data'].stem + '_processed.npy'
        )
        recording['lfp_sd'] = recording['lfp_data'].with_name(
            recording['lfp_data'].stem + '_sd.json'
        )
        recording['clustered_channels'] = recording['lfp_data'].with_name(
            f'channel_clustering_results_{num}.h5'
        )
        recording['depth_info'] = recording['lfp_data'].with_name(
            f'depth_info_{num}.json'
        )
        recording['catGT_ap_data'] = str(recording['spike_data']).replace("t0", "tcat")
        recording['catGT_ap_meta'] = str(recording['spike_meta']).replace("t0", "tcat") 
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
            # TODO: Use TdmsFile.open instead of read, and only load desired groups
            data = []
            for group in groups:
                channel = tdms_file[group].channels()[0]
                group_data = tdms_file[group].as_dataframe()
                group_data = group_data.rename(columns={channel.name: channel.path})
                data.append(group_data)
            df = pd.concat(data, axis=1)
    return df


def save_ndarray_as_video(video, path, frame_rate, dims=None):
    """
    Save a numpy.ndarray as video file.

    Parameters
    ----------
    video : numpy.ndarray, or generator
        Video data to save to file. It's dimensions should be (duration, height, width)
        and data should be of uint8 type. The file extension determines the resultant
        file type. Alternatively, this can be a generator that yields frames of this
        description, in which case 'dims' must also be passed.

    path : string / pathlib.Path object
        File to which the video will be saved.

    frame_rate : int
        The frame rate of the output video.

    dims : (int, int)
        (height, width) of video. This is only needed if 'video' is a generator that
        yields frames, as then the shape cannot be taken from it directly.

    """
    if isinstance(video, np.ndarray):
        _, height, width = video.shape
    else:
        height, width = dims

    path = Path(path)

    process = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{width}x{height}', r=frame_rate)
        .output(path.as_posix(), pix_fmt='yuv420p', r=frame_rate, crf=0, vcodec='libx264')
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )

    for frame in video:
        if not isinstance(frame, list):
            # We can accept a 3D array as a list of 3 2D arrays, or just one 2D array
            frame = [frame, frame, frame]
        process.stdin.write(
            np.stack(frame, axis=2)
            .astype(np.uint8)
            .tobytes()
        )

    process.stdin.close()
    process.wait()
    if not path.exists():
        raise PixelsError(f"Video creation failed: {path}")


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
    
    print('HDF5 saved to ', path)

    return


def get_sessions(mouse_ids, data_dir, meta_dir, session_date_fmt):
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

    meta_dir : str or None
        If not None, the path to the folder containing training metadata JSON files. If
        None, no metadata is collected.

    session_date_fmt : str
        String format used to extract the date from folder names.

    Returns
    -------
    list of dicts : Dictionaries containing the values that can be used to create new
        Behaviour subclass instances.

    """
    if not isinstance(mouse_ids, (list, tuple, set)):
        mouse_ids = [mouse_ids]
    sessions = {}
    raw_dir = data_dir / 'raw'

    for mouse in mouse_ids:
        mouse_sessions = list(raw_dir.glob(f'*{mouse}*'))

        if not mouse_sessions:
            print(f'Found no sessions for: {mouse}')
            continue

        if not meta_dir:
            # Do not collect metadata
            for session in mouse_sessions:
                name = session.stem
                if name not in sessions:
                    sessions[name] = []
                sessions[name].append(dict(
                    metadata=None,
                    data_dir=data_dir,
                ))
            continue

        meta_file = meta_dir / (mouse + '.json')
        with meta_file.open() as fd:
            mouse_meta = json.load(fd)
        # az: change date format into yyyymmdd
        session_dates = [
            datetime.datetime.strptime(s.stem.split("_")[0], session_date_fmt) for s in mouse_sessions
        ]

        if len(session_dates) != len(set(session_dates)):
            raise PixelsError(f"{mouse}: Data folder dates must be unique.")

        included_sessions = set()
        for i, session in enumerate(mouse_meta):
            try:
                meta_date = datetime.datetime.strptime(session['date'], '%Y-%m-%d')
            except ValueError:
                # also allow this format
                meta_date = datetime.datetime.strptime(session['date'], '%Y%m%d')
            except TypeError:
                raise PixelsError(f"{mouse} session #{i}: 'date' not found in JSON.")

            for index, ses_date in enumerate(session_dates):
                if ses_date == meta_date and not session.get('exclude', False):
                    name = mouse_sessions[index].stem
                    if name not in sessions:
                        sessions[name] = []
                    sessions[name].append(dict(
                        metadata=session,
                        data_dir=data_dir,
                    ))
                    included_sessions.add(name)

        if included_sessions:
            print(f'{mouse} has {len(included_sessions)} sessions:', ", ".join(included_sessions))
        else:
            print(f'No session dates match between folders and metadata for: {mouse}')

    return sessions


def tdms_parse_timestamps(metadata):
    """Extract timestamps from video metadata."""
    ts_high = np.uint64(metadata["/'keys'/'IMAQdxTimestampHigh'"])
    ts_low = np.uint64(metadata["/'keys'/'IMAQdxTimestampLow'"])
    stamps = ts_low + np.left_shift(ts_high, 32)
    return stamps / 1000000


def _parse_tdms_metadata(meta_path):
    meta = read_tdms(meta_path)

    stamps = tdms_parse_timestamps(meta)
    rate = round(np.median(np.diff(stamps)))
    print(f"    Frame rate is {rate} ms per frame, {1000/rate} fps")

    # Indexes of the dropped frames
    if "/'frames'/'ind_skipped'" in meta:
        # We add one here to account for 1-based indexing
        # (I think. Compare with where actual_heights == 0)
        skipped = meta["/'frames'/'ind_skipped'"].dropna().size
        print(f"    Warning: video has skipped {skipped} frames.")
    else:
        skipped = 0

    actual_heights = meta["/'keys'/'IMAQdxActualHeight'"]
    height = int(actual_heights.max())  # Largest height is presumably the real one
    # The number of points with heights==0 should match skipped
    remainder = skipped - actual_heights[actual_heights != height].size
    duration = actual_heights.size - remainder
    fps = 1000 / rate

    return fps, height, duration


def load_tdms_video(path, meta_path, frame=None):
    """
    Calculate the 3 dimensions for a given video from TDMS metadata and reshape the
    video to these dimensions.

    Parameters
    ----------
    path : pathlib.Path
        File path to TDMS video file.

    meta_path : pathlib.Path
        File path to TDMS file containing metadata about the video.

    frame : int, optional
        Read this one single frame rather than them all.

    """
    fps, height, duration = _parse_tdms_metadata(meta_path)

    if frame is None:
        video = read_tdms(path)
        width = int(video.size / (duration * height))
        return video.values.reshape(duration, height, width), fps

    with TdmsFile.open(path) as tdms_file:
        group = tdms_file.groups()[0]
        channel = group.channels()[0]
        width = int(len(channel) / (duration * height))
        length = width * height
        start = frame * length
        video = channel[start : start + length]
        return video.reshape(height, width), fps


def tdms_to_video(tdms_path, meta_path, output_path):
    """
    Convert a TDMS video to a video file. This streams data from TDMS to the saved video
    in a way that never loads all data into memory, so works well on huge videos.

    Parameters
    ----------
    tdms_path : pathlib.Path
        File path to TDMS video file.

    meta_path : pathlib.Path
        File path to TDMS file containing metadata about the video.

    output_path : pathlib.Path
        Save the video to this file. The video format used is taken from the file
        extension of this path.

    """
    fps, height, duration = _parse_tdms_metadata(meta_path)

    tdms_file = TdmsFile.open(tdms_path)
    group = tdms_file.groups()[0]
    channel = group.channels()[0]

    if height == 480:
        # Normally we get duration from _parse_tdms_metadata, but on the occasion where
        # the metadata file has not been saved for whatever reason - which has happened
        # at least one time - if we know the height is 480 we can assume the width is
        # 640 and calculate the duration from the video's size itself
        width = 640
        duration = len(channel) // (height * width)
    else:
        width = int(len(channel) / (duration * height))
    step = width * height

    process = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{width}x{height}', r=fps)
        .output(output_path.as_posix(), pix_fmt='yuv420p', r=fps, crf=20, vcodec='libx264')
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )

    for frame in range(duration):
        pixels = channel[frame * step: (frame + 1) * step].reshape((width, height))
        process.stdin.write(
            np.stack([pixels, pixels, pixels], axis=2)
            .astype(np.uint8)
            .tobytes()
        )

    process.stdin.close()
    process.wait()
    tdms_file.close()


def load_video_frame(path, frame):
    """
    Load a frame from a video into a numpy array.

    Parameters
    ----------
    path : str
        File path to a video file.

    frame : int
        0-based index of frame to load.

    """
    video = cv2.VideoCapture(path)

    retval = video.set(cv2.CAP_PROP_POS_FRAMES, frame)
    assert retval  # Check it worked fine

    retval, frame = video.read()
    assert retval  # Check it worked fine

    return frame


def load_video_frames(path, frames):
    """
    Load a consecutive sequence of frames from a video into a numpy array.

    Parameters
    ----------
    path : str
        File path to a video file.

    frame : Sequence
        Array/list/etc of 0-based indices of frames to load. This function only
        considers the first value and the length, it doesn't check the actual values of
        the remaining elements.

    """
    if not isinstance(path, str):
        path = path.as_posix()

    video = cv2.VideoCapture(path)

    retval = video.set(cv2.CAP_PROP_POS_FRAMES, frames[0])
    assert retval  # Check it worked fine

    return stream_video(video, length=len(frames))


def get_video_dimensions(path):
    """
    Get a tuple of (width, height, frames) for a video.

    Parameters
    ----------
    path : str
        File path to a video file.

    """
    video = cv2.VideoCapture(path)

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    return width, height, frames


def get_video_fps(path):
    """
    Get the frame rate of a video.

    Parameters
    ----------
    path : str
        File path to a video file.

    """
    video = cv2.VideoCapture(path)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    return fps


def stream_video(video, length=None):
    """
    Iterate over a video's frames.

    Parameters
    ----------
    path : str or cv2.VideoCapture
        File path to a video file to open, or already open VideoCapture instance.

    length : int
        Positive integer representing the number of frames to load.

    """
    if not isinstance(video, cv2.VideoCapture):
        if isinstance(video, Path):
            video = video.as_posix()
        video = cv2.VideoCapture(video)

    if length is not None:
        assert length > 0

    while True:
        _, pixels = video.read()
        if pixels is None:
            break

        yield pixels[:, :, 0]  # TODO: should be 1 channel

        if length is not None:
            length -= 1
            if length == 0:
                break
