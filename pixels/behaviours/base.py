"""
This module provides a base class for experimental sessions that must be used as the
base for defining behaviour-specific processing.
"""

from __future__ import annotations

import functools
import json
import os
import pickle
import tarfile
import tempfile
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from shutil import copyfile
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import probeinterface as pi
import scipy.stats
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.sorters as ss
from scipy import interpolate

from pixels import ioutils
from pixels import signal
from pixels.error import PixelsError

if TYPE_CHECKING:
    from typing import Optional, Literal

BEHAVIOUR_HZ = 25000

np.random.seed(BEHAVIOUR_HZ)


def _cacheable(method):
    """
    Methods with this decorator will have their output cached to disk so that future
    calls with the same set of arguments will simply load the result from disk. However,
    if the key word argument list contains `units` and it is not either `None` or an
    instance of `SelectedUnits` then this is disabled.
    """
    def func(*args, **kwargs):
        if "units" in kwargs:
            units = kwargs["units"]
            if not isinstance(units, SelectedUnits) or not hasattr(units, "name"):
                return method(*args, **kwargs)

        as_list = list(args) + list(kwargs.values())
        self = as_list.pop(0)
        if not self._use_cache:
            return method(*args, **kwargs)

        as_list.insert(0, method.__name__)
        output = self.interim / 'cache' / ('_'.join(str(i) for i in as_list) + '.h5')
        if output.exists() and self._use_cache != "overwrite":
            df = ioutils.read_hdf5(output)
        else:
            df = method(*args, **kwargs)
            output.parent.mkdir(parents=True, exist_ok=True)
            ioutils.write_hdf5(output, df)
        return df
    return func


class SelectedUnits(list):
    name: str
    """
    This is the return type of Behaviour.select_units, which is a list in every way
    except that when represented as a string, it can return a name, if a `name`
    attribute has been set on it. This allows methods that have had `units` passed to be
    cached to file.
    """
    def __repr__(self):
        if hasattr(self, "name"):
            return self.name
        return list.__repr__(self)


class Behaviour(ABC):
    """
    This class represents a single individual recording session.

    Parameters
    ----------
    name : str
        The name of the session in the form YYMMDD_mouseID.

    data_dir : pathlib.Path
        The top level data folder, containing raw, interim and processed folders.

    metadata : dict, optional
        A dictionary of metadata for this session. This is typically taken from the
        session's JSON file.

    """

    sample_rate = 1000

    def __init__(self, name, data_dir, metadata=None):
        self.name = name
        self.data_dir = data_dir
        self.metadata = metadata

        self.raw = self.data_dir / 'raw' / self.name
        self.interim = self.data_dir / 'interim' / self.name
        self.processed = self.data_dir / 'processed' / self.name
        self.files = ioutils.get_data_files(self.raw, name)

        self.interim.mkdir(parents=True, exist_ok=True)
        self.processed.mkdir(parents=True, exist_ok=True)

        self._action_labels = None
        self._behavioural_data = None
        self._spike_data = None
        self._spike_times_data = None
        self._lfp_data = None
        self._lag = None
        self._use_cache = True
        self._cluster_info = None
        self.drop_data()

        self.spike_meta = [
            ioutils.read_meta(self.find_file(f['spike_meta'])) for f in self.files
        ]
        self.lfp_meta = [
            ioutils.read_meta(self.find_file(f['lfp_meta'])) for f in self.files
        ]

        # environmental variable PIXELS_CACHE={0,1} can be {disable,enable} cache
        self.set_cache(bool(int(os.environ.get("PIXELS_CACHE", 1))))

    def drop_data(self):
        """
        Clear attributes that store data to clear some memory.
        """
        self._action_labels = [None] * len(self.files)
        self._behavioural_data = [None] * len(self.files)
        self._spike_data = [None] * len(self.files)
        self._spike_times_data = [None] * len(self.files)
        self._spike_rate_data = [None] * len(self.files)
        self._lfp_data = [None] * len(self.files)
        self._motion_index = [None] * len(self.files)
        self._load_lag()

    def set_cache(self, on: bool | Literal["overwrite"]) -> None:
        if isinstance(on, bool):
            self._use_cache = on
        else:
            assert on == "overwrite"
            self._use_cache = on

    def _load_lag(self):
        """
        Load previously-calculated lag information from a saved file if it exists,
        otherwise return Nones.
        """
        lag_file = self.processed / 'lag.json'
        self._lag = [None] * len(self.files)
        if lag_file.exists():
            with lag_file.open() as fd:
                lag = json.load(fd)
            for rec_num, rec_lag in enumerate(lag):
                if rec_lag['lag_start'] is None:
                    self._lag[rec_num] = None
                else:
                    self._lag[rec_num] = (rec_lag['lag_start'], rec_lag['lag_end'])

    def get_probe_depth(self):
        """
        Load probe depth in um from file if it has been recorded.
        """
        depth_file = self.processed / 'depth.txt'
        if not depth_file.exists():
            msg = f": Can't load probe depth: please add it in um to processed/{self.name}/depth.txt"
            raise PixelsError(msg)
        with depth_file.open() as fd:
            return [float(line) for line in fd.readlines()]

    def find_file(self, name: str, copy: bool=True) -> Optional[Path]:
        """
        Finds the specified file, looking for it in the processed folder, interim
        folder, and then raw folder in that order. If the the file is only found in the
        raw folder, it is copied to the interim folder and uncompressed if required.

        Parameters
        ----------
        name : str or pathlib.Path
            The basename of the file to be looked for.

        copy : bool, optional
            Normally files are copied from raw to interim. If this is False, raw files
            will be decompressed in place but not copied to the interim folder.

        Returns
        -------
        pathlib.Path : the full path to the desired file in the correct folder.

        """
        processed = self.processed / name
        if processed.exists():
            return processed

        interim = self.interim / name
        if interim.exists():
            return interim

        raw = self.raw / name
        if raw.exists():
            if copy:
                print(f"    {self.name}: Copying {name} to interim")
                copyfile(raw, interim)
                return interim
            return raw

        tar = raw.with_name(raw.name + '.tar.gz')
        if tar.exists():
            if copy:
                print(f"    {self.name}: Extracting {tar.name} to interim")
                with tarfile.open(tar) as open_tar:
                    open_tar.extractall(path=self.interim)
                return interim
            print(f"    {self.name}: Extracting {tar.name}")
            with tarfile.open(tar) as open_tar:
                open_tar.extractall(path=self.raw)
            return raw

        return None

    def sync_data(self, rec_num, behavioural_data=None, sync_channel=None):
        """
        This method will calculate the lag between the behavioural data and the
        neuropixels data for each recording and save it to file and self._lag.

        behavioural_data and sync_channel will be loaded from file and downsampled if
        not provided, otherwise if provided they must already be the same sample
        frequency.

        Parameters
        ----------
        rec_num : int
            The recording number, i.e. index of self.files to get file paths.

        behavioural_data : pandas.DataFrame, optional
            The unprocessed behavioural data loaded from the TDMS file.

        sync_channel : np.ndarray, optional
            The sync channel from either the spike or LFP data.

        """
        print("    Finding lag between sync channels")
        recording = self.files[rec_num]

        if behavioural_data is None:
            print("    Loading behavioural data")
            data_file = self.find_file(recording['behaviour'])
            behavioural_data = ioutils.read_tdms(data_file, groups=["NpxlSync_Signal"])
            behavioural_data = signal.resample(
                behavioural_data.values, BEHAVIOUR_HZ, self.sample_rate
            )

        if sync_channel is None:
            print("    Loading neuropixels sync channel")
            data_file = self.find_file(recording['lfp_data'])
            num_chans = self.lfp_meta[rec_num]['nSavedChans']
            sync_channel = ioutils.read_bin(data_file, num_chans, channel=384)
            orig_rate = int(self.lfp_meta[rec_num]['imSampRate'])
            #sync_channel = sync_channel[:120 * orig_rate * 2]  # 2 mins, rec Hz, back/forward
            sync_channel = signal.resample(sync_channel, orig_rate, self.sample_rate)

        behavioural_data = signal.binarise(behavioural_data)
        sync_channel = signal.binarise(sync_channel)

        print("    Finding lag")
        plot = self.processed / f'sync_{rec_num}.png'
        lag_start, match = signal.find_sync_lag(
            behavioural_data, sync_channel, plot=plot,
        )

        lag_end = len(behavioural_data) - (lag_start + len(sync_channel))
        self._lag[rec_num] = (lag_start, lag_end)

        if match < 95:
            print("    The sync channels did not match very well. Check the plot.")
        print(f"    Calculated lag: {(lag_start, lag_end)}")

        lag_json = []
        for lag in self._lag:
            if lag is None:
                lag_json.append(dict(lag_start=None, lag_end=None))
            else:
                lag_start, lag_end = lag
                lag_json.append(dict(lag_start=lag_start, lag_end=lag_end))
        with (self.processed / 'lag.json').open('w') as fd:
            json.dump(lag_json, fd)

    def process_behaviour(self):
        """
        Process behavioural data from raw tdms and align to neuropixels data.
        """
        for rec_num, recording in enumerate(self.files):
            print(
                f">>>>> Processing behaviour for recording {rec_num + 1} of {len(self.files)}"
            )

            print(f"> Loading behavioural data")
            behavioural_data = ioutils.read_tdms(self.find_file(recording['behaviour']))

            # ignore any columns that have Nans; these just contain settings
            for col in behavioural_data:
                if behavioural_data[col].isnull().values.any():
                    behavioural_data.drop(col, axis=1, inplace=True)

            print(f"> Downsampling to {self.sample_rate} Hz")
            behav_array = signal.resample(behavioural_data.values, 25000, self.sample_rate)
            behavioural_data.iloc[:len(behav_array), :] = behav_array
            behavioural_data = behavioural_data[:len(behav_array)]

            print(f"> Syncing to Neuropixels data")
            if self._lag[rec_num] is None:
                self.sync_data(
                    rec_num,
                    behavioural_data=behavioural_data["/'NpxlSync_Signal'/'0'"].values,
                )
            lag_start, lag_end = self._lag[rec_num]
            behavioural_data = behavioural_data[max(lag_start, 0):-1-max(lag_end, 0)]
            behavioural_data.index = range(len(behavioural_data))

            print(f"> Extracting action labels")
            self._action_labels[rec_num] = self._extract_action_labels(rec_num, behavioural_data)
            output = self.processed / recording['action_labels']
            np.save(output, self._action_labels[rec_num])
            print(f">   Saved to: {output}")

            output = self.processed / recording['behaviour_processed']
            print(f"> Saving downsampled behavioural data to:")
            print(f"    {output}")
            behavioural_data.drop("/'NpxlSync_Signal'/'0'", axis=1, inplace=True)
            ioutils.write_hdf5(output, behavioural_data)
            self._behavioural_data[rec_num] = behavioural_data

        print("> Done!")

    def process_spikes(self):
        """
        Process the spike data from the raw neural recording data.
        """
        for rec_num, recording in enumerate(self.files):
            print(
                f">>>>> Processing spike data for recording {rec_num + 1} of {len(self.files)}"
            )
            output = self.processed / recording['spike_processed']

            if output.exists():
                continue

            data_file = self.find_file(recording['spike_data'])
            orig_rate = self.spike_meta[rec_num]['imSampRate']
            num_chans = self.spike_meta[rec_num]['nSavedChans']

            print("> Mapping spike data")
            data = ioutils.read_bin(data_file, num_chans)

            print(f"> Downsampling to {self.sample_rate} Hz")
            data = signal.resample(data, orig_rate, self.sample_rate)

            # Ideally we would median subtract before downsampling, but that takes a
            # very long time and is at risk of memory errors, so let's do it after.
            print("> Performing median subtraction across rows")
            data = signal.median_subtraction(data, axis=0)
            print("> Performing median subtraction across columns")
            data = signal.median_subtraction(data, axis=1)

            if self._lag[rec_num] is None:
                self.sync_data(rec_num, sync_channel=data[:, -1])
            lag_start, lag_end = self._lag[rec_num]

            print(f"> Saving data to {output}")
            if lag_end < 0:
                data = data[:lag_end]
            if lag_start < 0:
                data = data[- lag_start:]
            data = pd.DataFrame(data[:, :-1])
            ioutils.write_hdf5(output, data)

    def process_lfp(self):
        """
        Process the LFP data from the raw neural recording data.
        """
        for rec_num, recording in enumerate(self.files):
            print(f">>>>> Processing LFP for recording {rec_num + 1} of {len(self.files)}")

            data_file = self.find_file(recording['lfp_data'])
            orig_rate = self.lfp_meta[rec_num]['imSampRate']
            num_chans = self.lfp_meta[rec_num]['nSavedChans']

            print("> Mapping LFP data")
            data = ioutils.read_bin(data_file, num_chans)

            print(f"> Downsampling to {self.sample_rate} Hz")
            data = signal.resample(data, orig_rate, self.sample_rate)

            if self._lag[rec_num] is None:
                self.sync_data(rec_num, sync_channel=data[:, -1])
            lag_start, lag_end = self._lag[rec_num]

            output = self.processed / recording['lfp_processed']
            print(f"> Saving data to {output}")
            if lag_end < 0:
                data = data[:lag_end]
            if lag_start < 0:
                data = data[- lag_start:]
            data = pd.DataFrame(data[:, :-1])
            ioutils.write_hdf5(output, data)

    def sort_spikes(self):
        """
        Run kilosort spike sorting on raw spike data.
        """
        streams = {}

        for rec_num, files in enumerate(self.files):
            data_file = self.find_file(files['spike_data'])
            assert data_file, f"Spike data not found for {files['spike_data']}."

            stream_id = data_file.as_posix()[-12:-4]
            if stream_id not in streams:
                metadata = self.find_file(files['spike_meta'])
                streams[stream_id] = metadata

        for stream_num, stream in enumerate(streams.items()):
            stream_id, metadata = stream
            try:
                recording = se.SpikeGLXRecordingExtractor(self.interim, stream_id=stream_id)
            except ValueError as e:
                raise PixelsError(
                    f"Did the raw data get fully copied to interim? Full error: {e}"
                )

            print("> Running kilosort")
            output = self.processed / f'sorted_stream_{stream_num}'
            concat_rec = si.concatenate_recordings([recording])
            probe = pi.read_spikeglx(metadata.as_posix())
            concat_rec = concat_rec.set_probe(probe)
            ss.run_kilosort3(recording=concat_rec, output_folder=output)

    def assess_noise(self):
        """
        Assess noise in the raw AP data.
        """
        for rec_num, recording in enumerate(self.files):
            print(
                f">>>>> Assessing noise for recording {rec_num + 1} of {len(self.files)}"
            )
            output = self.processed / f'noise_{rec_num}.json'
            if output.exists():
                continue

            data_file = self.find_file(recording['spike_data'])
            num_chans = self.spike_meta[rec_num]['nSavedChans']
            data = ioutils.read_bin(data_file, num_chans)

            # sample some points -- it takes too long to use them all
            length = data.shape[0]
            points = np.random.choice(range(length), length // 60, replace=False)
            sample = data[points, :-1]
            median = np.median(sample, axis=1)
            SDs = []
            for i in range(384):
                SDs.append(np.std(sample[:, i] - median))

            results = dict(
                median=np.median(SDs),
                SDs=SDs,
            )

            with open(output, 'w') as fd:
                json.dump(results, fd)

    def extract_videos(self, force=False):
        """
        Extract behavioural videos from TDMS to avi. By default this will only run if
        the video does not aleady exist. Pass `force=True` to extract the video anyway,
        overwriting the destination file.
        """
        for recording in self.files:
            for v, video in enumerate(recording.get('camera_data', [])):
                path_out = self.interim / video.with_suffix('.avi')
                if not path_out.exists() or force:
                    meta = recording['camera_meta'][v]
                    ioutils.tdms_to_video(
                        self.find_file(video, copy=False),
                        self.find_file(meta),
                        path_out,
                    )

    def configure_motion_tracking(self, project: str) -> None:
        """
        Set up DLC project and add videos to it.
        """
        # bloated so imported when needed
        import deeplabcut  # pylint: disable=import-error

        self.extract_videos()

        working_dir = self.data_dir / "processed" / "DLC"
        matches = list(working_dir.glob(f"{project}*"))
        if matches:
            config = working_dir / matches[0] / "config.yaml"
        else:
            config = None

        videos = []
        for recording in self.files:
            for video in recording.get('camera_data', []):
                if project in video.stem:
                    videos.append(self.interim / video.with_suffix('.avi'))

        if not videos:
            print(self.name, ": No matching videos for project:", project)
            return

        if config:
            deeplabcut.add_new_videos(
                config,
                videos,
                copy_videos=False,
            )
        else:
            print(f"Config not found.")
            reply = input("Create new project? [Y/n]")
            if reply and reply[0].lower() == "n":
                raise PixelsError("A DLC project is needed for motion tracking.")

            deeplabcut.create_new_project(
                project,
                os.environ.get("USER"),
                videos,
                working_directory=working_dir,
                copy_videos=False,
            )
            raise PixelsError("Raising an exception to stop operation. Check new config.")

    def run_motion_tracking(
        self,
        project: str,
        analyse: bool = True,
        align: bool = True,
        create_labelled_video: bool = False,
        extract_outlier_frames: bool = False,
    ):
        """
        Run DeepLabCut motion tracking on behavioural videos.

        Parameters
        ==========

        project : str
            The name of the DLC project.

        create_labelled_video : bool
            Generate a labelled video from existing DLC output.

        """
        # bloated so imported when needed
        import deeplabcut  # pylint: disable=import-error

        working_dir = self.data_dir / "processed" / "DLC"
        matches = list(working_dir.glob(f"{project}*"))
        if not matches:
            raise PixelsError(f"DLC project {profile} not found.")
        config = working_dir / matches[0] / "config.yaml"
        output_dir = self.processed / "DLC"

        videos = []

        for recording in self.files:
            for v, video in enumerate(recording.get('camera_data', [])):
                if project in video.stem:
                    avi = self.interim / video.with_suffix('.avi')
                    if not avi.exists():
                        meta = recording['camera_meta'][v]
                        ioutils.tdms_to_video(
                            self.find_file(video, copy=False),
                            self.find_file(meta),
                            avi,
                        )
                    if not avi.exists():
                        raise PixelsError(f"Path {avi} should exist but doesn't... discuss.")
                    videos.append(avi.as_posix())

        if analyse:
            deeplabcut.analyze_videos(config, videos, destfolder=output_dir)
            deeplabcut.plot_trajectories(config, videos, destfolder=output_dir)
            deeplabcut.filterpredictions(config, videos, destfolder=output_dir)

        if align:
            for video in videos:
                stem = Path(video).stem
                try:
                    result = next(output_dir.glob(f"{stem}*_filtered.h5"))
                except StopIteration:
                    raise PixelsError(f"{self.name}: DLC output not found.")

                coords = pd.read_hdf(result)
                meta_file = None
                rec_num = None

                for rec_num, recording in enumerate(self.files):
                    for meta in recording.get('camera_meta', []):
                        if stem in meta.as_posix():
                            meta_file = meta
                            break
                    if meta_file:
                        break

                assert meta_file and rec_num is not None

                metadata = ioutils.read_tdms(self.find_file(meta_file))
                aligned = self._align_dlc_coords(rec_num, metadata, coords)

                ioutils.write_hdf5(
                    self.processed / f"motion_tracking_{project}_{rec_num}.h5",
                    aligned,
                )

        if extract_outlier_frames:
            deeplabcut.extract_outlier_frames(
                config, videos, destfolder=output_dir, automatic=True,
            )

        if create_labelled_video:
            deeplabcut.create_labeled_video(
                config, videos, destfolder=output_dir, draw_skeleton=True,
            )

    def _align_dlc_coords(self, rec_num, metadata, coords):
        recording = self.files[rec_num]
        behavioural_data = ioutils.read_tdms(self.find_file(recording['behaviour']))

        # ignore any columns that have Nans; these just contain settings
        for col in behavioural_data:
            if behavioural_data[col].isnull().values.any():
                behavioural_data.drop(col, axis=1, inplace=True)

        behav_array = signal.resample(behavioural_data.values, 25000, self.sample_rate)
        behavioural_data.iloc[:len(behav_array), :] = behav_array
        behavioural_data = behavioural_data[:len(behav_array)]

        trigger = signal.binarise(behavioural_data["/'CamFrames'/'0'"]).values
        onsets = np.where((trigger[:-1] == 1) & (trigger[1:] == 0))[0]
        timestamps = ioutils.tdms_parse_timestamps(metadata)
        assert len(onsets) == len(timestamps) == len(coords)

        # The last frame sometimes gets delayed a bit, so ignoring it, are the timestamp
        # diffs fixed?
        assert len(np.unique(np.diff(onsets[:-1]))) == 1

        if self._lag[rec_num] is None:
            self.sync_data(
                rec_num,
                behavioural_data=behavioural_data["/'NpxlSync_Signal'/'0'"].values,
            )
        lag_start, lag_end = self._lag[rec_num]
        no_lag = slice(max(lag_start, 0), -1-max(lag_end, 0))

        # If this fails, we'll have to understand why and if we need to change this
        # logic.
        assert len(coords.columns.get_level_values("scorer").unique()) == 1
        scorer = coords.columns.get_level_values("scorer")[0]
        bodyparts = coords.columns.get_level_values("bodyparts").unique()
        xs = np.arange(0, len(trigger))
        likelihood_threshold = 0.05
        processed = {}
        action_labels = self.get_action_labels()[rec_num]

        for label in bodyparts:
            label_coords = coords[scorer][label]

            # Un-invert y coordinates
            label_coords.y = 480 - label_coords.y

            # Remove unlikely coordinates from fitted B-spline
            bads = label_coords["likelihood"] < likelihood_threshold
            good_onsets = onsets[~bads]
            assert len(good_onsets) == len(onsets) - bads.sum()

            # Interpolate to desired sampling rate

            # B-spline poly fit
            # spl_x = interpolate.splrep(good_onsets, label_coords.x[~ bads])
            # ynew_x = interpolate.splev(xs, spl_x)
            # spl_y = interpolate.splrep(good_onsets, label_coords.y[~ bads])
            # ynew_y = interpolate.splev(xs, spl_y)

            # Linear fit
            ynew_x = interpolate.interp1d(
                good_onsets, label_coords.x[~ bads], fill_value="extrapolate",
            )(xs)
            ynew_y = interpolate.interp1d(
                good_onsets, label_coords.y[~ bads], fill_value="extrapolate",
            )(xs)

            data = np.array([ynew_x[no_lag], ynew_y[no_lag]]).T
            assert action_labels.shape == data.shape
            processed[label] = pd.DataFrame(data, columns=["x", "y"])

        df = pd.concat(processed, axis=1)
        return pd.concat({scorer: df}, axis=1, names=coords.columns.names)

    def draw_motion_index_rois(self, num_rois=1):
        """
        Draw motion index ROIs using EasyROI. If ROIs already exist, skip.

        Parameters
        ----------
        num_rois : int
            The number of ROIs to draw interactively. Default: 1

        """
        # Only needed for this method
        import cv2
        import EasyROI

        roi_helper = EasyROI.EasyROI(verbose=False)

        for i, recording in enumerate(self.files):
            if 'camera_data' in recording:
                roi_file = self.processed / f"motion_index_ROIs_{i}.pickle"
                if roi_file.exists():
                    continue

                # Load frame from video
                video = self.interim / recording['camera_data'].with_suffix('.avi')
                if not video.exists():
                    raise PixelsError(self.name + ": AVI video not found, run `extract_videos`")

                duration = ioutils.get_video_dimensions(video.as_posix())[2]
                frame = ioutils.load_video_frame(video.as_posix(), duration // 4)

                # Interactively draw ROI
                roi = roi_helper.draw_polygon(frame, num_rois)
                cv2.destroyAllWindows()  # Needed otherwise EasyROI errors

                # Save a copy of the frame with ROIs to PNG file
                png = self.processed / f'motion_index_ROIs_{i}.png'
                copy = EasyROI.visualize_polygon(frame, roi, color=(255, 0, 0))
                plt.imsave(png, copy, cmap='gray')

                # Save ROI to file
                with roi_file.open('wb') as fd:
                    pickle.dump(roi['roi'], fd)

    def process_motion_index(self):
        """
        Extract motion indexes from videos using already drawn ROIs.
        """

        ses_rois = {}

        # First collect all ROIs to catch errors early
        for i, recording in enumerate(self.files):
            if 'camera_data' in recording:
                roi_file = self.processed / f"motion_index_ROIs_{i}.pickle"
                if not roi_file.exists():
                    raise PixelsError(self.name + ": ROIs not drawn for motion index.")

                # Also check videos are available
                video = self.interim / recording['camera_data'].with_suffix('.avi')
                if not video.exists():
                    raise PixelsError(self.name + ": AVI video not found in interim folder.")

                with roi_file.open('rb') as fd:
                    ses_rois[i] = pickle.load(fd)

        # Then do the extraction
        for rec_num, recording in enumerate(self.files):
            if 'camera_data' in recording:

                # Get MIs
                raise NotImplementedError
                video = self.interim / recording['camera_data'].with_suffix('.avi')
                rec_rois = ses_rois[rec_num]
                rec_mi = signal.motion_index(video.as_posix(), rec_rois, self.sample_rate)

                # TODO: Use timestamps for real alignment
                # Get initial timestamp of behavioural data
                #behavioural_data = ioutils.read_tdms(self.find_file(recording['behaviour']))
                #for key in behavioural_data.keys():
                #    if key.startswith("/'t0'/"):
                #        t0 = behavioural_data[key][0]
                #        break
                #metadata = ioutils.read_tdms(self.find_file(recording['camera_meta']))
                #timestamps = ioutils.tdms_parse_timestamps(metadata)

                np.save(self.processed / f'motion_index_{i}.npy', rec_mi)

    def add_motion_index_action_label(
        self, label: int, event: int, roi: int, value: int
    ) -> None:
        """
        Add motion index onset times to the action labels as an event that can be
        aligned to.

        Paramters
        ---------

        label : int
            The action following which the onset is expected to occur. A baseline level
            of motion is determined from 5s preceding these actions, and the onset time
            is searched for from this period up to event.

        event : int
            The event before which the onset is expected to occur. MI onsets are looked
            for between the times of action (above) and this event.

        roi : int
            The index in the motion index vectors of the ROI to use to found onsets.

        value : int
            The value to use to represent the onset time event in the action labels i.e.
            a value that is (or could be) part of the behaviour's `Events` enum for
            representing movement onsets.

        """
        action_labels = self.get_action_labels()
        motion_indexes = self.get_motion_index_data()

        scan_duration = self.sample_rate * 10
        half = scan_duration // 2

        # We take 200 ms before the action begins as a short baseline period for each
        # trial. The smallest standard deviation of all SDs of these baseline periods is
        # used as a threshold to identify "clean" trials (`clean_threshold` below).
        # Non-clean trials are trials where TODO
        short_pre = int(0.2 * self.sample_rate)

        for rec_num, recording in enumerate(self.files):
            # Only recs with camera_data will have motion indexes
            if 'camera_data' in recording:

                # Get our numbers
                assert motion_indexes[rec_num] is not None
                num_rois = motion_indexes[rec_num].shape[1]
                if num_rois - 1 < roi:
                    raise PixelsError("ROI index is too high, there are only {num_rois} ROIs")
                motion_index = motion_indexes[rec_num][:, roi]

                actions = action_labels[rec_num][:, 0]
                events = action_labels[rec_num][:, 1]
                trial_starts = np.where(np.bitwise_and(actions, label))[0]

                # Cut out our trials
                trials = []
                pre_cue_SDs = []

                for start in trial_starts:
                    event_latency = np.where(np.bitwise_and(events[start:start + scan_duration], event))[0]
                    if len(event_latency) == 0:
                        raise PixelsError('Action labels probably miscalculated')
                    trial = motion_index[start:start + event_latency[0]]
                    trials.append(trial)

                    pre_cue_SDs.append(np.std(motion_index[start - short_pre:start]))

                clean_threshold = min(pre_cue_SDs)

                onsets = []

                for t in trials:
                    movements = np.where(t > clean_threshold * 10)[0]
                    if len(movements) == 0:
                        raise PixelsError("Thresholding for detecting MI onset is inadequate")
                    onsets.append(movements[0])

                assert 0


    @abstractmethod
    def _extract_action_labels(self, behavioural_data):
        """
        This method must be overriden with the derivation of action labels from
        behavioural data specific to the behavioural task.

        Parameters
        ----------
        behavioural_data : pandas.DataFrame
            A dataframe containing the behavioural DAQ data.

        Returns
        -------
        action_labels : 1D numpy.ndarray
            An array of actions of equal length to the behavioural_data.

        """

    def _get_processed_data(self, attr, key):
        """
        Used by the following get_X methods to load processed data.

        Parameters
        ----------
        attr : str
            The self attribute that stores the data.

        key : str
            The key for the files in each recording of self.files that contain this
            data.

        """
        saved = getattr(self, attr)

        if saved[0] is None:
            for rec_num, recording in enumerate(self.files):
                if key in recording:
                    file_path = self.processed / recording[key]
                    if file_path.exists():
                        if file_path.suffix == '.npy':
                            saved[rec_num] = np.load(file_path)
                        elif file_path.suffix == '.h5':
                            saved[rec_num] = ioutils.read_hdf5(file_path)
                    else:
                        msg = f"Could not find {attr[1:]} for recording {rec_num}."
                        msg += f"\nFile should be at: {file_path}"
                        raise PixelsError(msg)
        return saved

    def get_action_labels(self):
        """
        Returns the action labels, either from self._action_labels if they have been
        loaded already, or from file.
        """
        return self._get_processed_data("_action_labels", "action_labels")

    def get_behavioural_data(self):
        """
        Returns the downsampled behaviour channels.
        """
        return self._get_processed_data("_behavioural_data", "behaviour_processed")

    def get_motion_index_data(self):
        """
        Returns the motion indexes, either from self._motion_index if they have been
        loaded already, or from file.
        """
        return self._get_processed_data("_motion_index", "motion_index")

    def get_motion_tracking_data(self, dlc_project: str):
        """
        Returns the DLC coordinates from self._motion_tracking if they have been loaded
        already, or from file.
        """
        key = f"motion_tracking_{dlc_project}"
        attr = f"_{key}"
        if hasattr(self, attr):
            return getattr(self, attr)

        setattr(self, attr, [None] * len(self.files))

        for rec_num, recording in enumerate(self.files):
            if "camera_data" in recording:
                recording[key] = f"{key}_{rec_num}.h5"

        return self._get_processed_data(attr, key)

    def get_spike_data(self):
        """
        Returns the processed and downsampled spike data.
        """
        return self._get_processed_data("_spike_data", "spike_processed")

    def get_lfp_data(self):
        """
        Returns the processed and downsampled LFP data.
        """
        return self._get_processed_data("_lfp_data", "lfp_processed")

    def _get_spike_times(self):
        """
        Returns the sorted spike times.
        """
        saved = self._spike_times_data
        if saved[0] is None:
            times = self.processed / f'sorted_stream_0' / 'spike_times.npy'
            clust = self.processed / f'sorted_stream_0' / 'spike_clusters.npy'

            try:
                times = np.load(times)
                clust = np.load(clust)
            except FileNotFoundError:
                msg = ": Can't load spike times that haven't been extracted!"
                raise PixelsError(self.name + msg)

            times = np.squeeze(times)
            clust = np.squeeze(clust)
            by_clust = {}

            for c in np.unique(clust):
                by_clust[c] = pd.Series(times[clust == c]).drop_duplicates()
            saved[0]  = pd.concat(by_clust, axis=1, names=['unit'])
        return saved[0]

    def _get_aligned_spike_times(
        self, label, event, duration, rate=False, sigma=None, units=None
    ):
        """
        Returns spike times for each unit within a given time window around an event.
        align_trials delegates to this function, and should be used for getting aligned
        data in scripts.
        """
        action_labels = self.get_action_labels()

        if units is None:
            units = self.select_units()

        spikes = self._get_spike_times()[units]
        # Convert to ms (self.sample_rate)
        spikes /= int(self.spike_meta[0]['imSampRate']) / self.sample_rate

        if rate:
            # pad ends with 1 second extra to remove edge effects from convolution
            duration += 2

        scan_duration = self.sample_rate * 8
        half = int((self.sample_rate * duration) / 2)
        cursor = 0  # In sample points
        i = -1
        rec_trials = {}

        for rec_num in range(len(self.files)):
            actions = action_labels[rec_num][:, 0]
            events = action_labels[rec_num][:, 1]
            trial_starts = np.where(np.bitwise_and(actions, label))[0]

            # Account for multiple raw data files
            meta = self.spike_meta[rec_num]
            samples = int(meta["fileSizeBytes"]) / int(meta["nSavedChans"]) / 2
            assert samples.is_integer()
            milliseconds = samples / 30
            cursor_duration = cursor / 30
            rec_spikes = spikes[
                (cursor_duration <= spikes) & (spikes < (cursor_duration + milliseconds))
            ] - cursor_duration
            cursor += samples

            # Account for lag, in case the ephys recording was started before the
            # behaviour
            lag_start, _ = self._lag[rec_num]
            if lag_start < 0:
                rec_spikes = rec_spikes + lag_start

            for i, start in enumerate(trial_starts, start=i + 1):
                centre = np.where(np.bitwise_and(events[start:start + scan_duration], event))[0]
                if len(centre) == 0:
                    raise PixelsError('Action labels probably miscalculated')
                centre = start + centre[0]

                trial = rec_spikes[centre - half < rec_spikes]
                trial = trial[trial <= centre + half]
                trial = trial - centre
                tdf = []

                for unit in trial:
                    u_times = trial[unit].values
                    u_times = u_times[~np.isnan(u_times)]
                    u_times = np.unique(u_times)  # remove double-counted spikes
                    udf = pd.DataFrame({int(unit): u_times})
                    tdf.append(udf)

                assert len(tdf) == len(units)
                tdfc = pd.concat(tdf, axis=1)
                if rate:
                    tdfc = signal.convolve(tdfc, duration * 1000, sigma)
                rec_trials[i] = tdfc

        trials = pd.concat(rec_trials, axis=1, names=["trial", "unit"])
        trials = trials.reorder_levels(["unit", "trial"], axis=1)
        trials = trials.sort_index(level=0, axis=1)

        if rate:
            # Set index to seconds and remove the padding 1 sec at each end
            points = trials.shape[0]
            start = (- duration / 2) + (duration / points)
            # Having trouble with float values
            #timepoints = np.linspace(start, duration / 2, points, dtype=np.float64)
            timepoints = list(range(int(start * 1000), int(duration * 1000 / 2) + 1))
            trials['time'] = pd.Series(timepoints, index=trials.index) / 1000
            trials = trials.set_index('time')
            trials = trials.iloc[self.sample_rate : - self.sample_rate]

        return trials

    def select_units(
        self, group='good', min_depth=0, max_depth=None, min_spike_width=None,
        max_spike_width=None, uncurated=False, name=None
    ):
        """
        Select units based on specified criteria. The output of this can be passed to
        some other methods to apply those methods only to these units.

        Parameters
        ----------
        group : str, optional
            The group to which the units that are wanted are part of. One of: 'group',
            'mua', 'noise' or None. Default is 'good'.

        min_depth : int, optional
            (Only used when getting spike data). The minimum depth that units must be at
            to be included. Default is 0 i.e. in the brain.

        max_depth : int, optional
            (Only used when getting spike data). The maximum depth that units must be at
            to be included. Default is None i.e.  no maximum.

        min_spike_width : int, optional
            (Only used when getting spike data). The minimum median spike width that
            units must have to be included. Default is None i.e. no minimum.

        max_spike_width : int, optional
            (Only used when getting spike data). The maximum median spike width that
            units must have to be included. Default is None i.e. no maximum.

        uncurated : bool, optional
            Use uncurated units. Default: False.

        name : str, optional
            Give this selection of units a name. This allows the list of units to be
            represented as a string, which enables caching. Future calls to cacheable
            methods with a selection of units of the same name will read cached data
            from disk. It is up to the user to ensure that the actual selection of units
            is the same between uses of the same name.

        """
        cluster_info = self.get_cluster_info()
        selected_units = SelectedUnits()
        if name is not None:
            selected_units.name = name

        if min_depth is not None or max_depth is not None:
            probe_depth = self.get_probe_depth()[0]

        if min_spike_width == 0:
            min_spike_width = None
        if min_spike_width is not None or max_spike_width is not None:
            widths = self.get_spike_widths()
        else:
            widths = None

        rec_num = 0

        id_key = 'id' if 'id' in cluster_info else 'cluster_id'
        grouping = 'KSLabel' if uncurated else 'group'

        for unit in cluster_info[id_key]:
            unit_info = cluster_info.loc[cluster_info[id_key] == unit].iloc[0].to_dict()

            # we only want units that are in the specified group
            if not group or unit_info[grouping] == group:

                # and that are within the specified depth range
                if min_depth is not None:
                    if probe_depth - unit_info['depth'] < min_depth:
                        continue
                if max_depth is not None:
                    if probe_depth - unit_info['depth'] > max_depth:
                        continue

                # and that have the specified median spike widths
                if widths is not None:
                    width = widths[widths['unit'] == unit]['median_ms']
                    assert len(width.values) == 1
                    if min_spike_width is not None:
                        if width.values[0] < min_spike_width:
                            continue
                    if max_spike_width is not None:
                        if width.values[0] > max_spike_width:
                            continue

                selected_units.append(unit)

        return selected_units

    def _get_neuro_raw(self, kind):
        raw = []
        meta = getattr(self, f"{kind}_meta")
        for rec_num, recording in enumerate(self.files):
            data_file = self.find_file(recording[f'{kind}_data'])
            orig_rate = int(meta[rec_num]['imSampRate'])
            num_chans = int(meta[rec_num]['nSavedChans'])
            factor = orig_rate / self.sample_rate

            data = ioutils.read_bin(data_file, num_chans)

            if self._lag[rec_num] is None:
                self.sync_data(rec_num, sync_channel=data[:, -1])
            lag_start, lag_end = self._lag[rec_num]

            lag_start = int(lag_start * factor)
            lag_end = int(lag_end * factor)
            if lag_end < 0:
                data = data[:lag_end]
            if lag_start < 0:
                data = data[- lag_start:]
            raw.append(pd.DataFrame(data[:, :-1]))

        return raw, orig_rate

    def get_spike_data_raw(self):
        """
        Returns the raw spike data with lag region removed.
        """
        return self._get_neuro_raw('spike')

    def get_lfp_data_raw(self):
        """
        Returns the raw spike data with lag region removed.
        """
        return self._get_neuro_raw('lfp')

    @_cacheable
    def align_trials(
        self, label, event, data='spike_times', raw=False, duration=1, sigma=None,
        units=None, dlc_project=None,
    ):
        """
        Get trials aligned to an event. This finds all instances of label in the action
        labels - these are the start times of the trials. Then this finds the first
        instance of event on or after these start times of each trial. Then it cuts out
        a period around each of these events covering all units, rearranges this data
        into a MultiIndex DataFrame and returns it.

        Parameters
        ----------
        label : int
            An action label value to specify which trial types are desired.

        event : int
            An event type value to specify which event to align the trials to.

        data : str, optional
            The data type to align.

        raw : bool, optional
            Whether to get raw, unprocessed data instead of processed and downsampled
            data. Defaults to False.

        duration : int/float, optional
            The length of time in seconds desired in the output. Default is 1 second.

        sigma : int, optional
            Time in milliseconds of sigma of gaussian kernel to use when aligning firing
            rates. Default is 50 ms.

        units : list of lists of ints, optional
            The output from self.select_units, used to only apply this method to a
            selection of units.

        dlc_project : str | None
            The DLC project from which to get motion tracking coordinates, if aligning
            to motion_tracking data.

        """
        data = data.lower()

        data_options = [
            'behavioural',  # Channels from behaviour TDMS file
            #'spike',        # Raw/downsampled channels from probe (AP)
            'spike_times',  # List of spike times per unit
            'spike_rate',   # Spike rate signals from convolved spike times
            #'lfp',          # Raw/downsampled channels from probe (LFP)
            #'motion_index', # Motion indexes per ROI from the video
            'motion_tracking', # Motion tracking coordinates from DLC
        ]
        if data not in data_options:
            raise PixelsError(f"align_trials: 'data' should be one of: {data_options}")

        if data in ("spike_times", "spike_rate"):
            print(f"Aligning {data} to trials.")
            # we let a dedicated function handle aligning spike times
            return self._get_aligned_spike_times(
                label, event, duration, rate=data == "spike_rate", sigma=sigma,
                units=units
            )

        if data == "motion_tracking" and not dlc_project:
            raise PixelsError("When aligning to 'motion_tracking', dlc_project is needed.")

        action_labels = self.get_action_labels()

        if raw:
            print(f"Aligning raw {data} data to trials.")
            getter = getattr(self, f"get_{data}_data_raw", None)
            if not getter:
                raise PixelsError(f"align_trials: {data} doesn't have a 'raw' option.")
            values, sample_rate = getter()

        else:
            print(f"Aligning {data} data to trials.")
            if dlc_project:
                values = self.get_motion_tracking_data(dlc_project)
            else:
                values = getattr(self, f"get_{data}_data")()
            sample_rate = self.sample_rate

        if not values or values[0] is None:
            raise PixelsError(f"align_trials: Could not get {data} data.")

        trials = []
        # The logic here is that the action labels will always have a sample rate of
        # self.sample_rate, whereas our data here may differ. 'duration' is used to scan
        # the action labels, so always give it 5 seconds to scan, then 'half' is used to
        # index data.
        scan_duration = self.sample_rate * 10
        half = (sample_rate * duration) // 2
        if isinstance(half, float):
            assert half.is_integer()  # In case duration is a float < 1
            half = int(half)

        for rec_num in range(len(self.files)):
            if values[rec_num] is None:
                # This means that each recording is using the same piece of data for
                # this data type, e.g. all recordings using motion indexes from a single
                # video
                break

            actions = action_labels[rec_num][:, 0]
            events = action_labels[rec_num][:, 1]
            trial_starts = np.where(np.bitwise_and(actions, label))[0]

            for start in trial_starts:
                centre = np.where(np.bitwise_and(events[start:start + scan_duration], event))[0]
                if len(centre) == 0:
                    raise PixelsError('Action labels probably miscalculated')
                centre = start + centre[0]
                centre = int(centre * sample_rate / self.sample_rate)
                trial = values[rec_num][centre - half + 1:centre + half + 1]

                if isinstance(trial, np.ndarray):
                    trial = pd.DataFrame(trial)
                trials.append(trial.reset_index(drop=True))

        if not trials:
            raise PixelsError("Seems the action-event combo you asked for doesn't occur")

        if data == "motion_tracking":
            ses_trials = pd.concat(
                trials,
                axis=1,
                copy=False,
                keys=range(len(trials)),
                names=["trial"] + trials[0].columns.names
            )
        else:
            ses_trials = pd.concat(
                trials, axis=1, copy=False, keys=range(len(trials)), names=["trial", "unit"]
            )
            ses_trials = ses_trials.sort_index(level=1, axis=1)
            ses_trials = ses_trials.reorder_levels(["unit", "trial"], axis=1)

        points = ses_trials.shape[0]
        start = (- duration / 2) + (duration / points)
        timepoints = np.linspace(start, duration / 2, points)
        ses_trials['time'] = pd.Series(timepoints, index=ses_trials.index)
        ses_trials = ses_trials.set_index('time')
        return ses_trials

    def get_cluster_info(self):
        if self._cluster_info is None:
            info_file = self.processed / f'sorted_stream_0' / 'cluster_info.tsv'
            try:
                info = pd.read_csv(info_file, sep='\t')
            except FileNotFoundError:
                msg = ": Can't load cluster info. Did you sort this session yet?"
                raise PixelsError(self.name + msg)
            self._cluster_info = info
        return self._cluster_info

    @_cacheable
    def get_spike_widths(self, units=None):
        from phylib.io.model import load_model
        from phylib.utils.color import selected_cluster_color

        print("Calculating spike widths")
        waveforms = self.get_spike_waveforms(units=units)
        widths = []

        for unit in waveforms.columns.get_level_values('unit').unique():
            u_widths = []
            u_spikes = waveforms[unit]
            for s in u_spikes:
                spike = u_spikes[s]
                trough = np.where(spike.values == min(spike))[0][0]
                after = spike.values[trough:]
                width = np.where(after == max(after))[0][0]
                u_widths.append(width)
            widths.append((unit, np.median(u_widths)))

        df = pd.DataFrame(widths, columns=["unit", "median_ms"])
        # convert to ms from sample points
        orig_rate = int(self.spike_meta[0]['imSampRate'])
        df['median_ms'] = 1000 * df['median_ms'] / orig_rate
        return df

    @_cacheable
    def get_spike_waveforms(self, units=None):
        from phylib.io.model import load_model
        from phylib.utils.color import selected_cluster_color

        if units is None:
            units = self.select_units()

        paramspy = self.processed / 'sorted_stream_0' / 'params.py'
        if not paramspy.exists():
            raise PixelsError(f"{self.name}: params.py not found")
        model = load_model(paramspy)
        rec_forms = {}

        for unit in units:
            # get the waveforms from only the best channel
            spike_ids = model.get_cluster_spikes(unit)
            best_chan = model.get_cluster_channels(unit)[0]
            u_waveforms = model.get_waveforms(spike_ids, [best_chan])
            if u_waveforms is None:
                raise PixelsError(f"{self.name}: unit {unit} - waveforms not read")
            rec_forms[unit] = pd.DataFrame(np.squeeze(u_waveforms).T)

        assert rec_forms

        df = pd.concat(
            rec_forms,
            axis=1,
            names=['unit', 'spike']
        )
        # convert indexes to ms
        rate = 1000 / int(self.spike_meta[0]['imSampRate'])
        df.index = df.index * rate
        return df

    @_cacheable
    def get_aligned_spike_rate_CI(
        self, label, event,
        start=0.000, step=0.100, end=1.000,
        bl_label=None, bl_event=None, bl_start=None, bl_end=0.000,
        ss=20, CI=95, bs=10000,
        units=None, sigma=None,
    ):
        """
        Get the confidence intervals of the mean firing rates within a window aligned to
        a specified action label and event. An example would be to align firing rates to
        cue and take a 200 ms pre-cue window, mean the windows, and compute bootstrapped
        confidence intervals for those values. Optionally baseline the windowed values
        using values from another window.

        Parameters
        ----------
        label : ActionLabel int
            Action to align to, from a specific behaviour's ActionLabels class.

        event : Event int
            Event to align to, from a specific behaviour's Events class.

        start : float, optional
            Time in milliseconds relative to event for the left edge of the bins.

        step : float, optional
            Time in milliseconds for the bin size.

        end : float, optional
            Time in milliseconds relative to event for the right edge of the bins.

        bl_label, bl_event, bl_start, bl_end : as above, all optional
            Equivalent to the above parameters but for baselining data. By default no
            baselining is performed.

        ss : int, optional.
            Sample size of bootstrapped samples.

        CI : int/float, optional
            Confidence interval size. Default is 95: this returns the 2.5%, 50% and
            97.5% bootstrap sample percentiles.

        bs : int, optional
            Number of bootstrapped samples. Default is 10000.

        units : list of lists of ints, optional
            The output from self.select_units, used to only apply this method to a
            selection of units.

        sigma : int, optional
            Time in milliseconds of sigma of gaussian kernel to use. Default is 50 ms.

        """
        # Set timepoints to 3 decimal places (ms) to make things easier
        start = round(start, 3)
        step = round(step, 3)
        end = round(end, 3)
        if bl_start is not None:
            bl_start = round(bl_start, 3)
        bl_end = round(bl_end, 3)
        # Get firing rates
        duration = round(2 * max(abs(start), abs(end)) + 0.002, 3)
        responses = self.align_trials(
            label, event, 'spike_rate', duration=duration, sigma=sigma, units=units
        )
        series = responses.index.values
        assert series[0] <= start < end <= series[-1] + 0.001

        bins = round((end - start) / step, 10)  # Round in case of floating point recursive
        assert bins.is_integer()
        bin_responses = []

        for i in range(int(bins)):
            bin_start = start + i * step
            bin_end = bin_start + step
            bin_responses.append(responses.loc[bin_start : bin_end].mean())

        averages = pd.concat(bin_responses, axis=1)

        # Optionally baseline the firing rates
        if bl_start is not None and bl_end is not None:
            duration = 2 * max(abs(bl_start), abs(bl_end))

            if bl_label is None:
                bl_label = label
            if bl_event is None:
                bl_event = event

            baselines = self.align_trials(
                bl_label, bl_event, 'spike_rate', duration=duration, sigma=sigma, units=units
            )
            series = baselines.index.values
            assert series[0] <= (bl_start + 0.001) < bl_end <= series[-1]
            baseline = baselines.loc[bl_start : bl_end].mean()
            for i in averages:
                averages[i] = averages[i] - baseline

        # Calculate the confidence intervals for each unit and bin
        lower = (100 - CI) / 2
        upper = 100 - lower
        percentiles = [lower, 50, upper]
        cis = []

        rec_cis = []
        for unit in units:
            u_resps = averages.loc[unit]
            samples = np.array([
                [np.random.choice(u_resps[i], size=ss) for b in range(bs)]
                for i in u_resps
            ])
            medians = np.median(samples, axis=2)
            results = np.percentile(medians, percentiles, axis=1)
            rec_cis.append(pd.DataFrame(results))

        if rec_cis:
            rec_df = pd.concat(rec_cis, axis=1, keys=units)
        else:
            rec_df = pd.DataFrame(
                {rec_num: np.nan},
                index=range(3),
                columns=pd.MultiIndex.from_arrays([[-1], [-1]], names=['unit', 'bin'])
            )
        cis.append(rec_df)

        df = pd.concat(cis, axis=1, names=['unit', 'bin'])
        df.set_index(pd.Index(percentiles, name="percentile"), inplace=True)
        return df
