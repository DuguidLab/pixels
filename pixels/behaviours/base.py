"""
This module provides a base class for experimental sessions that must be used as the
base for defining behaviour-specific processing.
"""


import functools
import json
import os
import tarfile
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from shutil import copyfile

import numpy as np
import pandas as pd
import scipy.stats
import spikeextractors as se
import spikesorters as ss
import spiketoolkit as st

from pixels import ioutils
from pixels import signal
from pixels.error import PixelsError


BEHAVIOUR_HZ = 25000

np.random.seed(BEHAVIOUR_HZ)


def _cacheable(method):
    def func(*args, **kwargs):
        as_list = list(args) + list(kwargs.values())
        self = as_list.pop(0)
        if not self._use_cache:
            return method(*args, **kwargs)

        as_list.insert(0, method.__name__)
        output = self.interim / 'cache' / ('_'.join(str(i) for i in as_list) + '.h5')
        if output.exists():
            df = ioutils.read_hdf5(output)
        else:
            df = method(*args, **kwargs)
            output.parent.mkdir(parents=True, exist_ok=True)
            ioutils.write_hdf5(output, df)
        return df
    return func


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
        self._load_lag()

    def set_cache(self, on):
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
        if depth_file.exists():
            with depth_file.open() as fd:
                return float(fd.read())
        raise PixelsError(self.name + ": Can't load probe depth: please add it in um to processed/depth.txt")

    def find_file(self, name):
        """
        Finds the specified file, looking for it in the processed folder, interim
        folder, and then raw folder in that order. If the the file is only found in the
        raw folder, it is copied to the interim folder and uncompressed if required.

        Parameters
        ----------
        name : str or pathlib.Path
            The basename of the file to be looked for.

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
            print(f"    {self.name}: Copying {name} to interim")
            copyfile(raw, interim)
            return interim

        tar = raw.with_name(raw.name + '.tar.gz')
        if tar.exists():
            print(f"    {self.name}: Extracting {tar.name} to interim")
            with tarfile.open(tar) as open_tar:
                open_tar.extractall(path=self.interim)
            return interim

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
            self._action_labels[rec_num] = self._extract_action_labels(behavioural_data)
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

            data_file = self.find_file(recording['spike_data'])
            orig_rate = self.spike_meta[rec_num]['imSampRate']
            num_chans = self.spike_meta[rec_num]['nSavedChans']

            print("> Mapping spike data")
            data = ioutils.read_bin(data_file, num_chans)

            #print("> Performing median subtraction across rows")  # TODO: fix
            #data = signal.median_subtraction(data, axis=0)
            #print("> Performing median subtraction across columns")
            #data = signal.median_subtraction(data, axis=1)

            print(f"> Downsampling to {self.sample_rate} Hz")
            data = signal.resample(data, orig_rate, self.sample_rate)

            if self._lag[rec_num] is None:
                self.sync_data(rec_num, sync_channel=data[:, -1])
            lag_start, lag_end = self._lag[rec_num]

            output = self.processed / recording['spike_processed']
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
        for rec_num, recording in enumerate(self.files):
            print(
                f">>>>> Spike sorting recording {rec_num + 1} of {len(self.files)}"
            )

            output = self.processed / f'sorted_{rec_num}'
            data_file = self.find_file(recording['spike_data'])
            try:
                recording = se.SpikeGLXRecordingExtractor(file_path=data_file)
            except ValueError as e:
                raise PixelsError(
                    f"Did the raw data get fully copied to interim? Full error: {e}"
                )

            print(f"> Running kilosort")
            ss.run_kilosort3(recording=recording, output_folder=output)

    def extract_videos(self):
        """
        Extract behavioural videos from TDMS to avi.
        """
        for recording in self.files:
            path = self.find_file(recording['camera_data'])
            path_avi = path.with_suffix('.avi')
            if path_avi.exists():
                continue

            df = ioutils.read_tdms(path)
            meta = ioutils.read_tdms(self.find_file(recording['camera_meta']))
            actual_heights = meta["/'keys'/'IMAQdxActualHeight'"]
            if "/'frames'/'ind_skipped'" in meta:
                skipped = meta["/'frames'/'ind_skipped'"].dropna().size
            else:
                skipped = 0
            height = int(actual_heights.max())
            remainder = skipped - actual_heights[actual_heights != height].size
            duration = actual_heights.size - remainder
            width = int(df.size / (duration * height))
            if width != 640:
                raise PixelsError("Width calculation must be incorrect, discuss.")

            video = df.values.reshape((duration, height, int(width)))
            ioutils.save_ndarray_as_avi(video, path_avi, 50)

    def process_motion_tracking(self, config, create_labelled_video=False):
        """
        Run DeepLabCut motion tracking on behavioural videos.
        """
        # bloated so imported when needed
        import deeplabcut  # pylint: disable=import-error

        self.extract_videos()

        config = Path(config).expanduser()
        if not config.exists():
            raise PixelsError(f"Config at {config} not found.")

        for recording in self.files:
            video = self.find_file(recording['camera_data']).with_suffix('.avi')
            if not video.exists():
                raise PixelsError(f"Path {video} should exist but doesn't... discuss.")

            deeplabcut.analyze_videos(config, [video])
            deeplabcut.plot_trajectories(config, [video])
            if create_labelled_video:
                deeplabcut.create_labeled_video(config, [video])

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
            for rec_num, recording in enumerate(self.files):
                times = self.processed / f'sorted_{rec_num}' / 'spike_times.npy'
                clust = self.processed / f'sorted_{rec_num}' / 'spike_clusters.npy'

                try:
                    times = np.load(times)
                    clust = np.load(clust)
                except FileNotFoundError:
                    msg = ": Can't load spike times that haven't been extracted!"
                    raise PixelsError(self.name + msg)

                times = np.squeeze(times)
                clust = np.squeeze(clust)
                by_clust = {}

                lag_start, _ = self._lag[rec_num]
                if lag_start < 0:
                    times = times + lag_start

                for c in np.unique(clust):
                    by_clust[c] = pd.Series(times[clust == c]).drop_duplicates()
                saved[rec_num]  = pd.concat(by_clust, axis=1, names=['unit'])
        return saved

    def _get_aligned_spike_times(
        self, label, event, duration, group='good', min_depth=0, max_depth=None,
        min_spike_width=None, max_spike_width=None, rate=False, sigma=None,
        uncurated=False
    ):
        """
        Returns spike times for each unit within a given time window around an event.
        align_trials delegates to this function, and should be used for getting aligned
        data in scripts.
        """
        action_labels = self.get_action_labels()
        selected_units = self.filter_units(
            group, min_depth, max_depth, min_spike_width, max_spike_width, uncurated
        )
        spikes = self._get_spike_times()

        if rate:
            # pad ends with 1 second extra to remove edge effects from convolution
            duration += 2

        scan_duration = self.sample_rate * 5
        half = int((self.sample_rate * duration) / 2)
        trials = []

        for rec_num in range(len(self.files)):
            actions = action_labels[rec_num][:, 0]
            events = action_labels[rec_num][:, 1]
            trial_starts = np.where((actions == label))[0]

            rec_spikes = spikes[rec_num]
            rec_spikes = rec_spikes[selected_units[rec_num]]
            rec_trials = []
            f = int(self.spike_meta[rec_num]['imSampRate']) / self.sample_rate
            rec_spikes = rec_spikes / f

            for i, start in enumerate(trial_starts):
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
                if tdf:
                    tdfc = pd.concat(tdf, axis=1)
                    if rate:
                        tdfc = signal.convolve(tdfc, duration * 1000, sigma)
                    rec_trials.append(tdfc)

            rec_df = pd.concat(rec_trials, axis=1, keys=range(len(rec_trials)))
            trials.append(rec_df)

        trials = pd.concat(trials, axis=1, keys=range(len(trials)), names=["rec_num", "trial", "unit"])
        trials = trials.reorder_levels(["rec_num", "unit", "trial"], axis=1)
        trials = trials.sort_index(level=0, axis=1)

        if rate:
            # Set index to seconds and remove the padding 1 sec at each end
            points = trials.shape[0]
            start = (- duration / 2) + (duration / points)
            timepoints = np.linspace(start, duration / 2, points)
            trials['time'] = pd.Series(timepoints, index=trials.index)
            trials = trials.set_index('time')
            trials = trials.iloc[self.sample_rate : - self.sample_rate]

        return trials

    def filter_units(
        self, group='good', min_depth=0, max_depth=None, min_spike_width=None,
        max_spike_width=None, uncurated=False
    ):
        """
        Select units based on specified criteria.
        """
        cluster_info = self.get_cluster_info()
        selected_units = []

        if min_depth is not None or max_depth is not None:
            probe_depth = self.get_probe_depth()

        if min_spike_width == 0:
            min_spike_width = None
        if min_spike_width is not None or max_spike_width is not None:
            widths = self.get_spike_widths(
                group=group, min_depth=min_depth, max_depth=max_depth
            )
        else:
            widths = None

        grouping = 'KSLabel' if uncurated else 'good'

        for rec_num in range(len(self.files)):
            rec_info = cluster_info[rec_num]
            rec_units = []

            if widths is not None:
                rec_widths = widths[widths['rec_num'] == rec_num]

            for unit in rec_info['id']:
                unit_info = rec_info.loc[rec_info['id'] == unit].iloc[0].to_dict()

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
                        width = rec_widths[rec_widths['unit'] == unit]['median_ms']
                        assert len(width.values) == 1
                        if min_spike_width is not None:
                            if width.values[0] < min_spike_width:
                                continue
                        if max_spike_width is not None:
                            if width.values[0] > max_spike_width:
                                continue

                    rec_units.append(unit)
            selected_units.append(rec_units)
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
        self, label, event, data='spike_times', raw=False, duration=1, min_depth=0,
        max_depth=None, min_spike_width=None, max_spike_width=None, sigma=None,
        uncurated=False
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
            One of 'spike_times' (default), 'behavioural', 'spike', or 'lfp'.

        raw : bool, optional
            Whether to get raw, unprocessed data instead of processed and downsampled
            data. Defaults to False.

        duration : int/float, optional
            The length of time in seconds desired in the output. Default is 1 second.

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

        """
        data = data.lower()

        data_options = ['behavioural', 'spike', 'spike_times', 'spike_rate', 'lfp']
        if data not in data_options:
            raise PixelsError(f"align_trials: 'data' should be one of: {data_options}")

        if data in ("spike_times", "spike_rate"):
            print(f"Aligning {data} to trials.")
            # we let a dedicated function handle aligning spike times
            return self._get_aligned_spike_times(
                label, event, duration, min_depth=min_depth, max_depth=max_depth,
                min_spike_width=min_spike_width, max_spike_width=max_spike_width,
                rate=data == "spike_rate", sigma=sigma, uncurated=uncurated
            )

        action_labels = self.get_action_labels()

        if raw:
            print(f"Aligning raw {data} data to trials.")
            getter = getattr(self, f"get_{data}_data_raw", None)
            if not getter:
                raise PixelsError(f"align_trials: {data} doesn't have a 'raw' option.")
            values, sample_rate = getter()

        else:
            print(f"Aligning {data} data to trials.")
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

        for rec_num in range(len(self.files)):
            actions = action_labels[rec_num][:, 0]
            events = action_labels[rec_num][:, 1]
            trial_starts = np.where((actions == label))[0]

            for start in trial_starts:
                centre = np.where(np.bitwise_and(events[start:start + scan_duration], event))[0]
                if len(centre) == 0:
                    raise PixelsError('Action labels probably miscalculated')
                centre = start + centre[0]
                centre = int(centre * sample_rate / self.sample_rate)
                trial = values[rec_num][centre - half + 1:centre + half + 1]
                trials.append(trial.reset_index(drop=True))

        if not trials:
            raise PixelsError("Seems the action-event combo you asked for doesn't occur")

        trials = pd.concat(
            trials, axis=1, copy=False, keys=range(len(trials)), names=["trial", "unit"]
        )
        trials = trials.sort_index(level=1, axis=1)
        trials = trials.reorder_levels(["unit", "trial"], axis=1)

        points = trials.shape[0]
        start = (- duration / 2) + (duration / points)
        timepoints = np.linspace(start, duration / 2, points)
        trials['time'] = pd.Series(timepoints, index=trials.index)
        trials = trials.set_index('time')
        return trials

    def get_cluster_info(self):
        cluster_info = []

        for rec_num, recording in enumerate(self.files):
            info_file = self.processed / f'sorted_{rec_num}' / 'cluster_info.tsv'
            try:
                info = pd.read_csv(info_file, sep='\t')
            except FileNotFoundError:
                msg = ": Can't load cluster info. Did you sort this session yet?"
                raise PixelsError(self.name + msg)

            cluster_info.append(info)

        return cluster_info

    @_cacheable
    def get_spike_widths(self, group='good', min_depth=0, max_depth=None):
        from phylib.io.model import load_model
        from phylib.utils.color import selected_cluster_color

        print("Calculating spike widths")
        waveforms = self.get_spike_waveforms(
            group=group, min_depth=min_depth, max_depth=max_depth
        )
        widths = []

        for rec_num, recording in enumerate(self.files):
            for unit in waveforms[rec_num].columns.get_level_values('unit').unique():
                u_widths = []
                u_spikes = waveforms[rec_num][unit]
                for s in u_spikes:
                    spike = u_spikes[s]
                    trough = np.where(spike.values == min(spike))[0][0]
                    after = spike.values[trough:]
                    width = np.where(after == max(after))[0][0]
                    u_widths.append(width)
                widths.append((rec_num, unit, np.median(u_widths)))

        df = pd.DataFrame(widths, columns=["rec_num", "unit", "median_ms"])
        # convert to ms from sample points
        orig_rate = int(self.spike_meta[rec_num]['imSampRate'])
        df['median_ms'] = 1000 * df['median_ms'] / orig_rate
        return df

    @_cacheable
    def get_spike_waveforms(
        self, group='good', min_depth=0, max_depth=None, min_spike_width=None,
        max_spike_width=None, uncurated=False
    ):
        from phylib.io.model import load_model
        from phylib.utils.color import selected_cluster_color

        selected_units = self.filter_units(
            group, min_depth, max_depth, min_spike_width, max_spike_width, uncurated
        )

        waveforms = []

        for rec_num, recording in enumerate(self.files):
            paramspy = self.processed / f'sorted_{rec_num}' / 'params.py'
            if not paramspy.exists():
                raise PixelsError(f"{self.name}: params.py not found")
            model = load_model(paramspy)
            rec_forms = {}

            for unit in selected_units[rec_num]:
                # get the waveforms from only the best channel
                spike_ids = model.get_cluster_spikes(unit)
                best_chan = model.get_cluster_channels(unit)[0]
                u_waveforms = model.get_waveforms(spike_ids, [best_chan])
                if u_waveforms is None:
                    raise PixelsError(f"{self.name}: unit {unit} - waveforms not read")
                rec_forms[unit] = pd.DataFrame(np.squeeze(u_waveforms).T)
            waveforms.append(pd.concat(rec_forms, axis=1))

        df = pd.concat(
            waveforms,
            axis=1,
            keys=range(len(self.files)),
            names=['rec_num', 'unit', 'spike']
        )
        # convert indexes to ms
        rate = 1000 / int(self.spike_meta[rec_num]['imSampRate'])
        df.index = df.index * rate
        return df

    @_cacheable
    def get_aligned_spike_rate_CI(
        self, label, event, win,
        bl_label=None, bl_event=None, bl_win=None,
        ss=20, CI=95, bs=10000,
        group='good', min_depth=0, max_depth=None, min_spike_width=None, max_spike_width=None,
        sigma=None, uncurated=False
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

        win : slice
            Slice object with values in seconds used to extract the window data from
            aligned firing rate data.

        bl_label, bl_event, bl_win : as above, all optional
            Equivalent to the above three parameters but for baselining data. By default
            no baselining is performed.

        ss : int, optional.
            Sample size of bootstrapped samples.

        CI : int/float, optional
            Confidence interval size. Default is 95: this returns the 2.5%, 50% and
            97.5% bootstrap sample percentiles.

        bs : int, optional
            Number of bootstrapped samples. Default is 10000.

        Remaining parameters are passed to `align_trials`.
        """
        if not isinstance(win, slice):
            raise PixelsError("Third argument to get_aligned_spike_rate_CI should be a slice object")

        selected_units = self.filter_units(
            group, min_depth, max_depth, min_spike_width, max_spike_width, uncurated
        )

        duration = 2 * max(abs(win.start - 1), abs(win.stop))
        responses = self.align_trials(
            label, event, 'spike_rate', duration=duration, min_depth=min_depth,
            max_depth=max_depth, min_spike_width=min_spike_width,
            max_spike_width=max_spike_width, sigma=sigma
        )
        series = responses.index.values
        assert series[0] <= win.start < win.stop <= series[-1]
        responses = responses.loc[win].mean()

        if bl_win is not None:
            if not isinstance(bl_win, slice):
                raise PixelsError("bl_win arg for get_aligned_spike_rate_CI should be a slice object")
            duration = 2 * max(abs(bl_win.start - 1), abs(bl_win.stop))
            label = label if bl_label is None else bl_label
            event = event if bl_event is None else bl_event
            baselines = self.align_trials(
                label, event, 'spike_rate', duration=duration,
                min_depth=min_depth, max_depth=max_depth,
                min_spike_width=min_spike_width, max_spike_width=max_spike_width,
                sigma=sigma
            )
            series = baselines.index.values
            assert series[0] <= bl_win.start < bl_win.stop <= series[-1]
            responses = responses - baselines.loc[bl_win].mean()

        lower = (100 - CI) / 2
        upper = 100 - lower
        cis = []

        for rec_num, recording in enumerate(self.files):
            rec_cis = {}
            for unit in selected_units[rec_num]:
                u_resps = responses[rec_num][unit]
                samples = np.array([np.random.choice(u_resps, size=ss) for i in range(bs)])
                medians = np.median(samples, axis=1)
                results = np.percentile(medians, [lower, 50, upper])
                rec_cis[unit] = results
            cis.append(pd.DataFrame(rec_cis, index=[lower, 50, upper]))

        df = pd.concat(cis, axis=1, keys=range(len(self.files)), names=['rec_num', 'unit'])
        return df
