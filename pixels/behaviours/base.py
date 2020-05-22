"""
This module provides a base class for experimental sessions that must be used as the
base for defining behaviour-specific processing.
"""


import datetime
import json
import os
import tarfile
import time
from abc import ABC, abstractmethod
from shutil import copyfile

import numpy as np
import pandas as pd

from pixels import ioutils
from pixels import signal
from pixels.error import PixelsError


BEHAVIOUR_HZ = 25000


class Behaviour(ABC):

    sample_rate = 1000

    def __init__(self, name, data_dir, metadata=None):
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
        self.name = name
        self.data_dir = data_dir
        self.metadata = metadata

        self.raw = self.data_dir / 'raw' / self.name
        self.interim = self.data_dir / 'interim' / self.name
        self.processed = self.data_dir / 'processed' / self.name
        self.interim.mkdir(parents=True, exist_ok=True)
        (self.processed / 'cache').mkdir(parents=True, exist_ok=True)
        self.files = ioutils.get_data_files(self.raw, name)

        self._action_labels = None
        self._behavioural_data = None
        self._spike_data = None
        self._lfp_data = None
        self._lag = None
        self.drop_data()

        self.spike_meta = [
            ioutils.read_meta(self.find_file(f['spike_meta'])) for f in self.files
        ]
        self.lfp_meta = [
            ioutils.read_meta(self.find_file(f['lfp_meta'])) for f in self.files
        ]

    def drop_data(self):
        """
        Clear attributes that store data to clear some memory.
        """
        self._action_labels = [None] * len(self.files)
        self._behavioural_data = [None] * len(self.files)
        self._spike_data = [None] * len(self.files)
        self._lfp_data = [None] * len(self.files)
        self._load_lag()

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
                self._lag[rec_num] = (rec_lag['lag_start'], rec_lag['lag_end'])

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
            print(f"    Copying {name} to interim")
            copyfile(raw, interim)
            return interim

        tar = raw.with_name(raw.name + '.tar.gz')
        if tar.exists():
            print(f"    Extracting {tar.name} to interim")
            with tarfile.open(tar) as open_tar:
                open_tar.extractall(path=self.interim)
            return interim

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
        print("> Finding lag between sync channels")
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
            sync_channel = sync_channel[:120 * orig_rate * 2]  # 2 mins, rec Hz, back/forward
            sync_channel = signal.resample(sync_channel, orig_rate, self.sample_rate)

        behavioural_data = signal.binarise(behavioural_data)
        sync_channel = signal.binarise(sync_channel)

        print("    Finding lag")
        plot = self.processed / f'sync_{rec_num}.png'
        lag_start, match = signal.find_sync_lag(
            behavioural_data, sync_channel, length=120000, plot=plot,
        )
        if match < 95:
            print("    The sync channels did not match very well. Check the plot.")

        lag_end = len(behavioural_data) - (lag_start + len(sync_channel))
        self._lag[rec_num] = (lag_start, lag_end)

        lag_json = []
        for lag_start, lag_end in self._lag:
            lag_json.append(dict(lag_start=lag_start, lag_end=lag_end))
        with (self.processed / 'lag.json').open('w') as fd:
            json.dump(lag_json, fd)

        return

    def process_behaviour(self):
        """
        Process behavioural data from raw tdms and align to neuropixels data.
        """
        for rec_num, recording in enumerate(self.files):
            print(f">>>>> Processing behaviour for recording {rec_num + 1} of {len(self.files)}")

            print(f"> Loading behavioural data")
            behavioural_data = ioutils.read_tdms(self.find_file(recording['behaviour']))

            print(f"> Downsampling to {self.sample_rate} Hz")
            behav_array = signal.resample(behavioural_data.values, 25000, self.sample_rate)
            behavioural_data.iloc[:len(behav_array), :] = behav_array
            behavioural_data = behavioural_data[:len(behav_array)]
            del behav_array

            if self._lag[rec_num] is None:
                self.sync_data(
                    rec_num,
                    behavioural_data=behavioural_data["/'NpxlSync_Signal'/'0'"].values,
                )
            lag_start, lag_end = self._lag[rec_num]

            print(f"> Extracting action labels")
            behavioural_data = behavioural_data[max(lag_start, 0):-1-max(lag_end, 0)]
            behavioural_data.index = range(len(behavioural_data))
            self._action_labels[rec_num] = self._extract_action_labels(behavioural_data)

            output = self.processed / recording['action_labels']
            print(f"> Saving action labels to:")
            print(f"    {output}")
            np.save(output, self._action_labels[rec_num])

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
            print(f">>>>> Processing spike data for recording {rec_num + 1} of {len(self.files)}")

            data_file = self.find_file(recording['spike_data'])
            orig_rate = self.spike_meta[rec_num]['imSampRate']
            num_chans = self.spike_meta[rec_num]['nSavedChans']

            print("> Mapping spike data")
            data = ioutils.read_bin(data_file, num_chans)

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

    def extract_spikes(self):
        """
        Extract the spikes from raw spike data.
        """

    def extract_videos(self):
        """
        Extract behavioural videos from TDMS to avi.
        """
        for rec_num, recording in enumerate(self.files):
            path = self.find_file(recording['camera_data'])
            path_avi = path.with_suffix('.avi')
            if not path_avi.exists():
                df = ioutils.read_tdms(path)
                meta = ioutils.read_tdms(self.find_file(recording['camera_meta']))
                actual_heights = meta["/'keys'/'IMAQdxActualHeight'"]
                ind_skipped = meta["/'frames'/'ind_skipped'"].dropna()

                height = actual_heights.max()
                remainder = ind_skipped.size - actual_heights[actual_heights != height].size
                duration = actual_heights.size - remainder
                width = df.size / (duration * height)
                if width != 640:
                    raise PixelsError("Width calculation must be incorrect, discuss.")

                video = df.values.reshape((duration, height, int(width)))
                ioutils.save_ndarray_as_avi(video, path_avi, 50)

    def process_motion_tracking(self):
        """
        Process motion tracking data either from raw camera data, or from
        previously-generated deeplabcut coordinate data.
        """

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
                    print(f"Could not find {attr[1:]} for recording {rec_num}.")
                    saved[rec_num] = None
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

    def align_trials(self, label, event, data, raw=False, duration=1):
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

        data : str
            One of 'behaviour', 'spike' or 'lfp'.

        raw : bool, optional
            Whether to get raw, unprocessed data instead of processed and downsampled
            data. Defaults to False.

        duration : int/float, optional
            The length of time desired in the output.

        """
        print(f"Aligning {data} data to trials.")
        data = data.lower()
        action_labels = self.get_action_labels()

        if data not in ['behaviour', 'spike', 'lfp']:
            raise PixelsError(f"align_trials: data parameter should be 'behaviour', 'spike' or 'lfp'")
        if data == 'behaviour':
            data = 'behavioural'
        getter = f"get_{data}_data"
        if raw:
            data, sample_rate = getattr(self, f"{getter}_raw")()
        else:
            data = getattr(self, getter)()
            sample_rate = self.sample_rate

        if not data or data[0] is None:
            raise PixelsError(f"align_trials: Could not get {data} data.")

        trials = []
        # The logic here is that the action labels will always have a sample rate of
        # self.sample_rate, whereas our data here may differ. 'duration' is used to scan
        # the action labels, so always give it 5 seconds to scan, then 'half' is used to
        # index data.
        scan_duration = self.sample_rate * 5
        half = (sample_rate * duration) // 2

        for rec_num in range(len(self.files)):
            actions = action_labels[rec_num][:, 0]
            events = action_labels[rec_num][:, 1]
            trial_starts = np.where((actions == label))[0]

            for start in trial_starts:
                centre = start + np.where(events[start:start + scan_duration] == event)[0][0]
                centre *= int(sample_rate / self.sample_rate)
                trial = data[rec_num][centre - half + 1:centre + half + 1]
                trials.append(trial.reset_index(drop=True))

        trials = pd.concat(trials, axis=1, copy=False, keys=range(len(trials)), names=["trial", "unit"])
        trials.sort_index(level=1, axis=1, inplace=True)
        trials = trials.reorder_levels(["unit", "trial"], axis=1)
        points = trials.shape[0]
        start = (- duration / 2) + (duration / points)
        end = duration / 2
        timepoints = np.linspace(start, duration / 2, points)
        trials['time'] = pd.Series(timepoints, index=trials.index)
        trials = trials.set_index('time')
        return trials
