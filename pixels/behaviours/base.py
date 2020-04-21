"""
This module provides a base class for experimental sessions that must be used as the
base for defining behaviour-specific processing.
"""


import datetime
import json
import numpy as np
import os
import scipy.signal
import time
from abc import ABC, abstractmethod
from pathlib import Path

from pixels import ioutils
from pixels import signal


class Behaviour(ABC):

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

        self._action_labels = []
        self._lag = [None] * len(self.files)

    def process_behaviour(self):
        """
        Process behavioural data from raw tdms and align to neuropixels data.
        """
        self._action_labels = [None] * len(self.files)
        for rec_num, recording in enumerate(self.files):
            print(f">>>>> Processing behaviour for recording {rec_num + 1} of {len(self.files)}")

            print(f"> Loading behaviour TDMS")
            behavioural_data = ioutils.read_tdms(recording['behaviour'])

            print(f"> Downsampling to {self.sample_rate} Hz")
            behav_array = signal.resample(behavioural_data, 25000, self.sample_rate)
            behavioural_data.iloc[:len(behav_array), :] = behav_array
            behavioural_data = behavioural_data[:len(behav_array)]
            del behav_array

            if self._lag[rec_num] is not None:
                lag_start, lag_end = self._lag[rec_num]
            else:
                lag_start, lag_end = self.sync_data(rec_num, behavioural_data=behavioural_data)

            print(f"> Extracting action labels")
            behavioural_data = behavioural_data[max(lag_start, 0):-1-max(lag_end, 0)]
            behavioural_data.index = range(len(behavioural_data))
            self._action_labels[rec_num] = self._extract_action_labels(behavioural_data)

            print(f"> Saving action labels to:")
            print(f"  {recording['action_labels']}")
            np.save(recording['action_labels'], self._action_labels[rec_num])

        print("> Done!")

    def sync_data(self, rec_num, behavioural_data=None, sync_channel=None):
        """
        This method will calculate and return the lag between the behavioural data and
        the neuropixels data for each recording.

        behavioural_data and sync_channel will be loaded from file and downsampled if
        not provided, otherwise if provided they must already be the same sample
        frequency.

        Parameters
        ----------
        rec_num : int
            The recording number, i.e. index of self.files to get file paths.

        behavioural_data : pandas.DataFrame, optional
            The behavioural data loaded from the TDMS file.

        sync_channel : np.ndarray, optional
            The sync channel from either the spike or LFP data.

        Returns
        -------
        lag_start : int
            The number of sample points that the behavioural data has extra at the
            start of the recording.

        lag_end : int
            The same as above but at the end.

        """
        print("> Finding lag between sync channels")
        recording = self.files[rec_num]

        if behavioural_data is None:
            print("  Loading behaviour TDMS")
            behavioural_data = ioutils.read_tdms(recording['behaviour'])
            behav_array = signal.resample(behavioural_data, 25000, self.sample_rate)
            behavioural_data.iloc[:len(behav_array), :] = behav_array
            behavioural_data = behavioural_data[:len(behav_array)]
            del behav_array

        if sync_channel is None:
            print("  Loading neuropixels sync channel")
            sync_channel = ioutils.read_bin(
                recording['spike_data'],
                self.spike_meta[rec_num]['nSavedChans'],
                channel=384,
            )
            pixels_length = sync_channel.size
            original_samp_rate = float(recording['imSampRate'])
            sync_channel = sync_channel[:120 * original_samp_rate * 2]  # 2 mins, 30kHz, back/forward
            sync_channel = signal.resample(
                sync_channel, original_samp_rate, self.sample_rate
            )
            pixels_length //= 30
        else:
            pixels_length = len(sync_channel)

        sync_behav = signal.binarise(behavioural_data["/'NpxlSync_Signal'/'0'"])
        sync_channel = signal.binarise(sync_channel).squeeze()

        print("  Finding lag")
        plot_path = Path(recording['spike_data'])
        plot_path = plot_path.with_name(plot_path.stem + '_sync.png')
        lag_start, match = signal.find_sync_lag(
            sync_behav, sync_channel, length=120000, plot=plot_path,
        )
        if match < 90:
            print("  The sync channels did not match very well. Check the plot.")

        lag_end = len(behavioural_data) - (lag_start + pixels_length)
        self._lag[rec_num] = (lag_start, lag_end)
        return lag_start, lag_end

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
            A 1-dimensional array of actions of equal length to the behavioural_data.

        """

    def get_action_labels(self):
        if not self._action_labels:
            for rec_num, recording in enumerate(self.files):
                if os.path.isfile(recording['action_labels']):
                    self._action_labels[rec_num] = np.load(recording['action_labels'])
                else:
                    print(f"Action labels for recording no. {rec_num} not yet created.")
                    self._action_labels[rec_num] = None
        return self._action_labels

    def process_lfp(self):
        """
        Process the LFP data from the raw neural recording data.
        """
        for rec_num, recording in enumerate(self.files):
            print(f">>>>> Processing LFP for recording {rec_num + 1} of {len(self.files)}")

            print("> Mapping LFP data")
            lfp_data = ioutils.read_bin(
                recording['lfp_data'],
                self.lfp_meta[rec_num]['nSavedChans'],
            )

            print(f"> Downsampling to {self.sample_rate} Hz")
            lfp_data = signal.resample(
                lfp_data, self.lfp_meta[rec_num]['imSampRate'], self.sample_rate
            )

            if self._lag[rec_num] is not None:
                lag_start, lag_end = self._lag[rec_num]
            else:
                lag_start, lag_end = self.sync_data(rec_num, sync_channel=lfp_data[:, -1])

            output = Path(recording['lfp_data'])
            output = output.with_name(output.stem + '_processed.npy')
            print(f"> Saving data to {output}")
            lfp_data = lfp_data[max(-lag_start, 0):-1-max(-lag_end, 0)]
            np.save(output, lfp_data)

    def extract_spikes(self):
        """
        Extract the spikes from raw spike data.
        """

    def process_motion_tracking(self):
        """
        Process motion tracking data either from raw camera data, or from
        previously-generated deeplabcut coordinate data.
        """
