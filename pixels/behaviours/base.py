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

    def extract_spikes(self):
        """
        Extract the spikes from raw spike data.
        """

    def process_lfp(self):
        """
        Process the LFP data from the raw neural recording data.
        """

    def process_behaviour(self):
        """
        Process behavioural data from raw tdms and align to neuropixels data.
        """
        for rec_num, recording in enumerate(self.files):
            print(f">>>>> Processing behaviour for recording {rec_num + 1} of {len(self.files)}")

            print("[1/6] Loading behaviour TDMS")
            behavioural_data = ioutils.read_tdms(recording['behaviour'])

            print("[2/6] Loading neuropixels sync channel")
            sync_pixels = ioutils.read_bin(
                recording['spike_data'],
                self.spike_meta[rec_num]['nSavedChans'],
                channel=384,
            )
            pixels_length = sync_pixels.size
            sync_pixels = sync_pixels[:120000 * 30 * 2]

            print(f"[3/6] Downsampling to {self.sample_rate} Hz")
            behav_array = signal.resample(behavioural_data, 25000, self.sample_rate)
            behavioural_data.iloc[:len(behav_array), :] = behav_array
            behavioural_data = behavioural_data[:len(behav_array)]
            del behav_array
            sync_behav = signal.binarise(behavioural_data["/'NpxlSync_Signal'/'0'"])
            sync_pixels = signal.resample(sync_pixels, 30000, self.sample_rate)
            sync_pixels = signal.binarise(sync_pixels).squeeze()
            pixels_length //= 30

            print(f"[4/6] Finding lag between sync channels")
            plot_path = Path(recording['spike_data'])
            plot_path = plot_path.with_name(plot_path.stem + '_sync.png')
            lag, match = signal.find_sync_lag(
                sync_behav, sync_pixels, length=120000, plot=plot_path,
            )

            print(f"[5/6] Extracting action labels")
            if lag > 0:
                behavioural_data = behavioural_data[lag:]
            if len(behavioural_data) > pixels_length:
                behavioural_data = behavioural_data[:pixels_length]
            behavioural_data.index = range(len(behavioural_data))
            action_labels = self._extract_action_labels(behavioural_data)

            print(f"[6/6] Saving action labels to:")
            save_path = self.data_dir / self.name / f'action_labels_{rec_num}.npy'
            print(f"      {save_path}")
            np.save(save_path, action_labels)

        print(">>>>> Done!")

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

    def process_motion_tracking(self):
        """
        Process motion tracking data either from raw camera data, or from
        previously-generated deeplabcut coordinate data.
        """
