"""
This module provides lever-push specific operations.
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from pixels import Experiment, PixelsError
from pixels import signal
from pixels.behaviours import Behaviour


class ActionLabels:
    rewarded_push = 1
    uncued_push = 2
    missed_tone = 3


class Events:
    back_sensor_open = 1
    back_sensor_closed = 2
    front_sensor_open = 3
    front_sensor_closed = 4
    reward_signal = 5
    reset_signal = 6
    tone_onset = 7
    tone_offset = 8


class LeverPush(Behaviour):

    def _extract_action_labels(self, behavioural_data, plot=False):
        """
        Create lever-push action labels. Key:

            1: rewarded push
            2: uncued push
            3: missed tone

        """
        behavioural_data = signal.binarise(behavioural_data)
        action_labels = np.zeros((len(behavioural_data), 2))

        back_sensor_signal = behavioural_data["/'Back_Sensor'/'0'"].values
        back_sensor_onsets = np.where(
            (back_sensor_signal[:-1] == 0) & (back_sensor_signal[1:] == 1)
        )[0]
        back_sensor_offsets = np.where(
            (back_sensor_signal[:-1] == 1) & (back_sensor_signal[1:] == 0)
        )[0]
        action_labels[back_sensor_onsets, 1] = Events.back_sensor_open
        action_labels[back_sensor_offsets, 1] = Events.back_sensor_closed

        front_sensor_signal = behavioural_data["/'Front_Sensor'/'0'"].values
        front_sensor_onsets = np.where(
            (front_sensor_signal[:-1] == 0) & (front_sensor_signal[1:] == 1)
        )[0]
        front_sensor_offsets = np.where(
            (front_sensor_signal[:-1] == 1) & (front_sensor_signal[1:] == 0)
        )[0]
        action_labels[front_sensor_onsets, 1] = Events.front_sensor_open
        action_labels[front_sensor_offsets, 1] = Events.front_sensor_closed

        reset_signal = behavioural_data["/'Reset_Signal'/'0'"].values
        reset_onsets = np.where((reset_signal[:-1] == 0) & (reset_signal[1:] == 1))[0]
        action_labels[reset_onsets, 1] = Events.reset_signal

        reward_signal = behavioural_data["/'Reward_Signal'/'0'"].values
        reward_onsets = np.where(
            (reward_signal[:-1] == 0) & (reward_signal[1:] == 1)
        )[0]
        action_labels[reward_onsets, 1] = Events.reward_signal

        tone_signal = behavioural_data["/'Tone_Signal'/'0'"].values
        tone_onsets = np.where((tone_signal[:-1] == 0) & (tone_signal[1:] == 1))[0]
        tone_offsets = np.where((tone_signal[:-1] == 1) & (tone_signal[1:] == 0))[0]
        action_labels[tone_onsets, 1] = Events.tone_onset
        action_labels[tone_offsets, 1] = Events.tone_offset

        for reward in reward_onsets:
            previous_tone = tone_onsets[np.where(reward - tone_onsets >= 0)[0]]
            if previous_tone:
                action_labels[previous_tone[-1], 0] = ActionLabels.rewarded_push

        for tone in tone_onsets:
            if not action_labels[tone, 0]:
                action_labels[tone, 0] = ActionLabels.missed_tone

        for push in back_sensor_onsets:
            previous_tone = tone_onsets[np.where(push - tone_onsets >= 0)[0]]

            if not previous_tone:
                action_labels[push, 0] = ActionLabels.uncued_push
                continue  # no tones yet, must be uncued

            previous_reset = reset_onsets[np.where(push - reset_onsets >= 0)[0]]
            if not previous_reset:
                continue  # if no resets yet, must be within trial

            if previous_reset[-1] < previous_tone[-1]:
                continue  # must be within trial
            action_labels[push, 0] = ActionLabels.uncued_push

        if plot:
            plt.clf()
            _, axes = plt.subplots(7, 1, sharex=True, sharey=True)
            axes[0].plot(back_sensor_signal)
            axes[1].plot(behavioural_data["/'Front_Sensor'/'0'"].values)
            axes[2].plot(reward_signal)
            axes[3].plot(behavioural_data["/'Reset_Signal'/'0'"].values)
            axes[4].plot(tone_signal)
            axes[5].plot(action_labels[:, 0])
            axes[6].plot(action_labels[:, 1])
            plt.show()

        return action_labels

    def extract_ITIs(self, label, data, raw=False):
        """
        Get inter-trial intervals. This finds all inter-trial intervals, bound by the
        previous reset offset for trials following an uncued push or missed cue, or the
        previous reward offset for trials following a rewarded push, and terminating at
        the subsequent cue onset for missed cue and rewarded push, or subsequent back
        sensor break for uncued push. Then it cuts out the intervals defined by these
        end points, rearranges this data, pads with NaNs due to variable ITI length,
        puts it into a MultiIndex DataFrame and returns it.

        Parameters
        ----------
        label : int
            An action label value to specify which trial types are desired.

        data : str
            One of 'behaviour', 'spike' or 'lfp'.

        raw : bool, optional
            Whether to get raw, unprocessed data instead of processed and downsampled
            data. Defaults to False.

        """
        print(f"Extracting ITIs from {'raw ' if raw else ''}{data} data.")
        data = data.lower()
        action_labels = self.get_action_labels()

        if data in 'behavioural':
            data = 'behavioural'
        if data not in ['behavioural', 'spike', 'lfp']:
            raise PixelsError(
                f"align_trials: data parameter should be 'behaviour', 'spike' or 'lfp'"
            )
        getter = f"get_{data}_data"
        if raw:
            data, sample_rate = getattr(self, f"{getter}_raw")()
        else:
            data = getattr(self, getter)()
            sample_rate = self.sample_rate

        if not data or data[0] is None:
            raise PixelsError(f"LeverPush.extract_ITIs: Could not get {data} data.")

        itis = []
        # The logic here is to find the last non-zero event prior to the end of the ITI,
        # itself defined as the specific action label for that trial

        for rec_num in range(len(self.files)):
            actions = action_labels[rec_num][:, 0]
            events = action_labels[rec_num][:, 1]
            iti_ends = np.where((actions == label))[0]

            for end in iti_ends:
                try:
                    start = np.where(events[:end] != 0)[0][-1]
                except IndexError:
                    start = end - 10001
                start = int(start * sample_rate / self.sample_rate)
                iti = data[rec_num][start + 1:end]
                itis.append(iti.reset_index(drop=True))

        itis = pd.concat(
            itis, axis=1, copy=False, keys=range(len(itis)), names=["trial", "unit"]
        )
        itis = itis.sort_index(level=1, axis=1)
        itis = itis.reorder_levels(["unit", "trial"], axis=1)

        return itis


class LeverPushExp(Experiment):
    def extract_ITIs(self, label, data, raw=False):
        """
        Get inter-trial intervals preceding an action.
        """
        itis = []
        for session in self.sessions:
            itis.append(session.extract_ITIs(label, data, raw))
        df = pd.concat(
            itis, axis=1, copy=False,
            keys=range(len(itis)),
            names=["session", "unit", "trial"]
        )
        return df
