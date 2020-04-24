"""
This module provides lever-push specific operations.
"""


import numpy as np
import matplotlib.pyplot as plt

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
            if len(previous_tone):
                action_labels[previous_tone[-1], 0] = ActionLabels.rewarded_push

        for tone in tone_onsets:
            if not action_labels[tone, 0]:
                action_labels[tone, 0] = ActionLabels.missed_tone

        for push in back_sensor_onsets:
            previous_tone = tone_onsets[np.where(push - tone_onsets >= 0)[0]]

            if not len(previous_tone):
                action_labels[push, 0] = ActionLabels.uncued_push
                continue  # no tones yet, must be uncued

            previous_reset = reset_onsets[np.where(push - reset_onsets >= 0)[0]]
            if not len(previous_reset):
                continue  # if no resets yet, must be within trial

            if previous_reset[-1] < previous_tone[-1]:
                continue  # must be within trial
            action_labels[push, 0] = ActionLabels.uncued_push

        if plot:
            plt.clf()
            fig, axes = plt.subplots(7, 1, sharex=True, sharey=True)
            axes[0].plot(back_sensor_signal)
            axes[1].plot(behavioural_data["/'Front_Sensor'/'0'"].values)
            axes[2].plot(reward_signal)
            axes[3].plot(behavioural_data["/'Reset_Signal'/'0'"].values)
            axes[4].plot(tone_signal)
            axes[5].plot(action_labels[:, 0])
            axes[6].plot(action_labels[:, 1])
            plt.show()
    
        return action_labels

    def remove_incomplete_pushes(self):
        """
        Remove uncued pushes that do not break the front sensor.
        """
        pass

    def remove_rowing_pushes(self):
        """
        Remove rewarded pushes that break the front sensor more than once.
        """
        pass
