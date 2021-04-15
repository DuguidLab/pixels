"""
This module provides push-pull specific operations.
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from pixels import Experiment, PixelsError
from pixels import signal
from pixels.behaviours import Behaviour


class ActionLabels:
    # trials present in regular experiments
    rewarded_push = 1 << 0
    rewarded_pull = 1 << 1
    uncued_push = 1 << 2
    uncued_pull = 1 << 3
    missed_push = 1 << 4
    missed_pull = 1 << 5


class Events:
    back_sensor_open = 1 << 0
    back_sensor_closed = 1 << 1
    front_sensor_open = 1 << 2
    front_sensor_closed = 1 << 3
    reward_signal = 1 << 4
    reset = 1 << 5
    front_reset = 1 << 6
    tone_onset = 1 << 7
    tone_offset = 1 << 8


class PushPull(Behaviour):

    def _extract_action_labels(self, behavioural_data, plot=False):
        """
        Create action labels for the push-pull task.
        """
        behavioural_data = signal.binarise(behavioural_data)
        action_labels = np.zeros((len(behavioural_data), 2), dtype=np.int16)

        back_sensor_signal = behavioural_data["/'Back_Sensor'/'0'"].values
        back_sensor_onsets = np.where(
            (back_sensor_signal[:-1] == 0) & (back_sensor_signal[1:] == 1)
        )[0]
        back_sensor_offsets = np.where(
            (back_sensor_signal[:-1] == 1) & (back_sensor_signal[1:] == 0)
        )[0]
        action_labels[back_sensor_onsets, 1] += Events.back_sensor_open
        action_labels[back_sensor_offsets, 1] += Events.back_sensor_closed

        front_sensor_signal = behavioural_data["/'Front_Sensor'/'0'"].values
        front_sensor_onsets = np.where(
            (front_sensor_signal[:-1] == 0) & (front_sensor_signal[1:] == 1)
        )[0]
        front_sensor_offsets = np.where(
            (front_sensor_signal[:-1] == 1) & (front_sensor_signal[1:] == 0)
        )[0]
        action_labels[front_sensor_onsets, 1] += Events.front_sensor_open
        action_labels[front_sensor_offsets, 1] += Events.front_sensor_closed

        reward_signal = behavioural_data["/'Reward_Signal'/'0'"].values
        reward_onsets = np.where(
            (reward_signal[:-1] == 0) & (reward_signal[1:] == 1)
        )[0]
        action_labels[reward_onsets, 1] += Events.reward_signal

        reset_signal = behavioural_data["/'Reset_Signal'/'0'"].values
        reset_onsets = np.where((reset_signal[:-1] == 0) & (reset_signal[1:] == 1))[0]
        action_labels[reset_onsets, 1] += Events.reset

        tone_signal = behavioural_data["/'Tone_Signal'/'0'"].values
        tone_onsets = np.where((tone_signal[:-1] == 0) & (tone_signal[1:] == 1))[0]
        tone_offsets = np.where((tone_signal[:-1] == 1) & (tone_signal[1:] == 0))[0]
        action_labels[tone_onsets, 1] += Events.tone_onset
        action_labels[tone_offsets, 1] += Events.tone_offset

        front_reset = behavioural_data["/'Front_Reset'/'0'"].values
        front_reset_onsets = np.where((front_reset[:-1] == 0) & (front_reset[1:] == 1))[0]
        action_labels[front_reset_onsets, 1] += Events.front_reset

        for i, reward in enumerate(reward_onsets):
            previous_tone = tone_onsets[np.where(reward - tone_onsets >= 0)[0]][-1]
            if back_sensor_signal[previous_tone] == 0:
                action_labels[previous_tone, 0] = ActionLabels.rewarded_push
            elif front_sensor_signal[previous_tone] == 0:
                action_labels[previous_tone, 0] = ActionLabels.rewarded_pull
            else:
                raise PixelsError(f"Unsure if reward {i + 1} followed a push or a pull")

        for i, tone in enumerate(tone_onsets):
            if not action_labels[tone, 0]:
                if back_sensor_signal[tone] == 0:
                    action_labels[tone, 0] = ActionLabels.missed_push
                elif front_sensor_signal[tone] == 0:
                    action_labels[tone, 0] = ActionLabels.missed_pull
                else:
                    raise PixelsError(f"Unsure if missed tone {i + 1} cued a push or a pull")

        # TODO: uncued movements
        #for push in back_sensor_onsets:
        #    previous_tone = tone_onsets[np.where(push - tone_onsets >= 0)[0]]

        #    if not previous_tone.size:
        #        action_labels[push, 0] = ActionLabels.uncued_push
        #        continue  # no tones yet, must be uncued

        #    previous_reset = reset_onsets[np.where(push - reset_onsets >= 0)[0]]
        #    if not previous_reset.size:
        #        continue  # if no resets yet, must be within trial

        #    if previous_reset[-1] < previous_tone[-1]:
        #        continue  # must be within trial
        #    action_labels[push, 0] = ActionLabels.uncued_push

        #for pull in front_sensor_onsets:
        #    previous_tone = tone_onsets[np.where(pull - tone_onsets >= 0)[0]]

        #    if not previous_tone.size:
        #        action_labels[push, 0] = ActionLabels.uncued_push
        #        continue  # no tones yet, must be uncued

        #    previous_reset = reset_onsets[np.where(push - reset_onsets >= 0)[0]]
        #    if not previous_reset.size:
        #        continue  # if no resets yet, must be within trial

        #    if previous_reset[-1] < previous_tone[-1]:
        #        continue  # must be within trial
        #    action_labels[push, 0] = ActionLabels.uncued_push

        if plot:
            plt.clf()
            _, axes = plt.subplots(8, 1, sharex=True, sharey=False)
            axes[0].plot(back_sensor_signal)
            axes[1].plot(front_sensor_signal)
            axes[2].plot(reward_signal)
            axes[3].plot(reset_signal)
            axes[4].plot(tone_signal)
            axes[5].plot(front_sensor_signal)
            axes[6].plot(action_labels[:, 0])
            axes[7].plot(action_labels[:, 1])
            plt.show()

        return action_labels
