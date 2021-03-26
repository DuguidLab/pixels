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
    # trials present in regular experiments
    rewarded_push = 1 << 0
    uncued_push = 1 << 1
    missed_tone = 1 << 2

    # trials present in optogenetic experiments
    cued_shutter_nopush = 1 << 3
    cued_laser_nopush = 1 << 4
    uncued_shutter_nopush = 1 << 5
    uncued_laser_nopush = 1 << 6
    cued_shutter_push_partial = 1 << 7
    cued_laser_push_partial = 1 << 8
    uncued_shutter_push_partial = 1 << 9
    uncued_laser_push_partial = 1 << 10
    cued_shutter_push_full = 1 << 11
    cued_laser_push_full = 1 << 12
    uncued_shutter_push_full = 1 << 13
    uncued_laser_push_full = 1 << 14
    cued_shutter_push = cued_shutter_push_partial | cued_shutter_push_full
    cued_laser_push = cued_laser_push_partial | cued_laser_push_full
    uncued_shutter_push = cued_shutter_push_partial | cued_shutter_push_full
    uncued_laser_push = uncued_laser_push_partial | uncued_laser_push_full


class Events:
    back_sensor_open = 1 << 0
    back_sensor_closed = 1 << 1
    front_sensor_open = 1 << 2
    front_sensor_closed = 1 << 3
    reward_signal = 1 << 4
    reset_signal = 1 << 5
    tone_onset = 1 << 6
    tone_offset = 1 << 7
    shutter_onset = 1 << 8
    shutter_offset = 1 << 9
    laser_onset = 1 << 10
    laser_offset = 1 << 11


class LeverPush(Behaviour):

    def _extract_action_labels(self, behavioural_data, plot=False):
        """
        Create lever-push action labels. Key:

            1: rewarded push
            2: uncued push
            3: missed tone

        """
        if "/'Laser_Signal'/'0'" in behavioural_data.columns:
            if len(behavioural_data.columns) != 8:
                raise PixelsError("Unknown channel configuration. Cannot make action labels.")
            opto = True
            # we have to get the laser signal without binarising it
            behavioural_data["/'Laser_Signal'/'0'"].loc[
                behavioural_data["/'Laser_Signal'/'0'"] > 0.1] = behavioural_data["/'Laser_Signal'/'0'"].max()
        else:
            if len(behavioural_data.columns) != 6:
                raise PixelsError("Unknown channel configuration. Cannot make action labels.")
            opto = False

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

        reset_signal = behavioural_data["/'Reset_Signal'/'0'"].values
        reset_onsets = np.where((reset_signal[:-1] == 0) & (reset_signal[1:] == 1))[0]
        action_labels[reset_onsets, 1] += Events.reset_signal

        reward_signal = behavioural_data["/'Reward_Signal'/'0'"].values
        reward_onsets = np.where(
            (reward_signal[:-1] == 0) & (reward_signal[1:] == 1)
        )[0]
        action_labels[reward_onsets, 1] += Events.reward_signal

        tone_signal = behavioural_data["/'Tone_Signal'/'0'"].values
        tone_onsets = np.where((tone_signal[:-1] == 0) & (tone_signal[1:] == 1))[0]
        tone_offsets = np.where((tone_signal[:-1] == 1) & (tone_signal[1:] == 0))[0]
        action_labels[tone_onsets, 1] += Events.tone_onset
        action_labels[tone_offsets, 1] += Events.tone_offset

        if len(behavioural_data.columns) == 8:
            opto = True
        elif len(behavioural_data.columns) ==6 :
            opto = False
        else:
            raise PixelsError("Unknown channel configuration. Cannot make action labels.")

        if opto:
            shutter_signal = behavioural_data["/'Shutter_Signal'/'0'"].values
            shutter_onsets = []
            onsets = np.where((shutter_signal[:-1] == 0) & (shutter_signal[1:] == 1))[0]
            for i, onset in enumerate(onsets):
                if i == 0 or onset - onsets[i - 1] > 200:
                    shutter_onsets.append(onset)
            shutter_onsets = np.array(shutter_onsets)
            shutter_offsets = []
            offsets = np.where((shutter_signal[:-1] == 1) & (shutter_signal[1:] == 0))[0]
            last = len(offsets) - 1
            for i, offset in enumerate(offsets):
                if last == i or offsets[i + 1] - offset > 200:
                    shutter_offsets.append(offset)
            shutter_offsets = np.array(shutter_offsets)
            action_labels[shutter_onsets, 1] += Events.shutter_onset
            action_labels[shutter_offsets, 1] += Events.shutter_offset

            laser_signal = behavioural_data["/'Laser_Signal'/'0'"].values
            laser_onsets = np.where((laser_signal[:-1] == 0) & (laser_signal[1:] == 1))[0]
            laser_offsets = np.where((laser_signal[:-1] == 1) & (laser_signal[1:] == 0))[0]
            action_labels[laser_onsets, 1] += Events.laser_onset
            action_labels[laser_offsets, 1] += Events.laser_offset

            for shutter in shutter_onsets:
                # if a tone came on at roughly the same time, it is cued
                tone = (tone_onsets < shutter + 500) * (tone_onsets > shutter - 500)
                cued = tone.any()
                if cued:
                    action = "cued_shutter_"
                    shutter = min(tone_onsets[tone][0], shutter)  # must be earliest event
                else:
                    action = "uncued_shutter_"
                laser = (laser_onsets < shutter + 500) * (laser_onsets > shutter - 500)
                if laser.any():
                    shutter = min(laser_onsets[laser][0], shutter)  # must be earliest event
                back_sensor_open = np.delete(back_sensor_onsets, (back_sensor_onsets <= shutter))
                if not back_sensor_open.size:
                    # is it the end of the session?
                    next_reset = np.delete(reset_onsets, (reset_onsets <= shutter))
                    if not next_reset.size:
                        # there is no following reset, so the session ended
                        break
                    nopush = True
                else:
                    # if the next back sensor open happens after the shutter ends, there was no push
                    offset = np.delete(shutter_offsets, (shutter_offsets <= shutter))[0]
                    back_sensor_open = back_sensor_open[0]
                    nopush = back_sensor_open > offset
                if nopush:
                    action += "nopush"
                else:
                    front_closed = np.delete(front_sensor_offsets, (front_sensor_offsets <= shutter))
                    if front_closed.size:
                        front_closed = front_closed[0]
                    else:
                        front_closed = len(front_sensor_signal) - 1
                    if cued:
                        # if the front sensor close happened before the next reset, push was full
                        next_reset = np.delete(reset_onsets, (reset_onsets <= shutter))
                        if not next_reset.size:
                            break
                        next_reset = next_reset[0]
                        if front_closed < next_reset:
                            action += "push_full"
                        else:
                            action += "push_partial"
                    else:
                        # if the next back sensor close after laser onset is before the
                        # first front sensor close after laser onset, the push was partial
                        next_back_close = np.delete(back_sensor_offsets, (back_sensor_offsets <= shutter))
                        if not next_back_close.size:
                            break
                        next_back_close = next_back_close[0]
                        if next_back_close < front_closed:
                            action += "push_partial"
                        else:
                            action += "push_full"
                action_labels[shutter, 0] += getattr(ActionLabels, action)

            for laser in laser_onsets:
                # if a tone came on at roughly the same time, it is cued
                tone = (tone_onsets < laser + 500) * (tone_onsets > laser - 500)
                cued = tone.any()
                if cued:
                    action = "cued_laser_"
                    laser = min(tone_onsets[tone][0], laser)  # must be earliest event
                else:
                    action = "uncued_laser_"
                shutter = (shutter_onsets < laser + 500) * (shutter_onsets > laser - 500)
                if shutter.any():
                    laser = min(shutter_onsets[shutter][0], laser)  # must be earliest event
                back_sensor_open = np.delete(back_sensor_onsets, (back_sensor_onsets <= laser))
                if not back_sensor_open.size:
                    # is it the end of the session?
                    next_reset = np.delete(reset_onsets, (reset_onsets <= laser))
                    if not next_reset.size:
                        # there is no following reset, so the session ended
                        break
                    nopush = True
                else:
                    # if the next back sensor open happens after the laser ends, there was no push
                    back_sensor_open = back_sensor_open[0]
                    offset = np.delete(laser_offsets, (laser_offsets <= laser))[0]
                    nopush = back_sensor_open > offset
                if nopush:
                    action += "nopush"
                else:
                    front_closed = np.delete(front_sensor_offsets, (front_sensor_offsets <= laser))
                    if front_closed.size:
                        front_closed = front_closed[0]
                    else:
                        front_closed = len(front_sensor_signal) - 1
                    if cued:
                        # if the front sensor close happened before the next reset, push was full
                        next_reset = np.delete(reset_onsets, (reset_onsets <= laser))
                        if not next_reset.size:
                            break
                        next_reset = next_reset[0]
                        if front_closed < next_reset:
                            action += "push_full"
                        else:
                            action += "push_partial"
                    else:
                        # if the next back sensor close after laser onset is before the
                        # first front sensor close after laser onset, the push was partial
                        next_back_close = np.delete(back_sensor_offsets, (back_sensor_offsets <= laser))
                        if not next_back_close.size:
                            break
                        next_back_close = next_back_close[0]
                        if next_back_close < front_closed:
                            action += "push_partial"
                        else:
                            action += "push_full"
                action_labels[laser, 0] += getattr(ActionLabels, action)

        else:
            for reward in reward_onsets:
                previous_tone = tone_onsets[np.where(reward - tone_onsets >= 0)[0]]
                if previous_tone.size:
                    action_labels[previous_tone[-1], 0] += ActionLabels.rewarded_push

            for tone in tone_onsets:
                if not action_labels[tone, 0]:
                    action_labels[tone, 0] += ActionLabels.missed_tone

            for push in back_sensor_onsets:
                previous_tone = tone_onsets[np.where(push - tone_onsets >= 0)[0]]

                if not previous_tone.size:
                    action_labels[push, 0] += ActionLabels.uncued_push
                    continue  # no tones yet, must be uncued

                previous_reset = reset_onsets[np.where(push - reset_onsets >= 0)[0]]
                if not previous_reset.size:
                    continue  # if no resets yet, must be within trial

                if previous_reset[-1] < previous_tone[-1]:
                    continue  # must be within trial
                action_labels[push, 0] += ActionLabels.uncued_push

        if plot:
            plt.clf()
            _, axes = plt.subplots(9 if opto else 7, 1, sharex=True, sharey=False)
            axes[0].plot(back_sensor_signal)
            axes[1].plot(behavioural_data["/'Front_Sensor'/'0'"].values)
            axes[2].plot(reward_signal)
            axes[3].plot(behavioural_data["/'Reset_Signal'/'0'"].values)
            axes[4].plot(tone_signal)
            if opto:
                axes[5].plot(shutter_signal)
                axes[6].plot(laser_signal)
                axes[7].plot(action_labels[:, 0])
                axes[8].plot(action_labels[:, 1])
            else:
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
