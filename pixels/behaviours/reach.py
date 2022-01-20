"""
This module provides reach task specific operations.
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from reach.session import Outcomes, Targets

from pixels import Experiment, PixelsError
from pixels import signal
from pixels.behaviours import Behaviour


class ActionLabels:
    """
    These actions cover all possible trial types. 'Left' and 'right' correspond to the
    trial's correct side i.e. which LED was illuminated. This means `incorrect_left`
    trials involved reaches to the right hand target when the left LED was on.

    To align trials to more than one action type they can be bitwise OR'd i.e.
    `miss_left | miss_right` will match all miss trials.
    """
    miss_left = 1 << 0
    miss_right = 1 << 1
    correct_left = 1 << 2
    correct_right = 1 << 3
    incorrect_left = 1 << 4
    incorrect_right = 1 << 5

    # visual-only experiments with naive mice
    naive_left_short = 1 << 6
    naive_left_long = 1 << 7
    naive_right_short = 1 << 8
    naive_right_long = 1 << 9
    naive_left = naive_left_short | naive_left_long
    naive_right = naive_right_short | naive_right_long
    naive_short = naive_left_short | naive_right_short
    naive_long = naive_left_long | naive_right_long


class Events:
    led_on = 1
    led_off = 2


# These are used to convert the trial data into Actions and Events
_side_map = {
    Targets.LEFT: "left",
    Targets.RIGHT: "right",
}

_action_map = {
    Outcomes.MISSED: "miss",
    Outcomes.CORRECT: "correct",
    Outcomes.INCORRECT: "incorrect",
}



class Reach(Behaviour):
    def _preprocess_behaviour(self, rec_num, behavioural_data):
        # Correction for sessions where sync channel interfered with LED channel
        if behavioural_data["/'ReachLEDs'/'0'"].min() < -2:
            behavioural_data["/'ReachLEDs'/'0'"] = behavioural_data["/'ReachLEDs'/'0'"] \
                + 0.5 * behavioural_data["/'NpxlSync_Signal'/'0'"]

        behavioural_data = signal.binarise(behavioural_data)
        action_labels = np.zeros((len(behavioural_data), 2), dtype=np.int16)

        try:
            cue_leds = behavioural_data["/'ReachLEDs'/'0'"].values
        except KeyError:
            # some early recordings still used this key
            cue_leds = behavioural_data["/'Back_Sensor'/'0'"].values

        led_onsets = np.where((cue_leds[:-1] == 0) & (cue_leds[1:] == 1))[0]
        led_offsets = np.where((cue_leds[:-1] == 1) & (cue_leds[1:] == 0))[0]
        action_labels[led_onsets, 1] += Events.led_on
        action_labels[led_offsets, 1] += Events.led_off
        metadata = self.metadata[rec_num]

        # QA: Check that the JSON and TDMS data have the same number of trials
        if len(led_onsets) != len(metadata["trials"]):
            # If they do not have the same number, perhaps the TDMS was stopped too early
            meta_onsets = np.array([t["start"] for t in metadata["trials"]]) * 1000
            meta_onsets = (meta_onsets - meta_onsets[0] + led_onsets[0]).astype(int)
            if meta_onsets[-1] > len(cue_leds):
                # TDMS stopped too early, continue anyway.
                metadata["trials"].pop()
            else:
                # If you have come to debug and see why this error was raised, try:
                # led_onsets - meta_onsets[:-1]  # This might show the problem
                # Then just patch a fix here:
                if self.name == "211027_VR49" and rec_num == 1:
                    del metadata["trials"][52]  # Maybe cable fell out of DAQ input?
                else:
                    raise PixelsError(
                        f"{self.name}: Mantis and Raspberry Pi behavioural "
                        "data have different no. of trials"
                    )

        # QA: Last offset not found in tdms data?
        if len(led_offsets) < len(led_onsets):
            last_trial = self.metadata[rec_num]['trials'][-1]
            offset = led_onsets[-1] + (last_trial['end'] - last_trial['start']) * 1000
            led_offsets = np.append(led_offsets, int(offset))
            assert len(led_offsets) == len(led_onsets)

        # QA: For some reason, sometimes the final trial doesn't include the final led-off
        elif len(led_offsets) == len(led_onsets):
            # Not sure how to deal with this if led_offsets and led_onsets differ in length
            if len(metadata["trials"][-1]) == 1 and "start" in metadata["trials"][-1]:
                # Remove it, because we would have to check the video to get all of the
                # information about the trial, and it's too complicated.
                metadata["trials"].pop()
                led_onsets = led_onsets[:-1]
                led_offsets = led_offsets[:-1]

        # QA: Check that the cue durations (mostly) match between JSON and TDMS data
        # This compares them at 10s of milliseconds resolution
        cue_durations_tdms = (led_offsets - led_onsets) / 100
        cue_durations_json = np.array(
            [t['end'] - t['start'] for t in metadata['trials']]
        ) * 10
        error = sum(
            (cue_durations_tdms - cue_durations_json).round() != 0
        ) / len(led_onsets)
        if error > 0.05:
            raise PixelsError(
                f"{self.name}: Mantis and Raspberry Pi behavioural data have mismatching trial data."
            )

        return behavioural_data, action_labels, led_onsets

    def _extract_action_labels(self, rec_num, behavioural_data, plot=False):
        behavioural_data, action_labels, led_onsets = self._preprocess_behaviour(rec_num, behavioural_data)

        for i, trial in enumerate(self.metadata[rec_num]["trials"]):
            side = _side_map[trial["spout"]]
            outcome = trial["outcome"]
            if outcome in _action_map:
                action = _action_map[trial["outcome"]]
                action_labels[led_onsets[i], 0] += getattr(ActionLabels, f"{action}_{side}")

        if plot:
            plt.clf()
            _, axes = plt.subplots(4, 1, sharex=True, sharey=True)
            axes[0].plot(back_sensor_signal)
            if "/'Back_Sensor'/'0'" in behavioural_data:
                axes[1].plot(behavioural_data["/'Back_Sensor'/'0'"].values)
            else:
                axes[1].plot(behavioural_data["/'ReachCue_LEDs'/'0'"].values)
            axes[2].plot(action_labels[:, 0])
            axes[3].plot(action_labels[:, 1])
            plt.plot(action_labels[:, 1])
            plt.show()

        return action_labels


class VisualOnly(Reach):
    def _extract_action_labels(self, behavioural_data, plot=False):
        behavioural_data, action_labels, led_onsets = self._preprocess_behaviour(behavioural_data)

        for i, trial in enumerate(self.metadata["trials"]):
            label = "naive_" + _side_map[trial["spout"]] + "_"
            if trial["cue_duration"] > 125:
                label += "long"
            else:
                label += "short"
            action_labels[led_onsets[i], 0] += getattr(ActionLabels, label)

        return action_labels
