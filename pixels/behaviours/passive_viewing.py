"""
This module provides passive viewing task specific operations.
"""

from __future__ import annotations

import pickle

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from reach.session import Outcomes, Targets

from pixels import Experiment, PixelsError
from pixels import signal, ioutils
from pixels.behaviours import Behaviour


class VisStimLabels:
    """
    These visual stim labels cover all possible trial types.
    'sf_004' and 'sf_016' correspond to the grating trial's spatial frequency.
    This means trials with all temporal frequencies are included here.
    'left' and 'right' notes the direction where the grating is drifting towards.
    Gratings have six possible orientations in total, 30 deg aparts.
    'nat_movie' and 'cage_movie' correspond to movie trials, where a 60s of natural
    or animal home cage movie is played.

    for ESCN00 to 03, start of each of these trials is marked by a rise of 500 ms TTL
    pulse.
    Within each grating trial, each orientation grating shows for 2s, their
    beginnings and ends are both marked by a rise of 500 ms TTL pulse.

     *dec 16th 2022: in the near future, there will be 'nat_image' trial, where...

    To align trials to more than one action type they can be bitwise OR'd i.e.
    `miss_left | miss_right` will match all miss trials.
    """
    # gratings 0.04 & 0.16 spatial frequency
    sf_004 = 1 << 0
    sf_016 = 1 << 1

    # directions
    left = 1 << 2
    right = 1 << 3

    # orientations
    g_0 = 1 << 4
    g_30 = 1 << 5
    g_60 = 1 << 6
    g_90 = 1 << 7
    g_120 = 1 << 8
    g_150 = 1 << 9

    # iti marks
    black = 1 << 10
    gray = 1 << 11

    # movies
    nat_movie = 1 << 12
    cage_movie = 1 << 13

    """
    # spatial freq 0.04 directions 0:30:150
    g_004_0 = sf_004 & g_0 & left
    g_004_30 = sf_004 & g_30 & left
    g_004_60 = sf_004 & g_60 & left
    g_004_90 = sf_004 & g_90 & left
    g_004_120 = sf_004 & g_120 & left
    g_004_150 = sf_004 & g_150 & left

    # spatial freq 0.04 directions 180:30:330
    g_004_180 = sf_004 & g_0 & right
    g_004_210 = sf_004 & g_30 & right
    g_004_240 = sf_004 & g_60 & right
    g_004_270 = sf_004 & g_90 & right
    g_004_300 = sf_004 & g_120 & right
    g_004_330 = sf_004 & g_150 & right

    # spatial freq 0.16 directions 0:30:150
    g_016_0 = sf_016 & g_0 & left
    g_016_30 = sf_016 & g_30 & left
    g_016_60 = sf_016 & g_60 & left
    g_016_90 = sf_016 & g_90 & left
    g_016_120 = sf_016 & g_120 & left
    g_016_150 = sf_016 & g_150 & left

    # spatial freq 0.16 directions 180:30:330
    g_016_180 = sf_016 & g_0 & right
    g_016_210 = sf_016 & g_30 & right
    g_016_240 = sf_016 & g_60 & right
    g_016_270 = sf_016 & g_90 & right
    g_016_300 = sf_016 & g_120 & right
    g_016_330 = sf_016 & g_150 & right

    # spatial freq 0.04
    g_004 = sf_004 & (g_0 | )& (left | right)
    g_004 = g_004_0 | g_004_30 | g_004_60 | g_004_90 |

    # spatial freq 0.16
    """

    #TODO: natural images

class Events:
    start = 1 << 0
    end = 1 << 1

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



class PassiveViewing(Behaviour):
    #TODO: continue here, see how to map behaviour metadata
    def _preprocess_behaviour(self, rec_num, behavioural_data):
        # Correction for sessions where sync channel interfered with LED channel
        if behavioural_data["/'ReachLEDs'/'0'"].min() < -2:
            behavioural_data["/'ReachLEDs'/'0'"] = behavioural_data["/'ReachLEDs'/'0'"] \
                + 0.5 * behavioural_data["/'NpxlSync_Signal'/'0'"]

        behavioural_data = signal.binarise(behavioural_data)
        action_labels = np.zeros((len(behavioural_data), 2), dtype=np.uint64)

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
                i = -1
                while meta_onsets[i] > len(cue_leds):
                    metadata["trials"].pop()
                    i -= 1
                assert len(led_onsets) == len(metadata["trials"])
            else:
                # If you have come to debug and see why this error was raised, try:
                # led_onsets - meta_onsets[:len(led_onsets)]  # This might show the problem
                # meta_onsets - led_onsets[:len(meta_onsets)]  # This might show the problem
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
            if "end" in last_trial:
                # Take known offset from metadata
                offset = led_onsets[-1] + (last_trial['end'] - last_trial['start']) * 1000
                led_offsets = np.append(led_offsets, int(offset))
            else:
                # If not possible, just remove last onset
                led_onsets = led_onsets[:-1]
                metadata["trials"].pop()
            assert len(led_offsets) == len(led_onsets)

        # QA: For some reason, sometimes the final trial metadata doesn't include the
        # final led-off even though it is detectable in the TDMS data.
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

    def draw_slit_thresholds(self, project: str, force: bool = False):
        """
        Draw lines on the slits using EasyROI. If ROIs already exist, skip.

        Parameters
        ==========
        project : str
            The DLC project i.e. name/prefix of the camera.

        force : bool
            If true, we will draw new lines even if the output file exists.

        """
        # Only needed for this method
        import cv2
        import EasyROI

        output = self.processed / f"slit_thresholds_{project}.pickle"

        if output.exists() and not force:
            print(self.name, "- slits drawn already.")
            return

        # Let's take the average between the first and last frames of the whole session.
        videos = []

        for recording in self.files:
            for v, video in enumerate(recording.get("camera_data", [])):
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

        if not videos:
            raise PixelsError("No videos were found to draw slits on.")

        first_frame = ioutils.load_video_frame(videos[0], 1)
        last_duration = ioutils.get_video_dimensions(videos[-1])[2]
        last_frame = ioutils.load_video_frame(videos[-1], last_duration - 1)

        average_frame = np.concatenate(
            [first_frame[..., None], last_frame[..., None]],
            axis=2,
        ).mean(axis=2)
        average_frame = np.squeeze(average_frame) / 255

        # Interactively draw ROI
        global _roi_helper
        if _roi_helper is None:
            # Ugly but we can only have one instance of this
            _roi_helper = EasyROI.EasyROI(verbose=False)
        lines = _roi_helper.draw_line(average_frame, 2)
        cv2.destroyAllWindows()  # Needed otherwise EasyROI errors

        # Save a copy of the frame with ROIs to PNG file
        png = output.with_suffix(".png")
        copy = EasyROI.visualize_line(average_frame, lines, color=(255, 0, 0))
        plt.imsave(png, copy, cmap='gray')

        # Save lines to file
        with output.open('wb') as fd:
            pickle.dump(lines['roi'], fd)

    def inject_slit_crossings(self):
        """
        Take the lines drawn from `draw_slit_thresholds` above, get the reach
        coordinates from DLC output, identify the timepoints when successful reaches
        crossed the lines, and add the `reach_onset` event to those timepoints in the
        action labels. Also identify which trials need clearing up, i.e. those with
        multiple reaches or have failed motion tracking, and exclude those.
        """
        lines = {}
        projects = ("LeftCam", "RightCam")

        for project in projects:
            line_file = self.processed / f"slit_thresholds_{project}.pickle"

            if not line_file.exists():
                print(self.name, "- Lines not drawn for session.")
                return

            with line_file.open("rb") as f:
                proj_lines = pickle.load(f)

            lines[project] = {
                tt:[pd.Series(p, index=["x", "y"]) for p in points.values()]
                for tt, points in proj_lines.items()
            }

        action_labels = self.get_action_labels()
        event = Events.led_off

        # https://bryceboe.com/2006/10/23/line-segment-intersection-algorithm
        def ccw(A, B, C):
            return (C.y - A.y) * (B.x - A.x) > (B.y - A.y) * (C.x - A.x)

        for tt, action in enumerate(
            (ActionLabels.correct_left, ActionLabels.correct_right),
        ):
            data = {}
            trajectories = {}

            for project in projects:
                proj_data = self.align_trials(
                    action,
                    event,
                    "motion_tracking",
                    duration=6,
                    dlc_project=project,
                )
                proj_traj, = check_scorers(proj_data)
                data[project] = proj_traj
                trajectories[project] = get_reach_trajectories(proj_traj)[0]

            for rec_num, recording in enumerate(self.files):
                actions = action_labels[rec_num][:, 0]
                events = action_labels[rec_num][:, 1]
                trial_starts = np.where(np.bitwise_and(actions, action))[0]

                for t, start in enumerate(trial_starts):
                    centre = np.where(np.bitwise_and(events[start:start + 6000], event))[0]
                    if len(centre) == 0:
                        raise PixelsError('Action labels probably miscalculated')
                    centre = start + centre[0]
                    # centre is index of this rec's grasp
                    onsets = []
                    for project, motion in trajectories.items():
                        left_hand = motion["left_hand_median"][t][:0]
                        right_hand = motion["right_hand_median"][t][:0]
                        pt1, pt2 = lines[project][tt]

                        x_l = left_hand.iloc[-10:]["x"].mean()
                        x_r = right_hand.iloc[-10:]["x"].mean()
                        hand = right_hand
                        if project == "LeftCam":
                            if x_l > x_r:
                                hand = left_hand
                        else:
                            if x_r > x_l:
                                hand = left_hand

                        segments = zip(
                            hand.iloc[::-1].iterrows(),
                            hand.iloc[-2::-1].iterrows(),
                        )

                        for (end, ptend), (start, ptsta) in segments:
                            if (
                                ccw(pt1, ptsta, ptend) != ccw(pt2, ptsta, ptend) and
                                ccw(pt1, pt2, ptsta) != ccw(pt1, pt2, ptend)
                            ):
                                # These lines intersect
                                #print("x from ", ptsta.x, " to ", ptend.x)
                                break
                        #if ptend.y > 300 or ptsta.y > 300:
                        #    assert 0
                        onsets.append(start)

                    onset = max(onsets)
                    onset_timepoint = round(centre + (onset * 1000))
                    events[onset_timepoint] |= Events.reach_onset

                output = self.processed / recording['action_labels']
                np.save(output, action_labels[rec_num])


global _roi_helper
_roi_helper = None


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


def get_reach_velocities(*dfs: pd.DataFrame) -> tuple[pd.DataFrame]:
    """
    Get the velocity curves for the provided reach trajectories.
    """
    results = []

    for df in dfs:
        df = df.copy()
        deltas = np.square(df.iloc[1:].values - df.iloc[:-1].values)
        # Fill the start with a row of zeros - each value is delta in previous 1 ms
        deltas = np.append(np.zeros((1, deltas.shape[1])), deltas, axis=0)
        deltas = np.sqrt(deltas[:, ::2] + deltas[:, 1::2])
        df = df.drop([c for c in df.columns if "y" in c], axis=1)
        df = df.rename({"x": "delta"}, axis='columns')
        df.values[:] = deltas
        results.append(df)

    return tuple(results)


def get_reach_trajectories(*dfs: pd.DataFrame) -> tuple[pd.DataFrame]:
    """
    Get the median centre point of the hand coordinates - i.e. for the labels for each
    of the four digits and hand centre for both paws.
    """
    assert dfs
    bodyparts = get_body_parts(dfs[0])
    right_paw = [p for p in bodyparts if p.startswith("right")]
    left_paw = [p for p in bodyparts if p.startswith("left")]

    trajectories_l = []
    trajectories_r = []

    for df in dfs:
        per_ses_l = []
        per_ses_r = []

        # Ugly hack so this function works on single sessions
        if "session" not in df.columns.names:
            df = pd.concat([df], keys=[0], axis=1)
            sessions = [0]
            single_session = True
        else:
            single_session = False
            sessions = df.columns.get_level_values("session").unique()

        for s in sessions:
            per_trial_l = []
            per_trial_r = []

            trials = df[s].columns.get_level_values("trial").unique()
            for t in trials:
                tdf = df[s][t]
                left = pd.concat((tdf[p] for p in left_paw), keys=left_paw)
                right = pd.concat((tdf[p] for p in right_paw), keys=right_paw)
                per_trial_l.append(left.groupby(level=1).median())
                per_trial_r.append(right.groupby(level=1).median())

            per_ses_l.append(pd.concat(per_trial_l, axis=1, keys=trials))
            per_ses_r.append(pd.concat(per_trial_r, axis=1, keys=trials))

        trajectories_l.append(pd.concat(per_ses_l, axis=1, keys=sessions))
        trajectories_r.append(pd.concat(per_ses_r, axis=1, keys=sessions))

    if single_session:
        return tuple(
            pd.concat(
                [trajectories_l[i][0], trajectories_r[i][0]],
                axis=1,
                keys=["left_hand_median", "right_hand_median"],
            )
            for i in range(len(dfs))
        )
    return tuple(
        pd.concat(
            [trajectories_l[i], trajectories_r[i]],
            axis=1,
            keys=["left_hand_median", "right_hand_median"],
        )
        for i in range(len(dfs))
    )


def check_scorers(*dfs: pd.DataFrame) -> tuple[pd.DataFrame]:
    """
    Checks that the scorers are identical for all data in the dataframes. These are the
    dataframes as returned from Exp.align_trials for motion_tracking data.

    It returns the dataframes with the scorer index level removed.
    """
    scorers = set(
        s
        for df in dfs
        for s in df.columns.get_level_values("scorer").unique()
    )

    assert len(scorers) == 1, scorers
    return tuple(df.droplevel("scorer", axis=1) for df in dfs)


def get_body_parts(df: pd.DataFrame) -> list[str]:
    """
    Get the list of body part labels.
    """
    return df.columns.get_level_values("bodyparts").unique()
