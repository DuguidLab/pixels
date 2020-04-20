"""
This module provides gonogo-specific operations.
"""


from pixels import signal
from pixels.behaviours import Behaviour


class GoNoGo(Behaviour):

    def _extract_action_labels(self, behavioural_data):
        behavioural_data = signal.binarise(behavioural_data)
        action_labels = np.zeros(len(behavioural_data))

        back_sensor_on = np.where(behavioural_data["/'Back_Sensor'/'0'"] == 1)
        front_sensor_on = np.where(behavioural_data["/'Front_Sensor'/'0'"] == 1)
        front_sensor_off = np.where(behavioural_data["/'Front_Sensor'/'0'"] == 0)

        reward_signal = behavioural_data["/'Reward_Signal'/'0'"].values
        reward_onsets = np.where((reward_signal[:-1] == 0) & (reward_signal[1:] == 1))[0]
        reward_offsets = np.where((reward_signal[:-1] == 1) & (reward_signal[1:] == 0))[0]

        tone_signal = behavioural_data["/'Tone_Signal'/'0'"].values
        tone_onsets = np.where((tone_signal[:-1] == 0) & (tone_signal[1:] == 1))[0]


        raise Exception

        return action_labels
