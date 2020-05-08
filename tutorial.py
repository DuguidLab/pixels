"""
This is a tutorial on how to use the various parts of the pixels pipeline.
"""


import matplotlib.pyplot as plt
import seaborn as sns

from pixels import Experiment
from pixels.behaviours.leverpush import LeverPush, ActionLabels, Events


# Step 1: Load an experiment
#
# An experiment handles a group of mice that were trained in the same behaviour. It
# stores data and metadata for all included sessions belonging to the list of mice
# provided. The Experiment class is passed the mouse or list of mice, the class
# definition for the behaviour they were trained in (imported from pixels.behaviours),
# and the paths where it can find recording data (the folder containing 'raw', 'interim'
# and 'processed' folders) and training metadata.
#
myexp = Experiment(
    'MCos1497',
    LeverPush,
    '~/ardbeg/motor_choice/Npx_data',
    '~/ardbeg/CuedBehaviourAnalysis/Data/TrainingJSON',
)


# Step 2: Process raw data
#
# These methods each process a different type of raw data and save the output into the
# 'processed' folder. The outputs are all resampled to 1kHz, and are saved as:
#
#    - action labels (.npy)
#    - behavioural data (.h5)
#    - spike data (.h5)
#    - lfp data (.h5)
#

# This aligns, crops and downsamples behavioural data.
myexp.process_behaviour()

# This aligns, crops and downsamples LFP data.
myexp.process_lfp()

# This aligns, crops, and downsamples spike data.
myexp.process_spikes()

# This performs spike sorting, and ...
myexp.extract_spikes()

# This...
myexp.process_motion_tracking()


# Step 3: Run analyses
#
# Once all the data has been processed and converted into forms that are compatible with
# the rest of the data, we are ready to extract data organised by trials.
#

# behaviour
hit_trials = myexp.align_trials(ActionLabels.rewarded_push, Events.back_sensor_open, 'behaviour')[0]

plt.clf()
fig, axes = plt.subplots(6, 1, sharex=True)
tmp=hit_trials.columns.get_level_values('unit').unique()
for i in range(6):
    sns.lineplot(data=hit_trials[tmp[i]][1], estimator=None, style=None, ax=axes[i])
plt.show()

# lfp
hit_trials_lfp = myexp.align_trials(ActionLabels.rewarded_push, Events.back_sensor_open, 'lfp')[0]

plt.clf()
fig, axes = plt.subplots(10, 1, sharex=True)
for i in range(10):
    sns.lineplot(data=hit_trials_lfp[100 + i][8], estimator=None, style=None, ax=axes[i])
plt.show()
