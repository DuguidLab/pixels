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
    'MCos1497',  # This can be a list
    LeverPush,
    '~/path/to/data',
    '~/path/to/metadata',
)


# Step 2: Process raw data
#
# These methods each process a different type of raw data and save the output into the
# 'processed' folder. The outputs are all resampled to 1kHz, and are saved as:
#
#    - action labels (.npy)
#    - behavioural data (.h5)
#    - LFP data (.h5)
#    - spike data (.h5)
#    - sorted spikes (TODO)
#

# This aligns, crops and downsamples behavioural data.
myexp.process_behaviour()

# This aligns, crops and downsamples LFP data.
myexp.process_lfp()

# This aligns, crops, and downsamples spike data.
myexp.process_spikes()

# This runs the spike sorting algorithm and outputs the results in a form usable by phy.
myexp.sort_spikes()

# TODO: deeplabcut is not yet handled

# This extracts posture coordinates from TDMS videos using DeepLabCut
config = '/path/to/this/behaviours/deeplabcut/config.yaml'
myexp.process_motion_tracking(config)
# If you also want to output labelled videos, pass this keyword arg:
myexp.process_motion_tracking(config, create_labelled_video=True):
# This method will convert the videos from TDMS to AVI before running them through DLC.
# If you just want the AVI videos without the DLC, you can do so directly:
myexp.extract_videos()


# Step 3: Run exploratory analyses
#
# Once all the data has been processed and converted into forms that are compatible with
# the rest of the data, we are ready to explore the data.

# We can access individual sessions by indexing into the Experiment:
myexp[0]

# And from a session we can get continuous processed data:
spike_data = myexp[0].get_spike_data()
lfp_data = myexp[0].get_lfp_data()
behavioural_data = myexp[0].get_behavioural_data()

# Data can be loading as trial-aligned data using the Experiment.align_trials method.
# This returns a multidimensional pandas DataFrame containing the desired data organised
# by session, unit, and trial.
#
# Here are some examples of how data can be plotted:
#

# Plotting all behavioural data channels for session 1, trial 3
hits = myexp.align_trials(
    ActionLabels.rewarded_push,  # This selects which trials we want
    Events.back_sensor_open,  # This selects what event we want them aligned to 
    'behavioural'  # And this selects what kind of data we want
)

plt.figure()
fig, axes = plt.subplots(6, 1, sharex=True)
channels = hit_trials.columns.get_level_values('unit').unique()
trial = 3
session = 0
for i in range(6):
    chan_name = channels[i]
    sns.lineplot(
        data=hits[session][chan_name][trial],
        estimator=None,
        style=None,
        ax=axes[i]
    )
plt.show()

# Plotting spike data from session 1, trial 8, units 101 to 110
hits = myexp.align_trials(
    ActionLabels.rewarded_push,
    Events.back_sensor_open,
    'spikes'
)

plt.figure()
fig, axes = plt.subplots(10, 1, sharex=True)
trial = 8
session = 0
for i in range(10):
    sns.lineplot(
        data=hits[101 + i][trial],
        estimator=None,
        style=None,
        ax=axes[i]
    )
plt.show()
