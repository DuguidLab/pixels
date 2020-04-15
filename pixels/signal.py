"""
This module provides functions that operate on signal data.
"""


import scipy.signal
import time
import numpy as np
import matplotlib.pyplot as plt


def resample(array, from_hz, to_hz, padtype=None):
    """
    Resample an array from one sampling rate to another.

    Parameters
    ----------
    array : ndarray, Series or similar
        The data to be resampled.

    from_hz : int or float
        The starting frequency of the data.

    sample_rate : int or float, optional
        The resulting sample rate.

    padtype : string, optional
        How to deal with end effects. Default: 'minimum', where the resampling filter
        pretends the ends are extended using the array's minimum.

    """
    if from_hz == to_hz:
        return array
    elif from_hz > to_hz:
        up = 1
        down = from_hz / to_hz
    elif from_hz < to_hz:
        up = to_hz / from_hz
        down = 1

    array = scipy.signal.resample_poly(
        array, up, down, padtype=padtype or 'minimum',
    )
    return array


def find_sync_lag(array1, array2, length=None, plot=False):
    """
    Find the lag between two arrays where they have the greatest number of the same
    values.

    Parameters
    ----------
    array1 : array, Series or similar
        The first array. A positive result indicates that this array has leading data
        not present in the second array. e.g. if lag == 5 then array2 starts on the 5th
        index of array1.

    array2 : array, Series or similar
        The array to look for in the first.

    length : int, optional
        The distance to traverse the two arrays looking for the best match. Default:
        length of array1.

    plot : string, optional
        False (default), or a path specifying where to save a png of the best match.  If
        it already exists, it will be suffixed with the time.

    """
    if length is None:
        length = len(array1)

    sync_p = []
    for i in range(length):
        matches = np.count_nonzero(array1[i:i + length] == array2[:length])
        sync_p.append(100 * matches / length)
    match_p = max(sync_p)
    lag_p = sync_p.index(match_p)

    sync_n = []
    for i in range(length):
        matches = np.count_nonzero(array2[i:i + length] == array1[:length])
        sync_n.append(100 * matches / length)
    match_n = max(sync_n)
    lag_n = sync_n.index(match_n)
    
    if match_p > match_n:
        lag = lag_p
        match = match_p
    else:
        lag = lag_n
        match = match_n

    if plot:
        plot = Path(plot)
        if plot.exists():
            plot = plot.with_name(plot.stem + time.strftime('%y%m%d-%H%M%S') + '.png')
        fig, axes = plt.subplots(nrows=2, ncols=1)
        raise Exception
        # CONTINUE HERE WITH PLOTTING THE TWO ALIGNED SIGNALS ON SUBPLOTS
        axes[0].plot()

    return lag, match
