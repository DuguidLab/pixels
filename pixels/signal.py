"""
This module provides functions that operate on signal data.
"""


import scipy.signal
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


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
    from_hz = float(from_hz)
    to_hz = float(to_hz)

    if from_hz == to_hz:
        return array

    elif from_hz > to_hz:
        up = 1
        factor = from_hz / to_hz
        down = factor
        while not down.is_integer():
            down += factor
            up += 1

    elif from_hz < to_hz:
        factor = to_hz / from_hz
        up = factor
        down = 1
        while not up.is_integer():
            up += factor
            down += 1

    return scipy.signal.resample_poly(array, up, down, padtype=padtype or 'minimum')


def binarise(data):
    """
    This normalises an array to between 0 and 1 and then makes all values below 0.5
    equal to 0 and all values above 0.5 to 1. The array is returned as np.int8 to save
    some memory when using large datasets.

    Parameters
    ----------
    data : numpy.ndarray or pandas.DataFrame
        If the data is a dataframe then each column will individually be binarised.

    """
    if isinstance(data, pd.DataFrame):
        for column in data.columns:
            data[column] = _binarise_real(data[column])
    else:
        data = _binarise_real(data)

    return data


def _binarise_real(data):
    data = (data - min(data)) / max(data)
    return (data > 0.5).astype(np.int8)


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
        half the length of array1. This cannot be greater than half the length of either
        array.

    plot : string, optional
        False (default), or a path specifying where to save a png of the best match.  If
        it already exists, it will be suffixed with the time.

    """
    if length is None:
        length = len(array1) // 2

    if len(array1) // 2 < length or len(array2) // 2 < length:
        raise Exception(f'Arrays must be at least twice the size of length parameter.')

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
            plot = plot.with_name(plot.stem + '_' + time.strftime('%y%m%d-%H%M%S') + '.png')
        fig, axes = plt.subplots(nrows=2, ncols=1)
        plot_length = min(length, 5000)
        if lag >= 0:
            axes[0].plot(array1[lag:lag + plot_length])
            axes[1].plot(array2[:plot_length])
        else:
            axes[0].plot(array1[:plot_length])
            axes[1].plot(array2[-lag:-lag + plot_length])
        fig.savefig(plot)
        print(f"Sync plot saved at: {plot}")

    return lag, match
