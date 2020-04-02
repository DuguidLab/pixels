"""
This module contains helper functions for reading and writing files.
"""


import os


def read_tdms(path):
    tdms_file = TdmsFile(path)
    group_name = [i for i in tdms_file.groups()]

    channel_data = {}
    for channel in tdms_file.groups():
        channel_data[channel.name] = channel['0'].data

    return channel_data
