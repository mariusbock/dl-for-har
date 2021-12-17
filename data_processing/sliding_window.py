##################################################
# All functions related to applying sliding window on a dataset
##################################################
# Author: Marius Bock
# Email: marius.bock(at)uni-siegen.de
##################################################

import numpy as np


def sliding_window_seconds(data, length_in_seconds=1, sampling_rate=50, overlap_ratio=None):
    """
    Return a sliding window measured in seconds over a data array.

    :param data: dataframe
        Input array, can be numpy or pandas dataframe
    :param length_in_seconds: int, default: 1
        Window length as seconds
    :param sampling_rate: int, default: 50
        Sampling rate in hertz as integer value
    :param overlap_ratio: int, default: None
        Overlap is meant as percentage and should be an integer value
    :return: tuple of windows and indices
    """
    windows = []
    indices = []
    curr = 0
    overlapping_elements = 0
    win_len = int(length_in_seconds * sampling_rate)
    if overlap_ratio is not None:
        overlapping_elements = int((overlap_ratio / 100) * win_len)
        if overlapping_elements >= win_len:
            print('Number of overlapping elements exceeds window size.')
            return
    while curr < len(data) - win_len:
        windows.append(data[curr:curr + win_len])
        indices.append([curr, curr + win_len])
        curr = curr + win_len - overlapping_elements
    return np.array(windows), np.array(indices)


def sliding_window_samples(data, samples_per_window, overlap_ratio):
    """
    Return a sliding window measured in number of samples over a data array.

    :param data: dataframe
        Input array, can be numpy or pandas dataframe
    :param samples_per_window: int
        Window length as number of samples per window
    :param overlap_ratio: int
        Overlap is meant as percentage and should be an integer value
    :return: dataframe, list
        Tuple of windows and indices
    """
    windows = []
    indices = []
    curr = 0
    win_len = int(samples_per_window)
    if overlap_ratio is not None:
        overlapping_elements = int((overlap_ratio / 100) * (win_len))
        if overlapping_elements >= win_len:
            print('Number of overlapping elements exceeds window size.')
            return
    while curr < len(data) - win_len:
        windows.append(data[curr:curr + win_len])
        indices.append([curr, curr + win_len])
        curr = curr + win_len - overlapping_elements
    try:
        result_windows = np.array(windows)
        result_indices = np.array(indices)
    except:
        result_windows = np.empty(shape=(len(windows), win_len, data.shape[1]), dtype=object)
        result_indices = np.array(indices)
        for i in range(0, len(windows)):
            result_windows[i] = windows[i]
            result_indices[i] = indices[i]
    return result_windows, result_indices


def apply_sliding_window(data_x, data_y, sliding_window_size, unit, sampling_rate, sliding_window_overlap):
    """
    Function which transforms a dataset into windows of a specific size and overlap.

    :param data_x: numpy float array
        Array containing the features (can be 2D)
    :param data_y: numpy float array
        Array containing the corresponding labels to the dataset (is 1D)
    :param sliding_window_size: integer or float
        Size of each window (either in seconds or units)
    :param unit: string, ['units', 'seconds']
        Unit in which the sliding window is measured
    :param sampling_rate: integer
        Number of hertz in which the dataset is sampled
    :param sliding_window_overlap: integer
        Amount of overlap between the sliding windows (measured in percentage, e.g. 20 is 20%)
    :return:
    """
    full_data = np.concatenate((data_x, data_y[:, None]), axis=1)
    output_x = None
    output_y = None

    for i, subject in enumerate(np.unique(full_data[:, 0])):
        subject_data = full_data[full_data[:, 0] == subject]
        subject_x, subject_y = subject_data[:, :-1], subject_data[:, -1]
        if unit == 'units':
            tmp_x, _ = sliding_window_samples(subject_x, sliding_window_size, sliding_window_overlap)
            tmp_y, _ = sliding_window_samples(subject_y, sliding_window_size, sliding_window_overlap)
        elif unit == 'seconds':
            tmp_x, _ = sliding_window_seconds(subject_x, sliding_window_size, sampling_rate, sliding_window_overlap)
            tmp_y, _ = sliding_window_seconds(subject_y, sliding_window_size, sampling_rate, sliding_window_overlap)
        if output_x is None:
            output_x = tmp_x
            output_y = tmp_y
        else:
            output_x = np.concatenate((output_x, tmp_x), axis=0)
            output_y = np.concatenate((output_y, tmp_y), axis=0)
    output_y = [[i[-1]] for i in output_y]
    return output_x, np.array(output_y).flatten()
