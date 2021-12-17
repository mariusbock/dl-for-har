##################################################
# All functions related to plotting and visualizing data
##################################################
# Author: Kristof Van Laerhoven
# Email: kvl(at)eti.uni-siegen.de
##################################################

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def plot_imu_data(imu0, imu1, imu2, title):
    """
    Function to plot three IMU data streams

    :param imu0: data series
        First IMU data stream
    :param imu1: data series
        Second IMU data stream
    :param imu2: data series
        Third IMU data stream
    :param title: string
        Title of plot
    :return:
    """
    indices = []

    for i in range(0, len(imu1)):
        indices.append(i)
    fig, axes = plt.subplots(3, 1, sharex=True, sharey=False, figsize=(14, 8))
    x_patch = mpatches.Patch(color='orange', label='x-axis')
    y_patch = mpatches.Patch(color='green', label='y-axis')
    z_patch = mpatches.Patch(color='blue', label='z-axis')
    plt.legend(handles=[x_patch, y_patch, z_patch])
    axes[0].plot(imu0, lw=2, ls='-')
    axes[0].set_ylabel('original', fontsize=12)
    axes[1].plot(imu1, lw=2, ls='-')
    axes[1].set_ylabel('Normalized at once', fontsize=12)
    axes[1].set_ylim([-1, 1])
    axes[2].plot(imu2, lw=2, ls='-')
    axes[2].set_ylabel('Activity-Wise normalized', fontsize=12)
    axes[2].set_ylim([-1, 1])
    plt.xlabel('Index', fontsize=16)
    plt.suptitle(title, fontsize=19)
    plt.tight_layout()
    fig.subplots_adjust(top=0.88)
    plt.show()


def plot_data(data, title):
    """
    Function to plot data

    :param data: dataframe
        Data to be plotted
    :param title: string
        Title of plot
    :return:
    """
    indices = []

    for i in range(0, len(data)):
        indices.append(i)
    fig, axes = plt.subplots(1, 1, sharex=True, sharey=False, figsize=(14, 8))
    x_patch = mpatches.Patch(color='orange', label='x-axis')
    y_patch = mpatches.Patch(color='green', label='y-axis')
    z_patch = mpatches.Patch(color='blue', label='z-axis')
    plt.legend(handles=[x_patch, y_patch, z_patch])
    axes.plot(data, lw=2, ls='-')
    axes.set_ylabel('Acceleration (mg)', fontsize=12)
    plt.xlabel('Index', fontsize=16)
    plt.suptitle(title, fontsize=19)
    plt.tight_layout()
    fig.subplots_adjust(top=0.88)
    plt.show()
