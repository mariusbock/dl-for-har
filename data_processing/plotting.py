import matplotlib
try:
    matplotlib.use('TkAgg')
except:
    pass
import matplotlib.pyplot as plt

def plot_imu_data(imu, title, creation_time, sampling_rate, time_as_indices=True):
    """
    Method written by kvl for reading out the files that is written by the imu itself
    :param acc:
    :param gyr:
    :param mag:
    :param title:
    :param creation_time:
    :param sampling_rate:
    :param time_as_indices:
    """
    # acc = imu[:, 1:4]
    # gyr = imu[:, 4:7]
    # mag = imu[:, 7:10]

    acc = imu[:, 0:3]
    # gyr = imu[:, 3:6]
    # mag = imu[:, 6:9]

    indices = []

    for i in range(0, len(acc)):
        indices.append(i)
    fig, axes = plt.subplots(3, 1, sharex=True, sharey=False, figsize=(14, 8))

    axes[0].plot(acc, lw=2, ls='-')
    axes[0].set_ylabel('Acceleration (mg)', fontsize=16)
    # axes[0].set_ylim((-10, 10))
    # axes[1].plot(gyr, lw=2, ls='-')
    # axes[1].set_ylabel('Gyroscope (rad/sec)', fontsize=16)
    # axes[2].plot(mag, lw=2, ls='-')
    # axes[2].set_ylabel('Magnetation (rad/sec)', fontsize=16)
    plt.xlabel('Index', fontsize=16)
    # plt.gcf().autofmt_xdate()
    plt.suptitle(title, fontsize=19)
    plt.tight_layout()
    fig.subplots_adjust(top=0.88)
    plt.show()