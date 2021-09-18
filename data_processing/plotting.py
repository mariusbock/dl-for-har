import matplotlib.pyplot as plt

def plot_imu_data(imu1, imu2, title):

    indices = []

    for i in range(0, len(imu1)):
        indices.append(i)
    fig, axes = plt.subplots(2, 1, sharex=True, sharey=False, figsize=(14, 8))

    axes[0].plot(imu1, lw=2, ls='-')
    axes[0].set_ylabel('Scaled at once', fontsize=16)
    axes[1].plot(imu2, lw=2, ls='-')
    axes[1].set_ylabel('Activity-Wise splitted and scaled', fontsize=16)
    plt.xlabel('Index', fontsize=16)
    plt.suptitle(title, fontsize=19)
    plt.tight_layout()
    fig.subplots_adjust(top=0.88)
    plt.show()