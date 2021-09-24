import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_imu_data(imu0, imu1, imu2, title):

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