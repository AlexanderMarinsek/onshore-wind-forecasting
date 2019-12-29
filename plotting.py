from matplotlib import pyplot as plt
import matplotlib.patches as mpatches   # For legend
import matplotlib.lines as mlines
import numpy as np

# Combine default plot style (v2.0) with custom style
plt.style.use(["default", "./custom1.mplstyle"])

# Figure format
extension = "png"

# Custom settings only for scatter plots (not suitable for new MPL style)
scatter_marker = ['_', '_', '_', '_']
scatter_color = ['C0', 'C8', 'C2', 'C3']
scatter_lw = 2.5
scatter_alpha = 0.6


def plot_features(features):
    x = np.arange(0, len(features[:,0]), 1)

    plt.figure()

    plt.plot(x, features[:,0], label='sin_daily_tmpst')
    plt.plot(x, features[:,1], label='cos_daily_tmpst')
    plt.plot(x, features[:,2], label='sin_yearly_tmpst')
    plt.plot(x, features[:,3], label='cos_yearly_tmpst')

    plt.title('Timestamp')
    plt.legend()

    plt.savefig("fig-timestamp-1.png")
    #plt.show()
    plt.close()


def plot_forecast(test_labels_V, predictions_V, test_labels_U, predictions_U, figname):
    """
    Plot V and U predictions against real values in two subplots. Save plot.

    :param test_labels_V: Real wind V component values.
    :param predictions_V: Forecasted V values.
    :param test_labels_U: Real wind U component values.
    :param predictions_U: Forecasted U values.
    :param figname: string containing desired figure name without extension.

    :return: void.
    """

    x = np.arange(0, len(test_labels_V), 1)
    fig = plt.figure()

    ax1 = fig.add_subplot(121)
    ax1.plot(x, test_labels_V, label='actual')
    ax1.plot(x, predictions_V, label='predicted')
    ax1.set_title('V wind component')
    ax1.set_ylabel("Speed (m/s)")
    ax1.set_xlabel("Time (h)")
    ax1.legend()

    ax2 = fig.add_subplot(122)
    ax2.plot(x, test_labels_U, label='actual')
    ax2.plot(x, predictions_U, label='predicted')
    ax2.set_title('U wind component')
    ax2.set_ylabel("Speed (m/s)")
    ax2.set_xlabel("Time (h)")
    ax2.legend()

    fig.savefig("%s.%s" % (figname, extension))
    #fig.show()
    plt.close()


def plot_scores (x, x_label, bs_arr, ss_arr, bs_labels, ss_labels, figname):
    """
    Plot forecast scores (Brier and secondary). Save plot.

    :param x: List of X-axis values.
    :param bs_arr: 2D numpy array of Brier scores.
    :param ss_arr: 2D numpy array of secondary scores.
    :param bs_labels: List of Brier score labels.
    :param ss_labels: List of secondary score values.
    :param figname: string containing desired figure name without extension.

    :return: void.
    """

    fig = plt.figure()

    ax1 = fig.add_subplot(121)
    for i in range (0, bs_arr.shape[0]):
        ax1.scatter(x, bs_arr[i,:], label=bs_labels[i], marker=scatter_marker[i],
            color=scatter_color[i], alpha=scatter_alpha, lw=scatter_lw)

    ax1.set_title('Brier Score')
    ax1.set_xlabel(x_label)
    ax1.set_ylabel('Score')

    lns = [mlines.Line2D([], [], color=scatter_color[i], linestyle='None',
        marker=scatter_marker[i], ) for i in range(0, 4)]
    labs = bs_labels
    ax1.legend(lns, labs)

    ax2 = fig.add_subplot(122)
    for i in range (0, ss_arr.shape[0]):
        ax2.scatter(x, ss_arr[i,:], label=ss_labels[i], marker=scatter_marker[i],
            color=scatter_color[i], alpha=scatter_alpha, lw=scatter_lw)

    ax2.set_title('Secondary Score')
    ax2.set_xlabel(x_label)
    ax2.set_ylabel('Score')

    lns = [mlines.Line2D([], [], color=scatter_color[i], linestyle='None',
        marker=scatter_marker[i], ) for i in range(0, 4)]
    labs = ss_labels
    ax2.legend(lns, labs)

    fig.savefig("%s.%s" % (figname, extension))
    # fig.show()
    plt.close()


def plot_scores_all(x, x_label, bs_arr, ss_arr, bs_labels, ss_labels, figname):
    """
    Plot multiple forecast scores (Brier and secondary). Save plot.

    :param x: List of X-axis values.
    :param bs_arr: 3D numpy array of Brier scores. (0th dimension is separator)
    :param ss_arr: 3D numpy array of secondary scores.
    :param bs_labels: List of Brier score labels.
    :param ss_labels: List of secondary score values.
    :param figname: string containing desired figure name without extension.

    :return: void.
    """

    fig = plt.figure()

    ax1 = fig.add_subplot(121)
    for i in range(0, int(bs_arr.shape[0])):
        ax1.scatter(x, bs_arr[i,0,:], marker=scatter_marker[0], color=scatter_color[0],
           alpha=scatter_alpha, lw=scatter_lw)
        ax1.scatter(x, bs_arr[i,1,:], marker=scatter_marker[1], color=scatter_color[1],
           alpha=scatter_alpha, lw=scatter_lw)
        ax1.scatter(x, bs_arr[i,2,:], marker=scatter_marker[2], color=scatter_color[2],
           alpha=scatter_alpha, lw=scatter_lw)
        ax1.scatter(x, bs_arr[i,3,:], marker=scatter_marker[3], color=scatter_color[3],
           alpha=scatter_alpha, lw=scatter_lw)

    ax1.set_title('Brier Score')
    ax1.set_xlabel(x_label)
    ax1.set_ylabel('Score')

    #lns = [mpatches.Patch(color=scatter_color[i]) for i in range(0, 4)]
    #lns = [mlines.Line2D([], [], color=scatter_color[i]) for i in range(0, 4)]
    lns = [mlines.Line2D([], [], color=scatter_color[i], linestyle='None',
        marker=scatter_marker[3], ) for i in range(0, 4)]
    labs = bs_labels
    ax1.legend(lns, labs)

    ax2 = fig.add_subplot(122)
    for i in range(0, int(ss_arr.shape[0])):
        ax2.scatter(x, ss_arr[i,0,:], marker=scatter_marker[0], color=scatter_color[0],
            alpha=scatter_alpha, lw=scatter_lw)
        ax2.scatter(x, ss_arr[i,1,:], marker=scatter_marker[1], color=scatter_color[1],
            alpha=scatter_alpha, lw=scatter_lw)
        ax2.scatter(x, ss_arr[i,2,:], marker=scatter_marker[2], color=scatter_color[2],
            alpha=scatter_alpha, lw=scatter_lw)
        ax2.scatter(x, ss_arr[i,3,:], marker=scatter_marker[3], color=scatter_color[3],
            alpha=scatter_alpha, lw=scatter_lw)

    ax2.set_title('Secondary Score')
    ax2.set_xlabel(x_label)
    ax2.set_ylabel('Score')

    lns = [mlines.Line2D([], [], color=scatter_color[i], linestyle='None',
        marker=scatter_marker[3], ) for i in range(0, 4)]
    labs = ss_labels
    ax2.legend(lns, labs)

    fig.savefig("%s.%s" % (figname, extension))
    # fig.show()
    plt.close()


def plot_score_comparison(x, x_label, bs_arr, ss_arr, bs_labels, ss_labels, figname):
    """
    Plot multiple forecast scores (Brier and secondary). Save plot.

    :param x: List of X-axis values.
    :param bs_arr: 3D numpy array of Brier scores. (0th dimension is separator)
    :param ss_arr: 3D numpy array of secondary scores.
    :param labels: List of score labels.
    :param figname: string containing desired figure name without extension.

    :return: void.
    """

    _marker = ['o','^','x','s']

    fig = plt.figure()

    ax1 = fig.add_subplot(121)
    for i in range(0, int(bs_arr.shape[0])):    # Iterate forecasts
        for j in range(0, int(bs_arr.shape[2])):    # Iterate dates
            ax1.scatter(x, bs_arr[i, 0, j], marker=_marker[i],
                        color=scatter_color[0], alpha=scatter_alpha)
            ax1.scatter(x, bs_arr[i, 1, j], marker=_marker[i],
                        color=scatter_color[1], alpha=scatter_alpha)
            ax1.scatter(x, bs_arr[i, 2, j], marker=_marker[i],
                        color=scatter_color[2], alpha=scatter_alpha)
            ax1.scatter(x, bs_arr[i, 3, j], marker=_marker[i],
                        color=scatter_color[3], alpha=scatter_alpha)

    ax1.set_title('Brier Score')
    ax1.set_xlabel(x_label)
    ax1.set_ylabel('Score')

    lns = [mlines.Line2D([], [], color='C7', linestyle='None', marker=_marker[i],
        alpha=scatter_alpha) for i in range(0, int(bs_arr.shape[0]))]
    labs = bs_labels
    ax1.legend(lns, labs)

    ax2 = fig.add_subplot(122)
    for i in range(0, int(ss_arr.shape[0])):    # Iterate forecasts
        for j in range(0, int(bs_arr.shape[2])):    # Iterate dates
            ax2.scatter(x, ss_arr[i, 0, j], marker=_marker[i],
                        color=scatter_color[0], alpha=scatter_alpha)
            ax2.scatter(x, ss_arr[i, 1, j], marker=_marker[i],
                        color=scatter_color[1], alpha=scatter_alpha)
            ax2.scatter(x, ss_arr[i, 2, j], marker=_marker[i],
                        color=scatter_color[2], alpha=scatter_alpha)
            ax2.scatter(x, ss_arr[i, 3, j], marker=_marker[i],
                        color=scatter_color[3], alpha=scatter_alpha)

    ax2.set_title('Secondary Score')
    ax2.set_xlabel(x_label)
    ax2.set_ylabel('Score')

    lns = [mlines.Line2D([], [], color=scatter_color[i], linestyle='None',
        marker=scatter_marker[3], ) for i in range(0, 4)]
    labs = ss_labels
    ax2.legend(lns, labs)

    fig.savefig("%s.%s" % (figname, extension))
    # fig.show()
    plt.close()