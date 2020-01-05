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


def plot_forecast(labels, forecast, figname):
    """
    Plot V and U predictions against real values in two subplots. Save plot.

    :param figname: string containing desired figure name without extension.

    :return: void.
    """

    x = np.arange(0, len(labels), 1)
    fig = plt.figure()

    ax1 = fig.add_subplot(111)
    ax1.plot(x, labels, label='actual')
    ax1.plot(x, forecast, label='forecast')
    ax1.set_title("Abs. Wind Speed")
    ax1.set_ylabel("Speed (m/s)")
    ax1.set_xlabel("Time (h)")
    ax1.legend()

    fig.savefig("%s.%s" % (figname, extension))
    #fig.show()
    plt.close()


def plot_comparison_forecast(lab_for_2d, names, figname):

    # TODO: Use 'k' index as 3rd dimension
    # TODO: Prettyfy plot (test it in a separate test module)

    x = np.arange(0, lab_for_2d.shape[1], 1)
    fig = plt.figure()

    ax1 = fig.add_subplot(111)

    i = k = 0
    while (i<lab_for_2d.shape[0]):
        ax1.plot(x, lab_for_2d[i,:], label='actual %s' % names[k])
        ax1.plot(x, lab_for_2d[i+1,:], label='forecast %s' % names[k])
        k += 1; i += 2

    ax1.set_title("Abs. Wind Speed")
    ax1.set_ylabel("Speed (m/s)")
    ax1.set_xlabel("Time (h)")
    ax1.legend()

    fig.savefig("%s.%s" % (figname, extension))
    # fig.show()
    plt.close()


def plot_power_forecast(power_2d, names, figname):

    # TODO: Add 2nd subplot - WT curve
    # TODO: Prettyfy plot (test it in a separate test module)

    x = np.arange(0, power_2d.shape[1], 1)
    fig = plt.figure()

    ax1 = fig.add_subplot(111)

    i = k = 0
    while (i<power_2d.shape[0]):
        ax1.plot(x, power_2d[i,:], label='actual %s' % names[k])
        ax1.plot(x, power_2d[i+1,:], label='forecast %s' % names[k])
        k += 1; i += 2

    ax1.set_title("WT Power Production")
    ax1.set_ylabel("Power (kW)")
    ax1.set_xlabel("Time (h)")
    ax1.legend()

    fig.savefig("%s.%s" % (figname, extension))
    # fig.show()
    plt.close()


def plot_rf_optimization( data, figname ):
    x = data[:,0]   # M
    y = data[:,2]   # G

    fig = plt.figure()

    # TODO: Calculate size relative to data[i, -2]    (rmse)
    # TODO: Calculate colors relative to data[i, -1]  (time)
    # TODO: Prettyfy plot (test it in a separate test module)

    ax1 = fig.add_subplot(111)
    ax1.scatter(x, y)
    ax1.set_title("RF optimization")
    ax1.set_xlabel("M")
    ax1.set_ylabel("G")

    fig.savefig("%s.%s" % (figname, extension))
    # fig.show()
    plt.close()


def plot_n_test( data, figname ):
    x = data[:,1]   # N

    fig = plt.figure()

    # TODO: Add legend without gray background lines
    # TODO: Prettyfy plot (test it in a separate test module)

    ax1 = fig.add_subplot(111)

    ax1.plot(x, data[:, 6], color='C0', linestyle="-", zorder=3, label="RMSE")

    ax1.plot(x, data[:, 5], color='C7', linestyle="--", zorder=2, label="MSE")
    ax1.plot(x, data[:, 5], color='black', linestyle="-", zorder=1)

    #ax1.plot(x, data[:, 4], color='C7', linestyle="-.", zorder=2, label="MAPE")
    #ax1.plot(x, data[:, 4], color='black', linestyle="-", zorder=1)

    ax1.plot(x, data[:, 3], color='C7', linestyle=":", zorder=2, label="MAE")
    ax1.plot(x, data[:, 3], color='black', linestyle="-", zorder=1)

    ax1.set_title("N Variable Test")
    ax1.set_xlabel("N")
    ax1.set_ylabel("Error")

    fig.savefig("%s.%s" % (figname, extension))
    # fig.show()
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