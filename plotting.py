from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Combine default plot style (v2.0) with custom style
plt.style.use(["default", "./custom1.mplstyle"])

# Figure format
extension = "png"

# Custom settings only for scatter plots (not suitable for new MPL style)
scatter_marker = ['_', '_', '_', '_']
scatter_color = ['C0', 'C9', 'C9', 'C3']
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
    # ax1.set_ylabel("Speed ($ms^{-1}$)")
    ax1.set_ylabel(r"Speed $(\frac{m}{s})$")
    ax1.set_xlabel("Time (h)")
    ax1.legend()

    fig.savefig("%s.%s" % (figname, extension))
    #fig.show()
    plt.close()


def plot_comparison_forecast(lab_for_2d, names, figname):

    # TODO: DONE Use 'k' index as 3rd dimension
    # TODO: Prettyfy plot (test it in a separate test module)

    # Add ERA5-L model/forecast name
    names.insert(0,"ERA5-L")

    # Plot colors
    colors = ["C0", "C1", "C9", "C3"]

    x = np.arange(0, lab_for_2d.shape[1], 1)
    Y = np.array([lab_for_2d[i*2+1,:] for i in range(0,int(lab_for_2d.shape[0]/2))])

    # Add ERA5-L data
    Y = np.concatenate((lab_for_2d[0,:].reshape(1,-1), Y), axis=0)

    # Polygon list
    verts = []

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Interpolation step
    # step = 0.1

    for i in range(0, Y.shape[0]):

        # Get Y val
        y = Y[i]

        # Plot forecasts on back plane
        ax.plot( x, [5]*x.shape[0], y, lw=1.5, ls='--', color=colors[i], alpha=0.4)
        # Plot 3D lines
        ax.plot( x, [i+1]*x.shape[0], y, lw=1.5, color=colors[i], alpha=0.8, label=names[i])
        # Append polygon associated with current line
        verts.append(polygon_under_graph(x, y))

        # Interpolate values
        # f = interpolate.interp1d(x, y)
        # x = np.arange(0, 23+step, 0.1)
        # y = f(x)

        # # Design linear segments
        # points = np.array([[i]*int(23/step+1), x, y]).T.reshape(-1, 1, 3)
        # segments = np.concatenate([points[:-1], points[1:]], axis=1)
        #
        # # Get minimum and maximum value for plot's color scale
        # c_min = y.min()
        # c_max = y.max()
        # # Normalize (map) color scale to data set
        # norm = plt.Normalize(c_min, c_max)
        # # Apply color scale to line segments and get line collection for plotting
        # lc = Line3DCollection(segments, cmap='viridis', norm=norm)
        # lc.set_array(y)
        # line1 = ax.add_collection(lc)
        # ax.add_collection(lc)

    # Append polygon associated with current line
    poly = PolyCollection(verts, facecolors=colors[:Y.shape[0]], alpha=0.5)
    ax.add_collection3d(poly, zs=range(1,Y.shape[0]+1), zdir='y')

    ax.set_xlim(0, 24)
    ax.set_ylim(0, 5)
    ax.set_zlim(0, Y.max())

    ax.set_title("Forecast comparison")
    ax.set_xlabel("Time (h)", labelpad=30)
    ax.set_ylabel("Model", labelpad=25)
    ax.set_zlabel(r"Speed $(\frac{m}{s})$", labelpad=10)
    ax.legend()

    ax.set_xticks([i*4 for i in range(0,6) ])

    ax.set_yticks(range(0,5))
    names.insert(0,"")
    ax.set_yticklabels(names)

    ax.xaxis.set_rotate_label(0)
    ax.yaxis.set_rotate_label(0)
    ax.zaxis.set_rotate_label(90)

    plt.tight_layout()
    #plt.show()
    plt.savefig("%s.%s" % (figname, extension))
    plt.close()


def polygon_under_graph(xlist, ylist):
    """
    Construct the vertex list which defines the polygon filling the space under
    the (xlist, ylist) line graph.  Assumes the xs are in ascending order.
    """
    return [(xlist[0], 0.), *zip(xlist, ylist), (xlist[-1], 0.)]


def plot_power_forecast(power_2d, ext_2d, p_curve, names, figname):

    # TODO: Add 2nd subplot - WT curve
    # TODO: Prettyfy plot (test it in a separate test module)

    # Add ERA5-L model/forecast name
    # names.insert(0,"ERA5-L")

    # Plot colors
    # colors = ["C0", "C1", "C2", "C3"]
    colors = ["C1", "C9", "C3"]

    # Get Xs
    x = np.arange(0, power_2d.shape[1], 1)
    # Get power forecasts
    Yp = np.array(
        [power_2d[i * 2 + 1, :] for i in range(0, int(power_2d.shape[0] / 2))])
    # Get extrapolated wind speed forecasts
    Ye = np.array(
        [ext_2d[i * 2 + 1, :] for i in range(0, int(ext_2d.shape[0] / 2))])

    # Add ERA5-L data
    # Yp = np.concatenate((power_2d[0, :].reshape(1, -1), Yp), axis=0)
    # Ye = np.concatenate((ext_2d[0, :].reshape(1, -1), Ye), axis=0)

    fig = plt.figure()

    # Define uneven subplot grid
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])

    # Plot power forecasts
    ax1 = fig.add_subplot(gs[0])

    for i in range(0, Yp.shape[0]):
        y = Yp[i, :]
        ax1.plot(x, y, ls="-", color=colors[i], label=names[i], alpha=0.85)

    ax1.set_title("WT Power Production")
    ax1.set_ylabel("Power (kW)")
    ax1.set_xlabel("Time (h)")
    ax1.legend()

    # Plot power curve (duplicate Y axis)
    ax2 = fig.add_subplot(gs[1], sharey=ax1)

    # First plot the power curve
    ax2.plot(p_curve[0, :], p_curve[1, :], ls="--", color="C7")

    for i in range(0, Ye.shape[0]):
        data = cut_power_curve(p_curve, Ye[i, :].max())
        ax2.fill_between(data[0, :], data[1, :], color=colors[i], alpha=0.3,
                         label=names[i])

    ax2.set_title("WT Power Curve")
    # ax2.set_ylabel("Power (kW)")
    ax2.set_xlabel(r"Speed $(\frac{m}{s})$")
    # ax2.legend()

    fig.savefig("%s.%s" % (figname, extension))
    # plt.show()
    plt.close()


def cut_power_curve (p_curve, max_speed):

    # for i in range(0, p_curve.shape[1]-1):
    for i in range(0, p_curve.shape[1]):
        if p_curve[0,i] > max_speed:
            break

    i+=1

    return p_curve[:,:i]


def plot_var_tuning( model_name, data, figname ):

    # TODO: DONE Calculate size relative to data[i, -2]    (rmse)
    # TODO: DONE Calculate colors relative to data[i, -1]  (time)
    # TODO: Prettyfy plot (test it in a separate test module)

    # colormap="viridis"
    # colormap="YlOrRd"
    # colormap="YlGnBu"
    colormap="bone_r"

    x = data[:,0]   # M
    y = data[:,2]   # G
    e = data[:,-4]  # RMSE
    t = data[:,-3]  # total time

    t_min = np.min(t, axis=0)
    t_max = np.max(t, axis=0)
    scale = [20000.0 * (_t-t_min*0.8) / t_max for _t in t]

    plt.figure()

    plt.scatter(x, y, s=scale, c=e, cmap=colormap, linewidths=2, edgecolors="#555555", zorder=2)
    plt.colorbar()

    plt.grid(zorder=2)

    plt.title("%s Tuning" % model_name)
    plt.xlabel("M")
    plt.ylabel("Grid")

    plt.xticks(x)
    # plt.yticks(y)
    plt.yticks( [0,3], [ "1x1", "3x3" ] )

    plt.xlim(x.min()-1, x.max()+1)
    plt.ylim(y.min()-1, y.max()+1)

    plt.savefig("%s.%s" % (figname, extension))
    # plt.show()
    plt.close()


def plot_n_eval( data, figname ):
    x = data[:,1]   # N

    fig = plt.figure()

    # TODO: Add legend without gray background lines
    # TODO: Prettyfy plot (test it in a separate test module)

    ax1 = fig.add_subplot(111)

    ax1.plot(x, data[:, 6], color='C0', linestyle="-", zorder=3, label="RMSE")

    ax1.plot(x, data[:, 5], color='black', linestyle="--", zorder=2, label="MSE")
    ax1.plot(x, data[:, 5], color='#bbbbbb', linestyle="-", zorder=1)

    MAPE_norm = data[:, 4] / data[:, 4].max()

    ax1.plot(x, MAPE_norm, color='black', linestyle="-.", zorder=2, label="nMAPE")
    ax1.plot(x, MAPE_norm, color='#bbbbbb', linestyle="-", zorder=1)

    ax1.plot(x, data[:, 3], color='black', linestyle=":", zorder=2, label="MAE")
    ax1.plot(x, data[:, 3], color='#bbbbbb', linestyle="-", zorder=1)

    ax1.legend()

    ax1.set_title("N Variable Evaluation")
    ax1.set_xlabel("N")
    ax1.set_ylabel("Error")

    # ax1.set_xlim(x.min(), x.max())

    fig.savefig("%s.%s" % (figname, extension))
    # fig.show()
    plt.close()

#
# def plot_scores (x, x_label, bs_arr, ss_arr, bs_labels, ss_labels, figname):
#     """
#     Plot forecast scores (Brier and secondary). Save plot.
#
#     :param x: List of X-axis values.
#     :param bs_arr: 2D numpy array of Brier scores.
#     :param ss_arr: 2D numpy array of secondary scores.
#     :param bs_labels: List of Brier score labels.
#     :param ss_labels: List of secondary score values.
#     :param figname: string containing desired figure name without extension.
#
#     :return: void.
#     """
#
#     fig = plt.figure()
#
#     ax1 = fig.add_subplot(121)
#     for i in range (0, bs_arr.shape[0]):
#         ax1.scatter(x, bs_arr[i,:], label=bs_labels[i], marker=scatter_marker[i],
#             color=scatter_color[i], alpha=scatter_alpha, lw=scatter_lw)
#
#     ax1.set_title('Brier Score')
#     ax1.set_xlabel(x_label)
#     ax1.set_ylabel('Score')
#
#     lns = [mlines.Line2D([], [], color=scatter_color[i], linestyle='None',
#         marker=scatter_marker[i], ) for i in range(0, 4)]
#     labs = bs_labels
#     ax1.legend(lns, labs)
#
#     ax2 = fig.add_subplot(122)
#     for i in range (0, ss_arr.shape[0]):
#         ax2.scatter(x, ss_arr[i,:], label=ss_labels[i], marker=scatter_marker[i],
#             color=scatter_color[i], alpha=scatter_alpha, lw=scatter_lw)
#
#     ax2.set_title('Secondary Score')
#     ax2.set_xlabel(x_label)
#     ax2.set_ylabel('Score')
#
#     lns = [mlines.Line2D([], [], color=scatter_color[i], linestyle='None',
#         marker=scatter_marker[i], ) for i in range(0, 4)]
#     labs = ss_labels
#     ax2.legend(lns, labs)
#
#     fig.savefig("%s.%s" % (figname, extension))
#     # fig.show()
#     plt.close()
#
#
# def plot_scores_all(x, x_label, bs_arr, ss_arr, bs_labels, ss_labels, figname):
#     """
#     Plot multiple forecast scores (Brier and secondary). Save plot.
#
#     :param x: List of X-axis values.
#     :param bs_arr: 3D numpy array of Brier scores. (0th dimension is separator)
#     :param ss_arr: 3D numpy array of secondary scores.
#     :param bs_labels: List of Brier score labels.
#     :param ss_labels: List of secondary score values.
#     :param figname: string containing desired figure name without extension.
#
#     :return: void.
#     """
#
#     fig = plt.figure()
#
#     ax1 = fig.add_subplot(121)
#     for i in range(0, int(bs_arr.shape[0])):
#         ax1.scatter(x, bs_arr[i,0,:], marker=scatter_marker[0], color=scatter_color[0],
#            alpha=scatter_alpha, lw=scatter_lw)
#         ax1.scatter(x, bs_arr[i,1,:], marker=scatter_marker[1], color=scatter_color[1],
#            alpha=scatter_alpha, lw=scatter_lw)
#         ax1.scatter(x, bs_arr[i,2,:], marker=scatter_marker[2], color=scatter_color[2],
#            alpha=scatter_alpha, lw=scatter_lw)
#         ax1.scatter(x, bs_arr[i,3,:], marker=scatter_marker[3], color=scatter_color[3],
#            alpha=scatter_alpha, lw=scatter_lw)
#
#     ax1.set_title('Brier Score')
#     ax1.set_xlabel(x_label)
#     ax1.set_ylabel('Score')
#
#     #lns = [mpatches.Patch(color=scatter_color[i]) for i in range(0, 4)]
#     #lns = [mlines.Line2D([], [], color=scatter_color[i]) for i in range(0, 4)]
#     lns = [mlines.Line2D([], [], color=scatter_color[i], linestyle='None',
#         marker=scatter_marker[3], ) for i in range(0, 4)]
#     labs = bs_labels
#     ax1.legend(lns, labs)
#
#     ax2 = fig.add_subplot(122)
#     for i in range(0, int(ss_arr.shape[0])):
#         ax2.scatter(x, ss_arr[i,0,:], marker=scatter_marker[0], color=scatter_color[0],
#             alpha=scatter_alpha, lw=scatter_lw)
#         ax2.scatter(x, ss_arr[i,1,:], marker=scatter_marker[1], color=scatter_color[1],
#             alpha=scatter_alpha, lw=scatter_lw)
#         ax2.scatter(x, ss_arr[i,2,:], marker=scatter_marker[2], color=scatter_color[2],
#             alpha=scatter_alpha, lw=scatter_lw)
#         ax2.scatter(x, ss_arr[i,3,:], marker=scatter_marker[3], color=scatter_color[3],
#             alpha=scatter_alpha, lw=scatter_lw)
#
#     ax2.set_title('Secondary Score')
#     ax2.set_xlabel(x_label)
#     ax2.set_ylabel('Score')
#
#     lns = [mlines.Line2D([], [], color=scatter_color[i], linestyle='None',
#         marker=scatter_marker[3], ) for i in range(0, 4)]
#     labs = ss_labels
#     ax2.legend(lns, labs)
#
#     fig.savefig("%s.%s" % (figname, extension))
#     # fig.show()
#     plt.close()
#
#
# def plot_score_comparison(x, x_label, bs_arr, ss_arr, bs_labels, ss_labels, figname):
#     """
#     Plot multiple forecast scores (Brier and secondary). Save plot.
#
#     :param x: List of X-axis values.
#     :param bs_arr: 3D numpy array of Brier scores. (0th dimension is separator)
#     :param ss_arr: 3D numpy array of secondary scores.
#     :param labels: List of score labels.
#     :param figname: string containing desired figure name without extension.
#
#     :return: void.
#     """
#
#     _marker = ['o','^','x','s']
#
#     fig = plt.figure()
#
#     ax1 = fig.add_subplot(121)
#     for i in range(0, int(bs_arr.shape[0])):    # Iterate forecasts
#         for j in range(0, int(bs_arr.shape[2])):    # Iterate dates
#             ax1.scatter(x, bs_arr[i, 0, j], marker=_marker[i],
#                         color=scatter_color[0], alpha=scatter_alpha)
#             ax1.scatter(x, bs_arr[i, 1, j], marker=_marker[i],
#                         color=scatter_color[1], alpha=scatter_alpha)
#             ax1.scatter(x, bs_arr[i, 2, j], marker=_marker[i],
#                         color=scatter_color[2], alpha=scatter_alpha)
#             ax1.scatter(x, bs_arr[i, 3, j], marker=_marker[i],
#                         color=scatter_color[3], alpha=scatter_alpha)
#
#     ax1.set_title('Brier Score')
#     ax1.set_xlabel(x_label)
#     ax1.set_ylabel('Score')
#
#     lns = [mlines.Line2D([], [], color='C7', linestyle='None', marker=_marker[i],
#         alpha=scatter_alpha) for i in range(0, int(bs_arr.shape[0]))]
#     labs = bs_labels
#     ax1.legend(lns, labs)
#
#     ax2 = fig.add_subplot(122)
#     for i in range(0, int(ss_arr.shape[0])):    # Iterate forecasts
#         for j in range(0, int(bs_arr.shape[2])):    # Iterate dates
#             ax2.scatter(x, ss_arr[i, 0, j], marker=_marker[i],
#                         color=scatter_color[0], alpha=scatter_alpha)
#             ax2.scatter(x, ss_arr[i, 1, j], marker=_marker[i],
#                         color=scatter_color[1], alpha=scatter_alpha)
#             ax2.scatter(x, ss_arr[i, 2, j], marker=_marker[i],
#                         color=scatter_color[2], alpha=scatter_alpha)
#             ax2.scatter(x, ss_arr[i, 3, j], marker=_marker[i],
#                         color=scatter_color[3], alpha=scatter_alpha)
#
#     ax2.set_title('Secondary Score')
#     ax2.set_xlabel(x_label)
#     ax2.set_ylabel('Score')
#
#     lns = [mlines.Line2D([], [], color=scatter_color[i], linestyle='None',
#         marker=scatter_marker[3], ) for i in range(0, 4)]
#     labs = ss_labels
#     ax2.legend(lns, labs)
#
#     fig.savefig("%s.%s" % (figname, extension))
#     # fig.show()
#     plt.close()