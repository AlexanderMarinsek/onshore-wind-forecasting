from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


plt.style.use(["default", "./custom1.mplstyle"])
extension = "png"


def main():

    # test_plot_var_tuning()

    # test_plot_power_forecast()

    test_plot_comparison_forecast()


def test_plot_comparison_forecast():
    filename = "test"

    np.random.seed(seed=1)

    lab_for_2d = np.random.rand(6, 24)

    names = ["BL", "RF","SVR"]

    plot_comparison_forecast(lab_for_2d, names, filename)


def polygon_under_graph(xlist, ylist):
    """
    Construct the vertex list which defines the polygon filling the space under
    the (xlist, ylist) line graph.  Assumes the xs are in ascending order.
    """
    return [(xlist[0], 0.), *zip(xlist, ylist), (xlist[-1], 0.)]


def plot_comparison_forecast(lab_for_2d, names, figname):

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
    poly = PolyCollection(verts, facecolors=colors, alpha=0.5)
    ax.add_collection3d(poly, zs=range(1,5), zdir='y')

    ax.set_xlim(0, 24)
    ax.set_ylim(0, 5)
    ax.set_zlim(0, Y.max())

    # ax.set_title("Forecast comparison")
    ax.set_xlabel("Time (h)", labelpad=30)
    ax.set_ylabel("Model", labelpad=25)
    ax.set_zlabel("Speed (m/s)", labelpad=10)
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


def test_plot_var_tuning():

    filename = "test"

    data = np.array([
        [1, 24, 0, 9, 9, 9, 1.5, 2.5, 9, 9],
        [2, 24, 0, 9, 9, 9, 2.5, 6.5, 9, 9],
        [4, 24, 0, 9, 9, 9, 2.0, 7.0, 9, 9],
        [1, 24, 3, 9, 9, 9, 0.5, 4.5, 9, 9],
        [2, 24, 3, 9, 9, 9, 1.5, 12.0, 9, 9],
        [4, 24, 3, 9, 9, 9, 1.0, 20.5, 9, 9]
    ])

    name = "RF"

    plot_var_tuning( name, data, filename )


def plot_var_tuning( model_name, data, figname ):

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

    plt.title("%s optimization" % model_name)
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


def test_plot_power_forecast():
    filename = "test"
    
    np.random.seed(seed=1)
    
    power_2d = np.random.rand(6, 24)
    ext_2d = np.random.rand(6, 24)
    
    names = ["BL", "RF","SVR"]

    for i in range(0, power_2d.shape[0]):
        for j in range(0, power_2d.shape[1]):
            if power_2d[i,j] > 0.6:
                power_2d[i,j] = 0.6

    p_curve = np.array([
        [3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10,
            10.5, 11, 11.5, 12, 12.5, 13, 13.5, 25],
        [0, 30, 63, 129, 194, 295, 395, 527, 658, 809, 959, 1152,
            1345, 1604, 1862, 2060, 2248, 2340, 2426, 2475, 2495, 2500, 2500]])

    power_2d *= (p_curve.max() / 0.6)
    ext_2d[0,:] *= 4
    ext_2d[1,:] *= 4
    ext_2d[3,:] *= 9
    ext_2d[5,:] *= 7

    plot_power_forecast(power_2d, ext_2d, p_curve, names, filename)


def plot_power_forecast(power_2d, ext_2d, p_curve, names, figname):

    # Add ERA5-L model/forecast name
    # names.insert(0,"ERA5-L")

    # Plot colors
    # colors = ["C0", "C1", "C2", "C3"]
    colors = ["C1", "C9", "C3"]

    # Get Xs
    x = np.arange(0, power_2d.shape[1], 1)
    # Get power forecasts
    Yp = np.array([power_2d[i * 2 + 1, :] for i in range(0, int(power_2d.shape[0] / 2))])
    # Get extrapolated wind speed forecasts
    Ye = np.array([ext_2d[i * 2 + 1, :] for i in range(0, int(ext_2d.shape[0] / 2))])

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
    ax2.plot(p_curve[0,:], p_curve[1,:], ls="--", color="C7")

    for i in range(0, Ye.shape[0]):
        data = cut_power_curve(p_curve, Ye[i,:].max())
        ax2.fill_between(data[0,:], data[1,:], color=colors[i], alpha=0.3, label=names[i])

    ax2.set_title("WT Power Curve")
    # ax2.set_ylabel("Power (kW)")
    ax2.set_xlabel("Speed (m/s)")
    # ax2.legend()


    fig.savefig("%s.%s" % (figname, extension))
    #plt.show()
    plt.close()


def cut_power_curve (p_curve, max_speed):
    for i in range(0, p_curve.shape[1]):
        if p_curve[0,i] > max_speed:
            break

    return p_curve[:,:i]


main()