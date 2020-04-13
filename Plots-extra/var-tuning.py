from csv import reader
from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

extension = "png"

# Combine default plot style (v2.0) with custom style
plt.style.use(["default", "./custom1.mplstyle"])

# Figure format
extension = "png"

# Custom settings only for scatter plots (not suitable for new MPL style)
scatter_marker = ['_', '_', '_', '_']
scatter_color = ['C0', 'C9', 'C9', 'C3']
scatter_lw = 2.5
scatter_alpha = 0.6

def plot_var_tuning( model_name, M, G, rmse, t, figname ):

    # colormap="viridis"
    # colormap="YlOrRd"
    # colormap="YlGnBu"
    colormap="bone_r"

    x = M
    y = G
    e = rmse
    t = t

    t_min = np.min(t, axis=0)
    t_max = np.max(t, axis=0)
    scale = [7500.0 * (_t-t_min*0.8) / t_max for _t in t]

    plt.figure()

    plt.scatter(x, y, s=scale, c=e, cmap=colormap, linewidths=2, edgecolors="#555555", zorder=2)

    clb = plt.colorbar()
    # clb.ax.set_title(r'RMSE ($\frac{m}{s}$)', loc='center', pad=10)

    plt.grid(zorder=2)

    #plt.title("%s Tuning" % model_name)
    plt.xlabel("M")
    plt.ylabel("Grid")

    plt.xticks(x)
    # plt.yticks(y)
    plt.yticks( [0,3], [ "1x1", "3x3" ] )

    plt.xlim(x.min()-1, x.max()+1)
    plt.ylim(y.min()-1, y.max()+1)

    plt.tight_layout(pad=0.2)

    plt.savefig("%s.%s" % (figname, 'png'), dpi=300)
    plt.savefig("%s.%s" % (figname, 'eps'), dpi=300)
    # plt.show()
    plt.close()


def main():

    filepath = "LSTM-var-tuning-errors-avg.csv"
    M = []; G = []; rmse = []; t = []

    with open(filepath) as csv_file:

        csv_reader = reader(csv_file, delimiter=',')
        i = 0

        for row in csv_reader:
            M.append(int(row[0]))
            G.append(int(row[1]))
            rmse.append(float(row[2]))
            t.append(float(row[3]))
            i += 1

    M = np.array(M)
    G = np.array(G)
    rmse = np.array(rmse)
    t = np.array(t)

    plot_var_tuning("LSTM" , M, G, rmse, t, "LSTM-var-tuning-errors-avg-PUB")


if __name__ == "__main__":
    main()