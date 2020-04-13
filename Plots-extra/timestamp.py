from csv import reader
from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from math import sin, cos

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

def plot_timestamp( x, y, s, c, figname ):

    fig = plt.figure()

    ax1 = fig.add_subplot(111)

    ax1.plot(x, s, color='C0', linestyle="-", zorder=3, label="$timestamp_3$")
    ax1.plot(x, c, color='C3', linestyle="-", zorder=2, label="$timestamp_4$")
    ax1.plot(x, y, color='C7', linestyle="--", zorder=1, label="raw timestamp")

    ax1.legend()

    #ax1.set_title("N Variable Evaluation")
    ax1.set_xlabel("Time (h)")
    ax1.set_ylabel("Timestamp value")

    ax1.set_xticks([0,6,12,18,24,30,36,42,48,54,60,66,72])
    ax1.set_xticklabels(["0 (24)","6","12","18","0","6","12","18","0","6","12","18","0"])

    ax1.set_ylim(top=35)
    ax1.set_yticks([0, 6, 12, 18, 24])

    fig.tight_layout(pad=0.2)

    fig.savefig("%s.%s" % (figname, 'png'), dpi=300)
    fig.savefig("%s.%s" % (figname, 'eps'), dpi=300)
    # fig.savefig("%s.%s" % (figname, 'tif'), dpi=300)
    # fig.savefig("%s.%s" % (figname, 'pdf'), dpi=300)
    # fig.show()
    plt.close()


def main():

    x = np.arange(0,72,0.1)
    y = np.arange(0,24,0.1)
    y = np.concatenate((y,y,y))
    s = np.array([sin(val/24.0*2*np.pi)*12+12 for val in x])
    c = np.array([cos(val/24.0*2*np.pi)*12+12 for val in x])

    plot_timestamp( x, y, s, c, "timestamp-PUB" )


if __name__ == "__main__":
    main()