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

def plot_n_eval( N, MAE, MAPE, MSE, RMSE, R2, figname ):
    x = N   # N

    fig = plt.figure()

    ax1 = fig.add_subplot(111)

    ax1.plot(x, RMSE, color='C0', linestyle="-", zorder=4, label="RMSE")

    ax1.plot(x, MSE, color='black', linestyle="--", zorder=2, label="MSE")
    ax1.plot(x, MSE, color='#bbbbbb', linestyle="-", zorder=1)

    MAPE_norm = MAPE / MAPE.max()

    ax1.plot(x, MAPE_norm, color='black', linestyle="-.", zorder=2, label="nMAPE")
    ax1.plot(x, MAPE_norm, color='#bbbbbb', linestyle="-", zorder=1)

    ax1.plot(x, MAE, color='black', linestyle=":", zorder=2, label="MAE")
    ax1.plot(x, MAE, color='#bbbbbb', linestyle="-", zorder=1)

    ax1.plot(x, R2, color='C3', linestyle="-", zorder=3, label="R$^2$")

    ax1.legend()

    #ax1.set_title("N Variable Evaluation")
    ax1.set_xlabel("N")
    ax1.set_ylabel("Error")

    # ax1.set_xlim(x.min(), x.max())

    fig.savefig("%s.%s" % (figname, extension))
    # fig.show()
    plt.close()


def plot_n_eval_comparison(data, names):

    fig = plt.figure()

    ax1 = fig.add_subplot(111)

    x = np.arange(1,25,1)

    colors = ['C9', 'C3', 'C8']

    for d, c, n in zip(data, colors, names):

        RMSE = d[0]
        R2 = d[1]

        ax1.plot(x, RMSE, color=c, linestyle="-", zorder=4, label="%s RMSE" % n)
        # ax1.plot(x, RMSE, color=c, linestyle="-", zorder=4, label="%s" % n)

        ax1.plot(x, R2, color=c, linestyle="--", zorder=3, label=r"%s R$^2$" % n)
        # ax1.plot(x, R2, color=c, linestyle=":", zorder=3)

    ax1.legend(loc='top right', ncol=3)
    ax1.set_ylim(top=2.49)

    # ax1.set_title("N Variable Evaluation")
    ax1.set_xlabel("N")
    ax1.set_ylabel(r'R$^2$ | RMSE ($\frac{m}{s}$)')

    # ax1.set_xlim(x.min(), x.max())
    ax1.set_xticks([1, 4, 8, 12, 16, 20, 24])

    plt.tight_layout(pad=0.2)

    fig.savefig("%s.%s" % ('n_iter_comparison', 'png'), dpi=300)
    plt.savefig("%s.%s" % ('n_iter_comparison', 'eps'), dpi=300)
    # fig.show()
    plt.close()


def get_rmse_and_r2_from_csv (filepath):

    RMSE = []
    R2 = []

    with open(filepath) as csv_file:
        csv_reader = reader(csv_file, delimiter=',')
        i = 0

        for row in csv_reader:
            RMSE.append(float(row[4]))
            R2.append(float(row[5]))
            i += 1

    RMSE = np.array(RMSE)
    R2 = np.array(R2)

    return [RMSE, R2]

def main():


    filepaths = ["RF-N-test-errors-avg.csv", "SVR-N-test-errors-avg.csv", "LSTM-N-test-errors-avg.csv"]
    names = ['RF', 'SVR', 'LSTM']

    data = []

    for filepath in filepaths:
        data.append(get_rmse_and_r2_from_csv(filepath))

    plot_n_eval_comparison( data, names )

# def main():
#
#     filepath = "SVR-N-test-errors-avg.csv"
#     N = []; MAE = []; MAPE = []; MSE = []; RMSE = []; R2 = []
#
#     with open(filepath) as csv_file:
#
#         csv_reader = reader(csv_file, delimiter=',')
#         i = 0
#
#         for row in csv_reader:
#             N.append(int(row[0]))
#             MAE.append(float(row[1]))
#             MAPE.append(float(row[2]))
#             MSE.append(float(row[3]))
#             RMSE.append(float(row[4]))
#             R2.append(float(row[5]))
#             i += 1
#
#     N = np.array(N)
#     MAE = np.array(MAE)
#     MAPE = np.array(MAPE)
#     MSE = np.array(MSE)
#     RMSE = np.array(RMSE)
#     R2 = np.array(R2)
#
#     plot_n_eval( N, MAE, MAPE, MSE, RMSE, R2, "SVR-N-test-errors-avg-PUB" )


if __name__ == "__main__":
    main()