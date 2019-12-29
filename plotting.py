from matplotlib import pyplot as plt
import numpy as np


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


def plot_results(test_labels_V, predictions_V, test_labels_U, predictions_U):
    x = np.arange(0, len(test_labels_V), 1)
    plt.figure()

    plt.subplot(121)
    plt.plot(x, test_labels_V, 'b', label='actual')
    plt.plot(x, predictions_V, 'r', label='predicted')
    plt.title('V wind component')
    plt.legend()

    plt.subplot(122)
    plt.plot(x, test_labels_U, 'b', label='actual')
    plt.plot(x, predictions_U, 'r', label='predicted')
    plt.title('U wind component')
    plt.legend()

    plt.savefig("fig-predictions-1.png")
    plt.show()
