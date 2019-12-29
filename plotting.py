from matplotlib import pyplot as plt
import numpy as np


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

    plt.show()
