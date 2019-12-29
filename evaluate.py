import numpy as np


def calc_brier_score(forecast, outcome, N):
    """
    Calculate Brier score for single value, or list of values.

    :param forecast: List containing forecasted data.
    :param outcome: List containing real (outcome) data.
    :param N: Number of elements.

    :return: Combined Brier score.
    """

    # if (len(forecast) != len(outcome)):
    #     print ("Brier error!")
    #     quit()

    sum = 0
    for i in range (0, N):
        sum += (forecast[i] - outcome[i])**2
    # Divide by the number of forecasting instances
    sum = float(sum) / N

    return sum


def calc_second_score (forecast, outcome, N):
    """
    Calculate Secondary score for single value, or list of values.

    :param forecast: List containing forecasted data.
    :param outcome: List containing real (outcome) data.
    :param N: Number of elements.

    :return: Combined Secondary score.
    """

    sum = 0
    for i in range (0, N):
        sum += abs(forecast[i] - outcome[i])
        #sum += (1.0 * forecast[i] / outcome[i])
    # Divide by the number of forecasting instances
    sum = float(sum) / N

    return sum


def calc_score_array (calc_score, test_labels_V, predictions_V, test_labels_U, predictions_U):
    """
    Calculate array of scores (0th element and all elements for both V and U).

    :param calc_score: Score calculation function (fun. pointer).
    :param test_labels_V: List of real V values.
    :param predictions_V: List of predicted V values.
    :param test_labels_U: List of real U values.
    :param predictions_U: List of predicted U values.

    :return: Numpy array of scores[V_0, V_tot, U_0, U_tot]
    """

    # 0th point score for V label
    V_0 = calc_score([test_labels_V[0]], [predictions_V[0]], 1)
    # Total combined score for V label
    V_tot = calc_score(test_labels_V, predictions_V, predictions_V.shape[0])
    # 0th point score for U label
    U_0 = calc_score([test_labels_U[0]], [predictions_U[0]], 1)
    # Total combined score for U label
    U_tot = calc_score(test_labels_U, predictions_U, predictions_U.shape[0])

    return np.array([[V_0, V_tot, U_0, U_tot]]).T
