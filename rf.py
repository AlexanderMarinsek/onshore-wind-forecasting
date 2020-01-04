from rf_converter import get_features_and_labels
from evaluate import *

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
import numpy as np


def run_rf (results, location, relative_path, start_loc_dt, stop_loc_dt, M, N, G):
    """
    Get features and labels, and create a forecast using random forest.

    :param results: Results object reference.
    :param location: Location directory name.
    :param start_loc_dt: Start time, localized (aware) datetime object.
    :param stop_loc_dt: Stop time, localized (aware) datetime object.
    :param M: Number of preceeding hours used in forecast.
    :param N: Number of forecasted hours. MAX 24 !!!
    :param G: Feature data points Grid type.

    :return: List of real and forecasted labels
        [test_labels_V, predictions_V, test_labels_U, predictions_U]
    """

    # _delta = timedelta(hours=N)
    # _stop_loc_dt = stop_loc_dt + _delta
    _stop_loc_dt = stop_loc_dt

    # label is target data column, features are all other data columns
    features, label_V, label_U = \
        get_features_and_labels(relative_path, location, start_loc_dt, _stop_loc_dt, M, N, G)

    # Split V labels and features with regard to time (dictated by N)
    train_features_V = features[0:-N,:]
    test_features_V = features[-N:,:]
    train_labels_V = label_V[0:-N]
    test_labels_V = label_V[-N:]
    # Split U labels and features with regard to time (dictated by N)
    train_features_U = features[0:-N,:]
    test_features_U = features[-N:,:]
    train_labels_U = label_U[0:-N]
    test_labels_U = label_U[-N:]

    # # Split V labels and features with regard to time (dictated by N)
    # train_features_V = features[0:-N,:]
    # test_features_V = features[-24-1:-24+N-1,:]
    # train_labels_V = label_V[0:-24]
    # test_labels_V = label_V[-24-1:-24+N-1]
    # # Split U labels and features with regard to time (dictated by N)
    # train_features_U = features[0:-N,:]
    # test_features_U = features[-24-1:-24+N-1,:]
    # train_labels_U = label_U[0:-24]
    # test_labels_U = label_U[-24-1:-24+N-1]

    # Initiate random forest
    # rf = RandomForestRegressor(n_estimators=1000, random_state=1)
    rf = RandomForestRegressor(n_estimators=10, random_state=1)

    # Tailor RF to V component and create forecast
    rf.fit(train_features_V, train_labels_V)
    predictions_V = rf.predict(test_features_V)

    # Tailor RF to U component and create forecast
    rf.fit(train_features_U, train_labels_U)
    predictions_U = rf.predict(test_features_U)

    # Plot and save forecasts
    filename = "RF-forecast_%s_%s_%d_%d_%d" % (
        start_loc_dt.strftime("%F"),
        stop_loc_dt.strftime("%F"),
        M, N, G)
    results.plot_forecast(test_labels_V, predictions_V, test_labels_U,
                          predictions_U, filename)
    results.save_forecast(test_labels_V, predictions_V, test_labels_U,
                          predictions_U, filename)

    return [test_labels_V, predictions_V, test_labels_U, predictions_U]


def iterate_M(results, location, data_path, start_loc_dt, stop_loc_dt, M_values, N, G):
    """
    Create forecast while varying M parameter.

    :param results: Results object reference.
    :param location: Location directory name.
    :param data_path: Path to ERA5 data storage.
    :param start_loc_dt: Start time, localized (aware) datetime object.
    :param stop_loc_dt: Stop time, localized (aware) datetime object.
    :param M_values: (list) Number of preceeding hours used in forecast.
    :param N: Number of forecasted hours.
    :param G: Feature data points Grid type.

    :return: Numpy array of score results obtained during iteration.
    """

    bs = []
    ss = []

    for M in M_values:

        now = datetime.now().strftime("%FT%02H:%02M:%02S")
        text = "* New M (%s): %d" % (now, M)
        print (text)
        results.append_log(text)

        [test_labels_V, predictions_V, test_labels_U, predictions_U] = \
            run_rf(results, location, data_path, start_loc_dt, stop_loc_dt, M, N, G)

        # Calculate brier scores (combine scores into column)
        tmp_bs = calc_score_array(calc_brier_score,
            test_labels_V, predictions_V, test_labels_U, predictions_U)
        # Calculate score ... what kind?
        tmp_ss = calc_score_array(calc_second_score,
            test_labels_V, predictions_V, test_labels_U, predictions_U)

        if len(bs) > 0:
            bs = np.concatenate((bs, tmp_bs), axis=1)
            ss = np.concatenate((ss, tmp_ss), axis=1)
        else:
            bs = tmp_bs
            ss = tmp_ss

    # Stack the results
    return np.concatenate((bs, ss), axis=0)


def iterate_N(results, location, data_path, start_loc_dt, stop_loc_dt, M, N_values, G):
    """
    Create forecast while varying N parameter.

    :param results: Results object reference.
    :param location: Location directory name.
    :param data_path: Path to ERA5 data storage.
    :param start_loc_dt: Start time, localized (aware) datetime object.
    :param stop_loc_dt: Stop time, localized (aware) datetime object.
    :param M: Number of preceeding hours used in forecast.
    :param N_values: (list) Number of forecasted hours.
    :param G: Feature data points Grid type.

    :return: Numpy array of score results obtained during iteration.
    """

    bs = []
    ss = []

    for N in N_values:

        now = datetime.now().strftime("%FT%02H:%02M:%02S")
        text = "* New N (%s): %d" % (now, N)
        print (text)
        results.append_log(text)

        [test_labels_V, predictions_V, test_labels_U, predictions_U] = \
            run_rf(results, location, data_path, start_loc_dt, stop_loc_dt, M, N, G)

        # Calculate brier scores (combine scores into column)
        tmp_bs = calc_score_array(calc_brier_score,
            test_labels_V, predictions_V, test_labels_U, predictions_U)
        # Calculate score ... what kind?
        tmp_ss = calc_score_array(calc_second_score,
            test_labels_V, predictions_V, test_labels_U, predictions_U)

        if len(bs) > 0:
            bs = np.concatenate((bs, tmp_bs), axis=1)
            ss = np.concatenate((ss, tmp_ss), axis=1)
        else:
            bs = tmp_bs
            ss = tmp_ss

    # Stack the results
    return np.concatenate((bs, ss), axis=0)


def iterate_G(results, location, data_path, start_loc_dt, stop_loc_dt, M, N, G_values):
    """
    Create forecast while varying G parameter.

    :param results: Results object reference.
    :param location: Location directory name.
    :param data_path: Path to ERA5 data storage.
    :param start_loc_dt: Start time, localized (aware) datetime object.
    :param stop_loc_dt: Stop time, localized (aware) datetime object.
    :param M: Number of preceeding hours used in forecast.
    :param N: Number of forecasted hours.
    :param G_values: (list) Feature data points Grid type.

    :return: Numpy array of score results obtained during iteration.
    """

    bs = []
    ss = []

    for G in G_values:

        now = datetime.now().strftime("%FT%02H:%02M:%02S")
        text = "* New G (%s): %d" % (now, G)
        print (text)
        results.append_log(text)

        [test_labels_V, predictions_V, test_labels_U, predictions_U] = \
            run_rf(results, location, data_path, start_loc_dt, stop_loc_dt, M, N, G)

        # Calculate brier scores (combine scores into column)
        tmp_bs = calc_score_array(calc_brier_score,
            test_labels_V, predictions_V, test_labels_U, predictions_U)
        # Calculate score ... what kind?
        tmp_ss = calc_score_array(calc_second_score,
            test_labels_V, predictions_V, test_labels_U, predictions_U)

        if len(bs) > 0:
            bs = np.concatenate((bs, tmp_bs), axis=1)
            ss = np.concatenate((ss, tmp_ss), axis=1)
        else:
            bs = tmp_bs
            ss = tmp_ss

    # Stack the results
    return np.concatenate((bs, ss), axis=0)


def optimal_forecast(results, location, data_path, start_loc_dt, stop_loc_dt, M, N, G):
    """
    Create RF forecast.

    :param results: Results object reference.
    :param location: Location directory name.
    :param data_path: Path to ERA5 data storage.
    :param start_loc_dt: Start time, localized (aware) datetime object.
    :param stop_loc_dt: Stop time, localized (aware) datetime object.
    :param M: Number of preceeding hours used in forecast.
    :param N: Number of forecasted hours.
    :param G: Feature data points Grid type.

    :return: Numpy array of score results obtained during iteration.
    """

    bs = []
    ss = []

    now = datetime.now().strftime("%FT%02H:%02M:%02S")
    text = "* New optimal RF forecast (%s)" % (now)
    print (text)
    results.append_log(text)

    [test_labels_V, predictions_V, test_labels_U, predictions_U] = \
        run_rf(results, location, data_path, start_loc_dt, stop_loc_dt, M, N, G)

    # Calculate brier scores (combine scores into column)
    tmp_bs = calc_score_array(calc_brier_score,
        test_labels_V, predictions_V, test_labels_U, predictions_U)
    # Calculate score ... what kind?
    tmp_ss = calc_score_array(calc_second_score,
        test_labels_V, predictions_V, test_labels_U, predictions_U)

    if len(bs) > 0:
        bs = np.concatenate((bs, tmp_bs), axis=1)
        ss = np.concatenate((ss, tmp_ss), axis=1)
    else:
        bs = tmp_bs
        ss = tmp_ss

    # Stack the results
    return np.concatenate((bs, ss), axis=0)