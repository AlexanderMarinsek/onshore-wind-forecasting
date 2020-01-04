from evaluate import *
from rf_converter import get_features_and_labels

import numpy as np
from datetime import datetime, timedelta

day_delta = timedelta(hours=24)


def run_baseline(results, location, data_path, start_loc_dt, stop_loc_dt):
    """
    Create baseline forecast - today will be equal to yesterday (full 24 hours).

    :param results: Results object reference.
    :param location: Location directory name.
    :param data_path: Path to ERA5 data storage.
    :param start_loc_dt: Start time, localized (aware) datetime object.
    :param stop_loc_dt: Stop time (only used in filename).

    :return: Numpy array of score results obtained during iteration.
    """

    now = datetime.now().strftime("%FT%02H:%02M:%02S")
    text = "* New 24h Baseline (%s)" % (now)
    print(text)
    results.append_log(text)

    # Only use 48h of data (24 columns, 24h between features and labels)
    _start_loc_dt = stop_loc_dt - day_delta

    # M=1, N=24, G=0
    features, label_V, label_U = \
        get_features_and_labels(data_path, location, _start_loc_dt, stop_loc_dt, 1, 24, 0)

    # Real V data
    test_labels_V = label_V
    # Predictions are equal to yesterday's values
    predictions_V = features[:,4]

    # Real U data
    test_labels_U = label_U
    # Predictions are equal to yesterday's values
    predictions_U = features[:,5]

    # Plot and save forecasts
    filename = "BL-forecast_24h_%s_%s" % (
        start_loc_dt.strftime("%F"),
        stop_loc_dt.strftime("%F"))
    results.plot_forecast(test_labels_V, predictions_V, test_labels_U,
                          predictions_U, filename)
    results.save_forecast(test_labels_V, predictions_V, test_labels_U,
                          predictions_U, filename)

    # Calculate brier scores (combine scores into column)
    bs = calc_score_array(calc_brier_score,
                              test_labels_V, predictions_V, test_labels_U,
                              predictions_U)
    # Calculate score ... what kind?
    ss = calc_score_array(calc_second_score,
                              test_labels_V, predictions_V, test_labels_U,
                              predictions_U)

    # Stack the results
    return np.concatenate((bs, ss), axis=0)