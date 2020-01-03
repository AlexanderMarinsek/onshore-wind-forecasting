from datetime import datetime
from pytz import timezone
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from results import Results
from svr_converter import normalize_data, save_data_as_npz, get_features_and_labels


def run_svr(results, location, relative_path, start_loc_dt, stop_loc_dt, M, N, G, scaler_V, scaler_U, c, e, k):
    """
    Get features and labels, and create a forecast using random forest.

    :param results: Results object reference.
    :param location: Location directory name.
    :param start_loc_dt: Start time, localized (aware) datetime object.
    :param stop_loc_dt: Stop time, localized (aware) datetime object.
    :param M: Number of preceeding hours used in forecast.
    :param N: Number of forecasted hours. MAX 24 !!!
    :param G: Feature data points Grid type.
    :param c: C paramater of the SVR model.
    :param e: Epsilon parameter of the SVR model.
    :param k: Kernel to be used by the SVR model.

    :return: List of real and forecasted labels
        [test_labels_V, predictions_V, test_labels_U, predictions_U]
    """
    features, label_V, label_U  = get_features_and_labels(relative_path, location, start_loc_dt, stop_loc_dt, M, N, G)

    train_features_V = features[0:-N, :]
    test_features_V = features[-N:, :]
    train_labels_V = label_V[0:-N]
    test_labels_V = label_V[-N:]

    train_features_U = features[0:-N, :]
    test_features_U = features[-N:, :]
    train_labels_U = label_U[0:-N]
    test_labels_U = label_U[-N:]

    svr = SVR(kernel=k, C=c, epsilon=e)

    svr.fit(train_features_V, train_labels_V)
    predictions_V = scaler_V.inverse_transform(svr.predict(test_features_V))

    svr.fit(train_features_U, train_labels_U)
    predictions_U = scaler_U.inverse_transform(svr.predict(test_features_U))

    # TODO: "%F" might be windows specific
    filename = "SVR-forecast_%s_%s_%d_%d_%d" % (
        start_loc_dt.strftime("%F"),
        stop_loc_dt.strftime("%F"),
        M, N, G)
    results.plot_forecast(test_labels_V, predictions_V, test_labels_U,
                          predictions_U, filename)
    results.save_forecast(test_labels_V, predictions_V, test_labels_U,
                          predictions_U, filename)

    return [test_labels_V, predictions_V, test_labels_U, predictions_U]


def test_svr():

    # sample data for testing:
    location = 'Area-44.5-28.5-44.7-28.7'
    relative_path = '../ERA5-Land'
    start_dt = [datetime(2018, 1, 1, 0), datetime(2018, 2, 1, 0),
                datetime(2018, 3, 1, 0), datetime(2018, 4, 1, 0),
                datetime(2018, 5, 1, 0)]
    stop_dt = [datetime(2018, 7, 1, 0), datetime(2018, 8, 1, 0),
               datetime(2018, 9, 1, 0), datetime(2018, 10, 1, 0),
               datetime(2018, 11, 1, 0)]
    tz = timezone("Europe/Bucharest")
    start_loc_dt = [tz.localize(s) for s in start_dt]
    stop_loc_dt = [tz.localize(s) for s in stop_dt]
    date_str = datetime.utcnow().strftime("%F")
    results = Results("./Results", "%sx01" % date_str)
    M = 1  # not tested with other values
    N = 24
    G = 0  # currently only supported value
    scaler_V = StandardScaler()
    scaler_U = StandardScaler()

    # TODO:
    # find a way to run scaler.inverse_transform() without having to run normalize_data() every time
    # ideally, normalize_data() and save_data_as_npz should be run only once

    data_norm = normalize_data(relative_path, location, G, scaler_V, scaler_U)

    # TODO: run_svr() assumes 'data_normalized.npz' files are present alongside 'data.npz' files
    # TODO: uncomment below when running the script for the first time
    # save_data_as_npz(relative_path, location, data_norm)

    # c = 0.1, e = 1 acceptable performance for scaled data based on limited testing
    c = 0.1
    e = 1
    k = 'rbf'
    for start, stop in zip(start_loc_dt, stop_loc_dt):
        run_svr(results, location, relative_path, start, stop, M, N, G, scaler_V, scaler_U, c, e, k)
