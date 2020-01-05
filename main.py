from results import Results
from bl_model import Bl
from rf_model import Rf
from svr_model import Svr
from evaluate import calc_errors

from turbines import *

from datetime import datetime, timedelta
from pytz import timezone
import numpy as np



# TODO: DONE take all 3x3 points into account
# TODO: use more features (also feature selection for optimal features)
# TODO: DONE use sin and cos to represent temporal features
# TODO: DONE add baseline model for comparison (average? current value?)
# TODO: add better evaluation methods (for accuracy etc.)

# TODO: DONE comment RF-connected code
# TODO: DONE save figs and logs in separate directories
# TODO: DONE automate evaluation
# TODO: change "secondary_score" to something with more theoretical background
# TODO: Expose RF parameters
# TODO: Add RNN (or similar for evaluation)
# TODO: Add more direct comparison between BL, RF, RNN... (not only plots) - compare scores, or forecasts?




def main():

    # set location (as string)
    location = 'Area-44.5-28.5-44.7-28.7'

    # set relative path to source data file
    data_path = '../ERA5-Land'

    # Romania time is UTC +3/+2 (summer/winter)
    tz = timezone("Europe/Bucharest")

    # Local time training data start/stop (forecast begins one hour after stop)
    # start_dt = [
    #     datetime(2017, 1, 15, 0), datetime(2017, 2, 15, 0),
    #     datetime(2017, 3, 15, 0), datetime(2017, 4, 15, 0),
    #     datetime(2017, 5, 15, 0), datetime(2017, 6, 15, 0),
    #     datetime(2017, 7, 15, 0), datetime(2017, 8, 15, 0),
    #     datetime(2017, 9, 15, 0), datetime(2017, 10, 15, 0),
    #     datetime(2017, 11, 15, 0), datetime(2017, 12, 15, 0)]
    # stop_dt = [
    #     datetime(2018, 1, 15, 0), datetime(2018, 2, 15, 0),
    #     datetime(2018, 3, 15, 0), datetime(2018, 4, 15, 0),
    #     datetime(2018, 5, 15, 0), datetime(2018, 6, 15, 0),
    #     datetime(2018, 7, 15, 0), datetime(2018, 8, 15, 0),
    #     datetime(2018, 9, 15, 0), datetime(2018, 10, 15, 0),
    #     datetime(2018, 11, 15, 0), datetime(2018, 12, 15, 0)]
    start_dt = [datetime(2018, 1, 1, 0), datetime(2018, 2, 1, 0)]
    stop_dt = [datetime(2018, 1, 5, 0), datetime(2018, 2, 5, 0)]

    # Convert to localized, aware datetime object (2018-...00:00+02:00)
    start_loc_dt = [tz.localize(s) for s in start_dt]
    stop_loc_dt = [tz.localize(s) for s in stop_dt]

    # Initiate new results directory and global object
    date_str = datetime.utcnow().strftime("%F")
    results = Results("./Results", "%sx01" % date_str)

    # M: Number of preceeding hours used in forecast     (CPU heavy)
    # N: Number of forecasted hours                      (CPU light)
    # G: Feature data points Grid type:                  (CPU heavy)
    #   0: single point (same as for labels)
    #   1: +
    #   2: x
    #   3: full (3x3)
    M = [1, 2]
    # M = [1]
    N = [24]
    G = [0, 3]
    # G = [0]

    models = [
        Bl("../ERA5-Land/Area-44.5-28.5-44.7-28.7", "BL"),
        Rf("../ERA5-Land/Area-44.5-28.5-44.7-28.7", "RF"),
        Svr("../ERA5-Land/Area-44.5-28.5-44.7-28.7", "SVR")
    ]



    initial_time = datetime.now()
    text = "Begin: %s" % initial_time.strftime("%FT%02H:%02M:%02S")
    print(text)
    results.append_log(text)

    text = "Results directory: %s" % results.results_name
    print (text)
    results.append_log(text)

    # results.results_name = "2020-01-05x05"

    # models[1].set_vars(3, 24, 2)
    # tune_model_vars( models[1], start_loc_dt, stop_loc_dt, M, N, G, results )

    compare_models( models, start_loc_dt, stop_loc_dt, results )

    # N = [n for n in range(1,25)]
    # test_model_n_variable( models[1], start_loc_dt, stop_loc_dt, N, results )

    extrapolate_and_calc_power ( models, start_loc_dt, stop_loc_dt, results )


def extrapolate( v0, h0, h, z0 ):
    v = v0 * ( np.log(h/z0) / np.log(h0/z0) )
    return v

def read_forecast( filename ):
    import csv
    filename = "%s.csv" % filename
    forecast = []
    labels = []
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        i = 0
        for row in csv_reader:
            if i > 0:
                labels.append(float(row[0]))
                forecast.append(float(row[1]))
            i += 1

    return [ np.array(labels), np.array(forecast) ]


def extrapolate_and_calc_power ( models, start_loc_dt, stop_loc_dt, results ):

    names = [model.name for model in models]

    turbine = Turbine("Vestas V90", 10, calc_power_vestas_v90_3mw)

    [h0, h, z0] = [10, 100, 0.2]

    res_3d = []

    for start, stop in zip(start_loc_dt, stop_loc_dt):

        res_2d = []
        power_2d = []

        for model in models:

            [m, n, g] = model.get_vars()
            filename = "%s/%s-forecast_%s_%s_%d_%d_%d" % (
                results.get_full_path(),
                model.name,
                start.strftime("%F"),
                stop.strftime("%F"),
                m, n, g)
            [labels_10, forecast_10] = read_forecast(filename)

            [labels_100, forecast_100] = extrapolate(np.array([labels_10, forecast_10]), h0, h, z0)

            labels_p = turbine.calc_power(labels_100)
            forecast_p = turbine.calc_power(forecast_100)

            errors_10 = calc_errors(labels_10, forecast_10)
            errors_100 = calc_errors(labels_100, forecast_100)
            errors_p = calc_errors(labels_p, forecast_p)

            res_tmp = np.array([errors_10, errors_100, errors_p])

            if len(res_2d) > 0:
                power_2d = np.concatenate((power_2d, np.array([labels_p, forecast_p])), axis=0)
                res_2d = np.concatenate((res_2d, res_tmp), axis=0)
            else:
                power_2d = np.array([labels_p, forecast_p])
                res_2d = res_tmp

        filename = "Power-forecast_%s_%s" % (
            start.strftime("%F"),
            stop.strftime("%F"))
        results.plot_power_forecast(power_2d, names, filename)

        if len(res_3d) > 0:
            res_3d = np.concatenate((res_3d, np.array([res_2d])), axis=0)
        else:
            res_3d = np.array([res_2d])

    # Save full (3D) collection of errors and times
    filename = "Power-errors_%s_%s" % (
        start.strftime("%F"),
        stop.strftime("%F"))
    results.save_npz(res_3d, filename)

    # Calculate average errors and times
    res_avg = np.zeros(res_3d[0, :, :].shape)
    length = res_3d.shape[0]  # number of dates

    for i in range(0, length):
        res_avg += (1.0 * res_3d[i, :, :] / length)

    # Save average errors and times
    names_ext = np.array([np.repeat(names, 3)]).T
    titles_ext = np.array([["v10", "v100", "p100"] * len(names)]).T
    data = np.concatenate( ( names_ext, titles_ext, res_avg ), axis=1 )
    c_names = ["Model", "Title","MAE", "MAPE", "MSE", "RMSE"]
    filename = "Power-results-avg"
    results.save_csv(data, c_names, filename)

    a = 1

def compare_models( models, start_loc_dt, stop_loc_dt, results ):

    names = [model.name for model in models]

    res_3d = []

    for start, stop in zip(start_loc_dt, stop_loc_dt):

        res_2d = lab_for_2d = []

        for model in models:

            [t, labels, forecast] = model.run( start, stop )
            errors = calc_errors(labels, forecast)
            res_tmp = np.array([np.concatenate((errors, t), axis=0)])

            [m, n, g] = model.get_vars()
            filename = "%s-forecast_%s_%s_%d_%d_%d" % (
                model.name,
                start.strftime("%F"),
                stop.strftime("%F"),
                m, n, g)
            results.save_forecast(labels, forecast, filename)
            results.plot_forecast(labels, forecast, filename)

            if len(lab_for_2d) > 0:
                lab_for_2d = np.concatenate((lab_for_2d, np.array([labels, forecast])), axis=0)
                res_2d = np.concatenate((res_2d, res_tmp), axis=0)
            else:
                lab_for_2d = np.array([labels, forecast])
                res_2d = res_tmp

        filename = "Comparison-forecast_%s_%s" % (
            start.strftime("%F"),
            stop.strftime("%F"))
        results.save_comparison_forecast(lab_for_2d, names, filename)
        results.plot_comparison_forecast(lab_for_2d, names, filename)

        if len(res_3d) > 0:
            res_3d = np.concatenate((res_3d, np.array([res_2d])), axis=0)
        else:
            res_3d = np.array([res_2d])

    # Save full (3D) collection of errors and times
    filename = "Comparison-errors_%s_%s" % (
        start.strftime("%F"),
        stop.strftime("%F"))
    results.save_npz(res_3d, filename)

    # Calculate average errors and times
    res_avg = np.zeros(res_3d[0, :, :].shape)
    length = res_3d.shape[0]  # number of dates

    for i in range(0, length):
        res_avg += (1.0 * res_3d[i, :, :] / length)

    # Save average errors and times
    data = np.concatenate( ( np.array([names]).T, res_avg ), axis=1 )
    c_names = ["Model", "MAE", "MAPE", "MSE", "RMSE",
               "t_tot [s]", "t_fit [s]", "t_for [s]"]
    filename = "Comparison-results-avg"
    results.save_csv(data, c_names, filename)


def tune_model_vars(model, start_loc_dt, stop_loc_dt, M, N, G, results):
    """
    Based on forecast errors, set optimal M, N, G variables for the model.

    :param model: model based on custom class.
    :param start_loc_dt: Start time, localized (aware) datetime object.
    :param stop_loc_dt: Stop time, localized (aware) datetime object.
    :param M: Number of preceeding hours used in forecast.
    :param N: Number of forecasted hours.
    :param G: Feature data points Grid type.
    :param results: Results object reference.

    :return: List containing elapsed time, actual and forecasted values
        [t, labels, forecast]
    """

    model.set_parameters(100, "mse")

    res_3d = []

    for start, stop in zip(start_loc_dt, stop_loc_dt):
        res_2d = []
        for m in M:
            for n in N:
                for g in G:
                    model.set_vars(m, n, g)     # Set M, N, G
                    [t, labels, forecast] = model.run(start, stop)

                    filename = "%s-forecast_%s_%s_%d_%d_%d" % (
                        model.name,
                        start.strftime("%F"),
                        stop.strftime("%F"),
                        m, n, g)
                    results.save_forecast(labels, forecast, filename)
                    results.plot_forecast(labels, forecast, filename)

                    errors = calc_errors(labels, forecast)
                    res_tmp = np.array([
                        np.concatenate(([m, n, g], errors, [t]), axis=0)])

                    if len(res_2d) > 0:
                        res_2d = np.concatenate((res_2d, res_tmp), axis=0)
                    else:
                        res_2d = res_tmp

        if len(res_3d) > 0:
            res_3d = np.concatenate((res_3d, np.array([res_2d])), axis=0)
        else:
            res_3d = np.array([res_2d])


    # Save full (3D) collection of errors and times
    filename = "%s-optimization-errors%s_%s" % (
        model.name,
        start.strftime("%F"),
        stop.strftime("%F"))
    results.save_npz(res_3d, filename)

    # Calculate average errors and times
    res_avg = np.zeros(res_3d[0,:,3:].shape)   # exclude M, N, G variables
    length = res_3d.shape[0]    # number of dates

    for i in range (0, length):
        res_avg += ( 1.0 * res_3d[i,:,3:] / length )

    # Save average errors and times
    data = np.concatenate((res_3d[0,:,:3], res_avg), axis=1)    # Add M, N, G
    c_names = ["M", "N", "G", "MAE", "MAPE", "MSE", "RMSE", "t [s]"]
    filename = "%s-optimization-errors-avg_%s_%s" % (
        model.name,
        start.strftime("%F"),
        stop.strftime("%F"))
    results.save_csv( data, c_names, filename )

    # Plot average errors and times
    results.plot_rf_optimization( data, filename )

    # Find optimal M, N, G combination
    [M_opt, N_opt, G_opt] = data[-1, 0:3]   # Last is optimal by default
    e_opt = data[-1, -2]
    for i in range(0, data.shape[0]-1):
        if data[i, -2] < e_opt:
            e_opt = data[i, -2]
            [M_opt, N_opt, G_opt] = data[i, 0:3]

    # Tune model according to optimal variables
    model.set_vars(int(M_opt), int(N_opt), int(G_opt))




def test_model_n_variable(model, start_loc_dt, stop_loc_dt, N, results):
    """
    Collect error data of model forecast for different N values.

    :param model: model based on custom class.
    :param start_loc_dt: Start time, localized (aware) datetime object.
    :param stop_loc_dt: Stop time, localized (aware) datetime object.
    :param N: Number of forecasted hours.
    :param results: Results object reference.

    :return: List containing elapsed time, actual and forecasted values
        [t, labels, forecast]
    """

    model.set_parameters(100, "mse")

    res_3d = []

    [m, n, g] = model.get_vars()

    for start, stop in zip(start_loc_dt, stop_loc_dt):
        res_2d = []
        for n in N:
            model.set_vars(m, n, g)  # Set M, N, G
            [t, labels, forecast] = model.run(start, stop)

            filename = "%s-forecast_%s_%s_%d_%d_%d" % (
                model.name,
                start.strftime("%F"),
                stop.strftime("%F"),
                m, n, g)
            results.save_forecast(labels, forecast, filename)
            results.plot_forecast(labels, forecast, filename)

            errors = calc_errors(labels, forecast)
            res_tmp = np.array([
                np.concatenate(([m, n, g], errors, [t]), axis=0)])

            if len(res_2d) > 0:
                res_2d = np.concatenate((res_2d, res_tmp), axis=0)
            else:
                res_2d = res_tmp

        if len(res_3d) > 0:
            res_3d = np.concatenate((res_3d, np.array([res_2d])), axis=0)
        else:
            res_3d = np.array([res_2d])

    # Save full (3D) collection of errors and times
    filename = "%s-N-test-errors_%s_%s" % (
        model.name,
        start.strftime("%F"),
        stop.strftime("%F"))
    results.save_npz(res_3d, filename)

    # Calculate average errors and times
    res_avg = np.zeros(res_3d[0,:,3:].shape)   # exclude M, N, G variables
    length = res_3d.shape[0]    # number of dates

    for i in range (0, length):
        res_avg += ( 1.0 * res_3d[i,:,3:] / length )

    # Save average errors and times
    data = np.concatenate((res_3d[0,:,:3], res_avg), axis=1)    # Add M, N, G
    c_names = ["M", "N", "G", "MAE", "MAPE", "MSE", "RMSE", "t [s]"]
    filename = "%s-N-test-errors-avg_%s_%s" % (
        model.name,
        start.strftime("%F"),
        stop.strftime("%F"))
    results.save_csv( data, c_names, filename )

    # Plot average errors and times
    results.plot_n_test( data, filename )


if __name__ == "__main__":
    main()