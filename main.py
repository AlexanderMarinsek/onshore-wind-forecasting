from bl_model import Bl
from rf_model import Rf
from svr_model import Svr
from results import Results
from evaluate import calc_errors
from turbines import *

from datetime import datetime
from pytz import timezone
import numpy as np
from csv import reader

# TODO: DONE take all 3x3 points into account
# TODO: DONE use more features (also feature selection for optimal features)
# TODO: DONE use sin and cos to represent temporal features
# TODO: DONE add baseline model for comparison (average? current value?)
# TODO: DONE add better evaluation methods (for accuracy etc.)

# TODO: DONE comment RF-connected code
# TODO: DONE save figs and logs in separate directories
# TODO: DONE automate evaluation
# TODO: DONE change "secondary_score" to something with more theoretical background
# TODO: DONE Expose RF parameters
# TODO: Add RNN (or similar for evaluation)
# TODO: DONE Add more direct comparison between BL, RF, RNN... (not only plots) - compare scores, or forecasts?

# TODO: Empirically find and set the best params in each model's set_params()

"""
Model (converter) variables

M: Number of preceeding hours used in forecast     (CPU heavy)
N: Number of forecasted hours                      (CPU light)
G: Feature data points Grid type:                  (CPU heavy)
    0: single point (same as for labels)
    1: +
    2: x
    3: full (3x3)
"""


def main():

    # Romania time is UTC +3/+2 (summer/winter)
    tz = timezone("Europe/Bucharest")

    # Local time training data start/stop (forecast begins on stop hour)
    # Validation training data start/stop times
    # valid_start_dt = [
    #     datetime(2016, 1, 15, 0), datetime(2016, 2, 15, 0),
    #     datetime(2016, 3, 15, 0), datetime(2016, 4, 15, 0),
    #     datetime(2016, 5, 15, 0), datetime(2016, 6, 15, 0),
    #     datetime(2016, 7, 15, 0), datetime(2016, 8, 15, 0),
    #     datetime(2016, 9, 15, 0), datetime(2016, 10, 15, 0),
    #     datetime(2016, 11, 15, 0), datetime(2016, 12, 15, 0)]
    # valid_stop_dt = [
    #     datetime(2018, 1, 15, 0), datetime(2018, 2, 15, 0),
    #     datetime(2018, 3, 15, 0), datetime(2018, 4, 15, 0),
    #     datetime(2018, 5, 15, 0), datetime(2018, 6, 15, 0),
    #     datetime(2018, 7, 15, 0), datetime(2018, 8, 15, 0),
    #     datetime(2018, 9, 15, 0), datetime(2018, 10, 15, 0),
    #     datetime(2018, 11, 15, 0), datetime(2018, 12, 15, 0)]

    # Test training data start/stop times
    # test_start_dt = [
    #     datetime(2016, 1, 15, 0), datetime(2016, 2, 15, 0),
    #     datetime(2016, 3, 15, 0), datetime(2016, 4, 15, 0),
    #     datetime(2016, 5, 15, 0), datetime(2016, 6, 15, 0),
    #     datetime(2016, 7, 15, 0), datetime(2016, 8, 15, 0)]
    # test_stop_dt = [
    #     datetime(2019, 1, 15, 0), datetime(2019, 2, 15, 0),
    #     datetime(2019, 3, 15, 0), datetime(2019, 4, 15, 0),
    #     datetime(2019, 5, 15, 0), datetime(2019, 6, 15, 0),
    #     datetime(2019, 7, 15, 0), datetime(2019, 8, 15, 0)]

    valid_start_dt = [datetime(2017, 2, 1, 0), datetime(2017, 3, 1, 0)]
    valid_stop_dt = [datetime(2017, 2, 3, 0), datetime(2017, 3, 3, 0)]

    test_start_dt = [datetime(2018, 2, 1, 0), datetime(2018, 3, 1, 0)]
    test_stop_dt = [datetime(2018, 2, 3, 0), datetime(2018, 3, 3, 0)]

    # Convert to localized, aware datetime object (2018-...00:00+02:00)
    valid_start_loc_dt = [tz.localize(s) for s in valid_start_dt]
    valid_stop_loc_dt = [tz.localize(s) for s in valid_stop_dt]
    test_start_loc_dt = [tz.localize(s) for s in test_start_dt]
    test_stop_loc_dt = [tz.localize(s) for s in test_stop_dt]

    # Preceeding hours; Forecast hours; Grid type
    M = [1, 2, 4]; N = [24]; G = [0, 3];

    # Measurement height; WT hub height; Surface Roughness (0.1~0.3 for RO)
    h0 = 10; h = 100; z0 = 0.2;

    # Define turbine
    # vestas_v90 = np.array([
    #     [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 25],
    #     [0, 75, 190, 353, 581, 886, 1273, 1710, 2145, 2544, 2837, 2965, 2995, 3000, 3000]])
    # turbine = Turbine("Vestas V90", 10, vestas_v90)
    ge_2_5_xl = np.array([
        [3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10,
            10.5, 11, 11.5, 12, 12.5, 13, 13.5, 25],
        [0, 30, 63, 129, 194, 295, 395, 527, 658, 809, 959, 1152,
            1345, 1604, 1862, 2060, 2248, 2340, 2426, 2475, 2495, 2500, 2500]])
    turbine = Turbine("GE 2.5xl", 10, ge_2_5_xl)

    data_dir = "../ERA5-Land/Area-44.5-28.5-44.7-28.7-10mVw-10mUw-2mt-Sp-Ssrd-2md"

    models = [
        Bl(data_dir, "BL"),
        Rf(data_dir, "RF"),
        Svr(data_dir, "SVR")
    ]

    models[0].set_parameters()
    models[1].set_parameters()
    models[2].set_parameters()

    # Initiate new results directory and global object
    date_str = datetime.utcnow().strftime("%F")
    results = Results("./Results", "%sx01" % date_str)

    # results.results_name = "2020-01-08x04"

    # Output and log
    text = "Begin (%s)" % datetime.now().strftime("%FT%02H:%02M:%02S")
    print(text)
    results.append_log(text)

    # Output and log
    text = "\nResults directory: %s" % results.results_name
    print (text)
    results.append_log(text)

    # Output and log
    text = "\nData directory: %s" % data_dir
    print (text)
    results.append_log(text)

    # Output and log
    text = "\nModels and params: %s" % \
           ["%s %s" % (model.name, str(model.get_parameters())) for model in models]
    print (text)
    results.append_log(text)

    # Output and log
    text = "\nTurbine: (h0-h-z0 %.2f-%.2f-%.2f) %s %s" % \
           (h0, h, z0, turbine.name, str(turbine.p_curve))
    print (text)
    results.append_log(text)

    # Output and log
    text = "\nValidation- and test- star/stop dates: \n %s\n %s\n %s\n %s" % (
        str(valid_start_loc_dt),
        str(valid_stop_loc_dt),
        str(test_start_loc_dt),
        str(test_stop_loc_dt))
    print(text)
    results.append_log(text)


    # Tune model variables based on validation training data
    # tune_model_vars(
    #     models[1], valid_start_loc_dt, valid_stop_loc_dt, M, N, G, results )
    # tune_model_vars(
    #     models[2], valid_start_loc_dt, valid_stop_loc_dt, M, N, G, results )

    # Compare model forecasts based on test training data
    compare_models(
        models, test_start_loc_dt, test_stop_loc_dt, results )

    # Extrapolate forecasted wind speeds and calulate power
    # extrapolate_and_calc_power (
    #     models, test_start_loc_dt, test_stop_loc_dt, h0, h, z0, turbine, results )
    #
    # # Evaluate the influence of the N variable for each model
    # N = [n for n in range(1,25)]
    # eval_model_n_var(
    #     models[1], test_start_loc_dt, test_stop_loc_dt, N, results )
    # eval_model_n_var(
    #     models[2], test_start_loc_dt, test_stop_loc_dt, N, results )



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

    :return: void
    """

    # Output and log
    text = "\nTune %s M-N-G variables (%s): %s-%s-%s" % (
        model.name, datetime.now().strftime("%FT%02H:%02M:%02S"),
        str(M), str(N), str(G) )
    print(text)
    results.append_log(text)

    res_3d = []

    for start, stop in zip(start_loc_dt, stop_loc_dt):

        text = " New dates (%s): %s-%s" % (
            datetime.now().strftime("%FT%02H:%02M:%02S"),
            start.strftime("%FT%02H:%02M:%02S"),
            stop.strftime("%FT%02H:%02M:%02S") )
        print(text)
        results.append_log(text)

        res_2d = []

        for m in M:
            for n in N:
                for g in G:

                    # Get forecast
                    model.set_vars(m, n, g)     # Set M, N, G
                    [t, labels, forecast] = model.run(start, stop)

                    # Save and plot forecast for each date and model
                    filename = "%s-forecast-10_%s-%s_%d-%d-%d" % (
                        model.name,
                        start.strftime("%F"),
                        stop.strftime("%F"),
                        m, n, g)
                    results.save_forecast(labels, forecast, filename)
                    results.plot_forecast(labels, forecast, filename)

                    # Get relevant error array
                    errors = calc_errors(labels, forecast)  # Calc errors
                    res_tmp = np.array([
                        np.concatenate(([m, n, g], errors, t), axis=0)])

                    # Append results for current m-n-g parameters
                    if len(res_2d) > 0:
                        res_2d = np.concatenate((res_2d, res_tmp), axis=0)
                    else:
                        res_2d = res_tmp

        # Append 2D results to 3D array containing the time dimension
        if len(res_3d) > 0:
            res_3d = np.concatenate((res_3d, np.array([res_2d])), axis=0)
        else:
            res_3d = np.array([res_2d])

    # Save full (3D) collection of errors and times
    filename = "%s-tuning-errors" % model.name
    results.save_npz(res_3d, filename)

    # Calculate average errors and times
    res_avg = np.zeros(res_3d[0,:,3:].shape)   # exclude M, N, G variables
    length = res_3d.shape[0]    # number of dates

    # Average results based on date
    for i in range (0, length):
        res_avg += ( 1.0 * res_3d[i,:,3:] / length )

    # Save average errors and times
    data = np.concatenate((res_3d[0,:,:3], res_avg), axis=1)    # Add M, N, G
    c_names = ["M", "N", "G", "MAE", "MAPE", "MSE", "RMSE",
        "t_tot [s]", "t_fit [s]", "t_for [s]"]
    filename = "%s-tuning-errors-avg" % model.name
    results.save_csv( data, c_names, filename )

    # Plot average errors and times
    results.plot_var_tuning( model.name, data, filename )

    # Find optimal M, N, G combination
    [M_opt, N_opt, G_opt] = data[-1, 0:3]   # Last is optimal by default
    e_opt = data[-1, -4]
    for i in range(0, data.shape[0]-1):
        if data[i, -4] < e_opt:
            e_opt = data[i, -4]
            [M_opt, N_opt, G_opt] = data[i, 0:3]

    # Tune model according to optimal variables
    model.set_vars(int(M_opt), int(N_opt), int(G_opt))

    # Output and log
    text = "Tuned %s M-N-G variables (%s): %d-%d-%d" % (
        model.name, datetime.now().strftime("%FT%02H:%02M:%02S"), M_opt, N_opt, G_opt)
    print(text)
    results.append_log(text)


def compare_models( models, start_loc_dt, stop_loc_dt, results ):
    """
    Compare models and save error comparison.

    :param models: list of models based on custom classes.
    :param start_loc_dt: Start time, localized (aware) datetime object.
    :param stop_loc_dt: Stop time, localized (aware) datetime object.
    :param results: Results object reference.

    :return: void
    """

    # Generate array of model names
    names = [model.name for model in models]

    # Output and log
    text = "\nCompare models (%s): %s" % (
        datetime.now().strftime("%FT%02H:%02M:%02S"), str(names) )
    print(text)
    results.append_log(text)

    res_3d = []

    for start, stop in zip(start_loc_dt, stop_loc_dt):

        text = " New dates (%s): %s-%s" % (
            datetime.now().strftime("%FT%02H:%02M:%02S"),
            start.strftime("%FT%02H:%02M:%02S"),
            stop.strftime("%FT%02H:%02M:%02S") )
        print(text)
        results.append_log(text)

        res_2d = []
        lab_for_2d = []

        for model in models:

            model.set_vars(1,12,0)

            # Get forecast and relevant error array
            [t, labels, forecast] = model.run( start, stop )

            # Save and plot forecast for each date and model
            [m, n, g] = model.get_vars()
            filename = "%s-forecast-10_%s-%s_%d-%d-%d" % (
                model.name,
                start.strftime("%F"),
                stop.strftime("%F"),
                m, n, g)
            results.save_forecast(labels, forecast, filename)
            results.plot_forecast(labels, forecast, filename)

            # Get relevant error array
            errors = calc_errors(labels, forecast)
            res_tmp = np.array([np.concatenate((errors, t), axis=0)])

            # Append model results for current date
            if len(lab_for_2d) > 0:
                lab_for_2d = np.concatenate((lab_for_2d, np.array([labels, forecast])), axis=0)
                res_2d = np.concatenate((res_2d, res_tmp), axis=0)
            else:
                lab_for_2d = np.array([labels, forecast])
                res_2d = res_tmp

        # Save and plot all model forecasts together for current date
        filename = "Comparison-forecast_%s-%s" % (
            start.strftime("%F"),
            stop.strftime("%F"))
        results.save_comparison_forecast(lab_for_2d, names.copy(), filename)
        results.plot_comparison_forecast(lab_for_2d, names.copy(), filename)

        # Append 2D results to D array containing the time dimension
        if len(res_3d) > 0:
            res_3d = np.concatenate((res_3d, np.array([res_2d])), axis=0)
        else:
            res_3d = np.array([res_2d])

    # Save full (3D) collection of errors and times
    filename = "Comparison-errors"
    results.save_npz(res_3d, filename)

    # Calculate average errors and times
    res_avg = np.zeros(res_3d[0, :, :].shape)
    length = res_3d.shape[0]  # number of dates

    # Average results based on date
    for i in range(0, length):
        res_avg += (1.0 * res_3d[i, :, :] / length)

    # Save average errors and times
    data = np.concatenate( ( np.array([names]).T, res_avg ), axis=1 )
    c_names = ["Model", "MAE", "MAPE", "MSE", "RMSE",
               "t_tot [s]", "t_fit [s]", "t_for [s]"]
    filename = "Comparison-errors-avg"
    results.save_csv(data, c_names, filename)


def eval_model_n_var( model, start_loc_dt, stop_loc_dt, N, results ):
    """
    Evalueta forecast error for different N values.

    :param model: model based on custom class.
    :param start_loc_dt: Start time, localized (aware) datetime object.
    :param stop_loc_dt: Stop time, localized (aware) datetime object.
    :param N: Number of forecasted hours.
    :param results: Results object reference.

    :return: void
    """

    # Get m & n variables for later usage
    [m, n, g] = model.get_vars()

    text = "\nEvaluate %s N (%s): %d-%s-%d" % (
        model.name,
        datetime.now().strftime("%FT%02H:%02M:%02S"),
        m, str(N), g)
    print(text)
    results.append_log(text)

    res_3d = []

    for start, stop in zip(start_loc_dt, stop_loc_dt):

        text = " New dates (%s): %s-%s" % (
            datetime.now().strftime("%FT%02H:%02M:%02S"),
            start.strftime("%FT%02H:%02M:%02S"),
            stop.strftime("%FT%02H:%02M:%02S") )
        print(text)
        results.append_log(text)

        res_2d = []

        for n in N:

            # Get forecast
            model.set_vars(m, n, g)  # Set M, N, G
            [t, labels, forecast] = model.run(start, stop)

            # Save and plot forecast for each date and N
            [m, n, g] = model.get_vars()
            filename = "%s-forecast-10_%s-%s_%d-%d-%d" % (
                model.name,
                start.strftime("%F"),
                stop.strftime("%F"),
                m, n, g)
            results.save_forecast(labels, forecast, filename)
            results.plot_forecast(labels, forecast, filename)

            # Get relevant error array
            errors = calc_errors(labels, forecast)
            res_tmp = np.array([
                np.concatenate(([m, n, g], errors, t), axis=0)])

            # Append results for current m-n-g parameters
            if len(res_2d) > 0:
                res_2d = np.concatenate((res_2d, res_tmp), axis=0)
            else:
                res_2d = res_tmp

        # Append 2D results to 3D array containing the time dimension
        if len(res_3d) > 0:
            res_3d = np.concatenate((res_3d, np.array([res_2d])), axis=0)
        else:
            res_3d = np.array([res_2d])

    # Save full (3D) collection of errors and times
    filename = "%s-N-test-errors" % model.name
    results.save_npz(res_3d, filename)

    # Calculate average errors and times
    res_avg = np.zeros(res_3d[0,:,3:].shape)   # exclude M, N, G variables
    length = res_3d.shape[0]    # number of dates

    # Average results
    for i in range (0, length):
        res_avg += ( 1.0 * res_3d[i,:,3:] / length )

    # Save average errors and times
    data = np.concatenate((res_3d[0,:,:3], res_avg), axis=1)    # Add M, N, G
    c_names = ["M", "N", "G", "MAE", "MAPE", "MSE", "RMSE", "t [s]"]
    filename = "%s-N-test-errors-avg" % model.name
    results.save_csv( data, c_names, filename )

    # Plot average errors and times
    results.plot_n_eval( data, filename )


def extrapolate_and_calc_power ( models, start_loc_dt, stop_loc_dt, h0, h, z0, turbine, results ):
    """
    Extrapolate values and calculate power.

    :param models: list of models based on custom classes.
    :param start_loc_dt: Start time, localized (aware) datetime object.
    :param stop_loc_dt: Stop time, localized (aware) datetime object.
    :param h0: Height (meters) at which measurements were made.
    :param h: Height (meters) of the turbine's hub.
    :param z0: Surface roughnes (meters).
    :param turbine: Turbine model (custom class).
    :param results: Results object reference.

    :return: void
    """

    # Generate array of model names
    names = [model.name for model in models]

    # Output and log
    text = "\nCalculate power (%s): %d %d %f %s" % (
        datetime.now().strftime("%FT%02H:%02M:%02S"),
        h0, h, z0, turbine.name)
    print(text)
    results.append_log(text)

    res_3d = []

    for start, stop in zip(start_loc_dt, stop_loc_dt):

        text = " New dates (%s): %s-%s" % (
            datetime.now().strftime("%FT%02H:%02M:%02S"),
            start.strftime("%FT%02H:%02M:%02S"),
            stop.strftime("%FT%02H:%02M:%02S") )
        print(text)
        results.append_log(text)

        res_2d = []
        power_2d = []
        ext_2d = []

        for model in models:

            # Open saved forecast  for specific model, date and variables
            [m, n, g] = model.get_vars()
            filename = "%s/%s-forecast-10_%s-%s_%d-%d-%d" % (
                results.get_full_path(),
                model.name,
                start.strftime("%F"),
                stop.strftime("%F"),
                m, n, g)
            [labels_10, forecast_10] = read_forecast(filename)

            # Extrapolate values to hub height
            [labels_ext, forecast_ext] = extrapolate(np.array([labels_10, forecast_10]), h0, h, z0)

            # Calculate output power
            labels_p = turbine.calc_power(labels_ext)
            forecast_p = turbine.calc_power(forecast_ext)

            # Save and plot extrapolated forecast
            [m, n, g] = model.get_vars()
            filename = "%s-forecast-%d_%s-%s_%d-%d-%d" % (
                model.name,
                h,
                start.strftime("%F"),
                stop.strftime("%F"),
                m, n, g)
            results.save_forecast(labels_ext, forecast_ext, filename)
            results.plot_forecast(labels_ext, forecast_ext, filename)

            # Save and plot power forecast
            [m, n, g] = model.get_vars()
            filename = "%s-forecast-P%d_%s-%s_%d-%d-%d" % (
                model.name,
                h,
                start.strftime("%F"),
                stop.strftime("%F"),
                m, n, g)
            results.save_forecast(labels_p, forecast_p, filename)
            results.plot_forecast(labels_p, forecast_p, filename)

            # Get relevant error arrays
            errors_10 = calc_errors(labels_10, forecast_10)
            errors_ext = calc_errors(labels_ext, forecast_ext)
            errors_p = calc_errors(labels_p, forecast_p)

            res_tmp = np.array([errors_10, errors_ext, errors_p])

            # Append 2D results to 3D array containing the time dimension
            if len(res_2d) > 0:
                power_2d = np.concatenate((power_2d, np.array([labels_p, forecast_p])), axis=0)
                ext_2d = np.concatenate((ext_2d, np.array([labels_ext, forecast_ext])), axis=0)
                res_2d = np.concatenate((res_2d, res_tmp), axis=0)
            else:
                power_2d = np.array([labels_p, forecast_p])
                ext_2d = np.array([labels_ext, forecast_ext])
                res_2d = res_tmp

        # Plot power outputs (don't save - forecast CSV already exists)
        filename = "Power-forecast_%s-%s" % (
            start.strftime("%F"),
            stop.strftime("%F"))
        results.plot_power_forecast(power_2d, ext_2d, turbine.p_curve, names.copy(), filename)

        # Append 2D results to 3D array containing the time dimension
        if len(res_3d) > 0:
            res_3d = np.concatenate((res_3d, np.array([res_2d])), axis=0)
        else:
            res_3d = np.array([res_2d])

    # Save full (3D) collection of errors and times
    filename = "Power-errors"
    results.save_npz(res_3d, filename)

    # Calculate average errors and times
    res_avg = np.zeros(res_3d[0, :, :].shape)
    length = res_3d.shape[0]  # number of dates

    # Average results
    for i in range(0, length):
        res_avg += (1.0 * res_3d[i, :, :] / length)

    # Save average errors and times
    names_ext = np.array([np.repeat(names, 3)]).T
    titles_ext = np.array([["v10", "v100", "p100"] * len(names)]).T
    data = np.concatenate( ( names_ext, titles_ext, res_avg ), axis=1 )
    c_names = ["Model", "Title","MAE", "MAPE", "MSE", "RMSE"]
    filename = "Power-errors-avg"
    results.save_csv(data, c_names, filename)


def extrapolate( v0, h0, h, z0 ):
    """
    Extrapolate values (works with Numpy arrays).

    :param h0: Height (meters) at which measurements were made.
    :param h: Height (meters) of the turbine's hub.
    :param z0: Surface roughnes (meters).

    :return: Extrapolated value, or array of values
    """
    v = v0 * ( np.log(h/z0) / np.log(h0/z0) )
    return v


def read_forecast( filepath ):
    """
    Read forecast CSV.

    :param filepath: String containing the path to the forecast.

    :return: Numpy array of labels and forecasts
    """

    filepath = "%s.csv" % filepath
    forecast = []
    labels = []
    with open(filepath) as csv_file:
        csv_reader = reader(csv_file, delimiter=',')
        i = 0
        for row in csv_reader:
            if i > 0:
                labels.append(float(row[0]))
                forecast.append(float(row[1]))
            i += 1

    return [ np.array(labels), np.array(forecast) ]


if __name__ == "__main__":
    main()