import rf
from baseline import run_baseline as run_bl
from results import Results

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
start_loc_dt = [ tz.localize(s) for s in start_dt ]
stop_loc_dt = [ tz.localize(s) for s in stop_dt ]

# Initiate new results directory and global object
date_str = datetime.utcnow().strftime("%04Y-%02m-%02d")
results = Results("./Results", "%sx01" % date_str)


# M: Number of preceeding hours used in forecast     (CPU heavy)
# N: Number of forecasted hours                      (CPU light)
# G: Feature data points Grid type:                  (CPU heavy)
#   0: single point (same as for labels)
#   1: +
#   2: x
#   3: full (3x3)
M_arr = [1, 2, 3]
# N_arr = [1, 2, 3, 6, 12, 24, 48]
N_arr = [1, 6, 12, 48]
G_arr = [0, 1, 2, 3]

# Default values (reset before M-N-G iterations)
M_default = 1
N_default = 24
G_default = 0

# Optimal forecast values
M_opt = 3
N_opt = 14
G_opt = 3



def main():


    initial_time = datetime.now()
    text = "Begin: %s" % initial_time.strftime("%04Y-%02m-%02dT%02H:%02M:%02S")
    print(text)
    results.append_log(text)

    text = "Results directory: %s" % results.results_name
    print (text)
    results.append_log(text)

    # Baseline scores
    BL_score_all = []
    # Random Forest scores
    RF_M_score_all = RF_N_score_all = RF_G_score_all = []
    # Optimal random forest
    RF_optimal_score_all = []

    # Iterate dates ############################################################
    for start, stop in zip(start_loc_dt, stop_loc_dt):

        now = datetime.now().strftime("%04Y-%02m-%02dT%02H:%02M:%02S")
        text = "* New dates (%s): \n\t%s - %s" % ( now, start, stop )
        print (text)
        results.append_log(text)


        # Iterate M ############################################################
        M = M_default
        N = N_default
        G = G_default

        RF_M_score = rf.iterate_M(results, location, data_path, start, stop, M_arr, N, G)
        bs_labels = ["bs_V_0_M", "bs_V_tot_M", "bs_U_0_M", "bs_U_tot_M"]
        ss_labels = ["ss_V_0_M", "ss_V_tot_M", "ss_U_0_M", "ss_U_tot_M"]
        figname = "RF-M-scores_%s_%s_M_%d_%d" % (
            start.strftime("%04Y-%02m-%02d"),
            stop.strftime("%04Y-%02m-%02d"),
            N, G)
        results.plot_RF_M_scores(M_arr, RF_M_score[:4, :], RF_M_score[4:8, :],
            bs_labels, ss_labels, figname)

        if len(RF_M_score_all) > 0:
            RF_M_score_all = np.concatenate((RF_M_score_all, RF_M_score), axis=0)
        else:
            RF_M_score_all = RF_M_score


        # Iterate N ############################################################
        M = M_default
        N = N_default
        G = G_default

        RF_N_score = rf.iterate_N(results, location, data_path, start, stop, M, N_arr, G)
        bs_labels = ["bs_V_0_N", "bs_V_tot_N", "bs_U_0_N", "bs_U_tot_N"]
        ss_labels = ["ss_V_0_N", "ss_V_tot_N", "ss_U_0_N", "ss_U_tot_N"]
        figname = "RF-N-scores_%s_%s_%d_N_%d" % (
            start.strftime("%04Y-%02m-%02d"),
            stop.strftime("%04Y-%02m-%02d"),
            M_default, G_default)
        results.plot_RF_N_scores(N_arr, RF_N_score[:4, :], RF_N_score[4:8, :],
            bs_labels, ss_labels, figname)

        if len(RF_N_score_all) > 0:
            RF_N_score_all = np.concatenate((RF_N_score_all, RF_N_score), axis=0)
        else:
            RF_N_score_all = RF_N_score


        # Iterate G ############################################################
        M = M_default
        N = N_default
        G = G_default

        RF_G_score = rf.iterate_G(results, location, data_path, start, stop, M, N, G_arr)
        bs_labels = ["bs_V_0_G", "bs_V_tot_G", "bs_U_0_G", "bs_U_tot_G"]
        ss_labels = ["ss_V_0_G", "ss_V_tot_G", "ss_U_0_G", "ss_U_tot_G"]
        figname = "RF-G-scores_%s_%s_%d_%d_G" % (
            start.strftime("%04Y-%02m-%02d"),
            stop.strftime("%04Y-%02m-%02d"),
            N_default, M_default)
        results.plot_RF_N_scores(G_arr, RF_G_score[:4, :], RF_G_score[4:8, :],
            bs_labels, ss_labels, figname)

        if len(RF_G_score_all) > 0:
            RF_G_score_all = np.concatenate((RF_G_score_all, RF_G_score), axis=0)
        else:
            RF_G_score_all = RF_G_score


        # Run baseline #########################################################
        BL_score = run_bl(results, location, data_path, start, stop)

        if len(BL_score_all) > 0:
            BL_score_all = np.concatenate((BL_score_all, BL_score), axis=0)
        else:
            BL_score_all = BL_score


        # Create optimal RF forecast ###########################################
        M = M_opt
        N = N_opt
        G = G_opt

        RF_optimal_score = rf.optimal_forecast(results, location, data_path, start, stop, M, N, G)

        if len(RF_optimal_score_all) > 0:
            RF_optimal_score_all = np.concatenate((RF_optimal_score_all, RF_optimal_score), axis=0)
        else:
            RF_optimal_score_all = RF_optimal_score


    # Generate combined score plot for M
    if len(RF_M_score_all) > 0:
        bs_labels = ["bs_V_0_M", "bs_V_tot_M", "bs_U_0_M", "bs_U_tot_M"]
        ss_labels = ["ss_V_0_M", "ss_V_tot_M", "ss_U_0_M", "ss_U_tot_M"]
        figname = "RF-M-scores-all_%s_%s_M_%d_%d" % (
            start_loc_dt[0].strftime("%04Y-%02m-%02d"),
            stop_loc_dt[-1].strftime("%04Y-%02m-%02d"),
            N_default, G_default)
        results.plot_RF_M_scores_all(M_arr,
            np.array([RF_M_score_all[i*8:i*8+4,:] for i in range(0, int(RF_M_score_all.shape[0]/8))]),
            np.array([RF_M_score_all[i*8+4:i*8+8,:] for i in range(0, int(RF_M_score_all.shape[0]/8))]),
            bs_labels, ss_labels, figname)

    # Generate combined score plot for N
    if len(RF_N_score_all) > 0:
        bs_labels = ["bs_V_0_N", "bs_V_tot_N", "bs_U_0_N", "bs_U_tot_N"]
        ss_labels = ["ss_V_0_N", "ss_V_tot_N", "ss_U_0_N", "ss_U_tot_N"]
        figname = "RF-N-scores-all_%s_%s_%d_N_%d" % (
            start_loc_dt[0].strftime("%04Y-%02m-%02d"),
            stop_loc_dt[-1].strftime("%04Y-%02m-%02d"),
            M_default, G_default)
        results.plot_RF_N_scores_all(N_arr,
            np.array([RF_N_score_all[i*8:i*8+4,:] for i in range(0, int(RF_N_score_all.shape[0]/8))]),
            np.array([RF_N_score_all[i*8+4:i*8+8,:] for i in range(0, int(RF_N_score_all.shape[0]/8))]),
            bs_labels, ss_labels, figname)

    # Generate combined score plot for G
    if len(RF_G_score_all) > 0:
        bs_labels = ["bs_V_0_G", "bs_V_tot_G", "bs_U_0_G", "bs_U_tot_G"]
        ss_labels = ["ss_V_0_G", "ss_V_tot_G", "ss_U_0_G", "ss_U_tot_G"]
        figname = "RF-G-scores-all_%s_%s_%d_%d_G" % (
            start_loc_dt[0].strftime("%04Y-%02m-%02d"),
            stop_loc_dt[-1].strftime("%04Y-%02m-%02d"),
            N_default, M_default)
        results.plot_RF_G_scores_all(G_arr,
            np.array([RF_G_score_all[i*8:i*8+4,:] for i in range(0, int(RF_G_score_all.shape[0]/8))]),
            np.array([RF_G_score_all[i*8+4:i*8+8,:] for i in range(0, int(RF_G_score_all.shape[0]/8))]),
            bs_labels, ss_labels, figname)


    # Generate combined score plot for Baseline
    if len(BL_score_all) > 0:
        bs_labels = ["bs_V_0_BL", "bs_V_tot_BL", "bs_U_0_BL", "bs_U_tot_BL"]
        ss_labels = ["ss_V_0_BL", "ss_V_tot_BL", "ss_U_0_BL", "ss_U_tot_BL"]
        figname = "BL-scores-all_%s_%s_%d_%d_BL" % (
            start_loc_dt[0].strftime("%04Y-%02m-%02d"),
            stop_loc_dt[-1].strftime("%04Y-%02m-%02d"),
            N_default, M_default)
        results.plot_BL_scores_all([0],
            np.array([BL_score_all[i*8:i*8+4,:] for i in range(0, int(BL_score_all.shape[0]/8))]),
            np.array([BL_score_all[i*8+4:i*8+8,:] for i in range(0, int(BL_score_all.shape[0]/8))]),
            bs_labels, ss_labels, figname)

    # Generate combined score plot for optimal RF
    if len(RF_optimal_score_all) > 0:
        bs_labels = ["bs_V_0_O", "bs_V_tot_O", "bs_U_0_O", "bs_U_tot_O"]
        ss_labels = ["ss_V_0_O", "ss_V_tot_O", "ss_U_0_O", "ss_U_tot_O"]
        figname = "RF-optimal-scores-all_%s_%s_%d_%d_%d" % (
            start_loc_dt[0].strftime("%04Y-%02m-%02d"),
            stop_loc_dt[-1].strftime("%04Y-%02m-%02d"),
            M_opt, N_opt, G_opt)
        results.plot_RF_optimal_scores_all([0],
            np.array([RF_optimal_score_all[i*8:i*8+4,:] for i in range(0, int(RF_optimal_score_all.shape[0]/8))]),
            np.array([RF_optimal_score_all[i*8+4:i*8+8,:] for i in range(0, int(RF_optimal_score_all.shape[0]/8))]),
            bs_labels, ss_labels, figname)


    # Generate combined plot for scores of forecasts
    scores = np.concatenate((BL_score_all[:, :], RF_optimal_score_all[:, :]), axis=1)
    bs_labels = ["BL", "RF"]
    ss_labels = ["ss_V_0", "ss_V_tot", "ss_U_0", "ss_U_tot"]
    figname = "Combined-scores_%s_%s" % (
        start_loc_dt[0].strftime("%04Y-%02m-%02d"),
        stop_loc_dt[-1].strftime("%04Y-%02m-%02d"))

    # [ forecast score (BL, RF...), error (four in total), date ]
    bs_arr = np.array([scores[i*8:i*8+4, :] for i in range(0,int(scores.shape[0]/8))]).T
    ss_arr = np.array([scores[i*8+4:i*8+8, :] for i in range(0,int(scores.shape[0]/8))]).T

    results.plot_score_comparison([0], bs_arr, ss_arr, bs_labels, ss_labels, figname)


    finish_time = datetime.now()
    elapsed = finish_time - initial_time

    text = "Finish: %s" % finish_time.strftime("%04Y-%02m-%02dT%02H:%02M:%02S")
    print (text)
    results.append_log(text)

    text = "Elapsed: %s" % str(elapsed)
    print (text)
    results.append_log(text)



if __name__ == "__main__":
    main()