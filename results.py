from plotting import *

import os
import csv


class Results:
    """
    Object for saving figures, forecasts and logs.
    """

    def __init__(self, results_dir, results_name):
        """
        Init results directory.

        :param results_dir: Path to general results directory.
        :param results_name: Name of new results directory.
        """

        self.results_dir = results_dir
        self.results_name = results_name

        self.verify_results_name()
        self.make_results_dir()


    def verify_results_name(self):
        """
        Verify reults directory doesn't exist yet. Change name accordingly, if
        it does.
        """

        for name in os.listdir(self.results_dir):
            if os.path.isdir(os.path.join(self.results_dir, name)):
                if self.results_name == name:
                    self.results_name = "%sx%02d" % \
                        (name.split("x")[0], int(name.split("x")[1])+1)
                    if (int(name.split("x")[1])+1 > 99):
                        print ("Result file error")
                        quit()
                    self.verify_results_name()


    def make_results_dir(self):
        """
        Make new results directory.
        """

        path = "%s/%s" % (self.results_dir, self.results_name)
        os.makedirs(path)


    def append_log(self, text):
        """
        Append text to log.
        """

        filename = "%s/%s/log" % (self.results_dir, self.results_name)
        with open(filename, "a") as log_file:
            log_file.write("%s\n" % text)   # add new line, like in print()


    def save_forecast(self, labels, forecast, filename):
        """
        Save forecast results and real data to CSV file.
        """

        filename = "%s/%s/%s.csv" % (self.results_dir, self.results_name, filename)
        with open(filename, 'w') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            # Write column names
            writer.writerow(
                ["labels", "forecast"])
            # Write data
            for i in range(0,len(labels)):
                writer.writerow([labels[i], forecast[i]])


    def save_comparison_forecast(self, lab_for_2d, names, filename):
        """
        Save forecast results and real data to CSV file.
        """

        name_labels = []
        for name in names:
            name_labels.append("labels-%s" % name)
            name_labels.append("forecast-%s" % name)

        filename = "%s/%s/%s.csv" % (self.results_dir, self.results_name, filename)
        with open(filename, 'w') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            # Write column names
            writer.writerow(name_labels)
            # Write data
            for i in range(0, lab_for_2d.shape[1]):
                writer.writerow(lab_for_2d[:,i])


    def save_csv(self, data, c_names, filename):
        filename = "%s/%s/%s.csv" % (self.results_dir, self.results_name, filename)
        with open(filename, 'w') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            # Write column names
            writer.writerow(c_names)
            # Write data
            for i in range(0,len(data)):
                writer.writerow(data[i])


    def save_npz(self, data, filename):

        filename = "%s/%s/%s.npz" % (self.results_dir, self.results_name, filename)
        np.savez(filename, data)


    def plot_forecast(self, labels, forecast, figname):
        """
        Plot V and U predictions against real values in two subplots. Save plot.

        :return: void.
        """

        figname = "%s/%s/%s" % (self.results_dir, self.results_name, figname)
        plot_forecast(labels, forecast, figname)


    def plot_comparison_forecast(self, lab_for_2d, names, filename):
        """
        Plot V and U predictions against real values in two subplots. Save plot.

        :return: void.
        """

        figname = "%s/%s/%s" % (self.results_dir, self.results_name, filename)
        plot_comparison_forecast( lab_for_2d, names, figname )


    def plot_rf_optimization(self, data, figname):
        """
        """

        figname = "%s/%s/%s" % (self.results_dir, self.results_name, figname)
        plot_rf_optimization(data, figname)


    def plot_n_test(self, data, figname):
        """
        """

        figname = "%s/%s/%s" % (self.results_dir, self.results_name, figname)
        plot_n_test(data, figname)


    def plot_RF_M_scores (self, M_arr, bs_arr, ss_arr, bs_labels, ss_labels, figname):
        """
        Plot forecast scores (Brier and secondary). Save plot.

        :param M_arr: List of X-axis values.
        :param bs_arr: 2D numpy array of Brier scores.
        :param ss_arr: 2D numpy array of secondary scores.
        :param bs_labels: List of Brier score labels.
        :param ss_labels: List of secondary score values.
        :param figname: string containing desired figure name without extension.

        :return: void.
        """

        figname = "%s/%s/%s" % (self.results_dir, self.results_name, figname)
        plot_scores (M_arr, 'M', bs_arr, ss_arr, bs_labels, ss_labels, figname)


    def plot_RF_N_scores (self, N_arr, bs_arr, ss_arr, bs_labels, ss_labels, figname):
        """
        Plot forecast scores (Brier and secondary). Save plot.

        :param N_arr: List of X-axis values.
        :param bs_arr: 2D numpy array of Brier scores.
        :param ss_arr: 2D numpy array of secondary scores.
        :param bs_labels: List of Brier score labels.
        :param ss_labels: List of secondary score values.
        :param figname: string containing desired figure name without extension.

        :return: void.
        """
        figname = "%s/%s/%s" % (self.results_dir, self.results_name, figname)
        plot_scores(N_arr, 'N', bs_arr, ss_arr, bs_labels, ss_labels, figname)


    def plot_RF_G_scores (self, G_arr, bs_arr, ss_arr, bs_labels, ss_labels, figname):
        """
        Plot forecast scores (Brier and secondary). Save plot.

        :param G_arr: List of X-axis values.
        :param bs_arr: 2D numpy array of Brier scores.
        :param ss_arr: 2D numpy array of secondary scores.
        :param bs_labels: List of Brier score labels.
        :param ss_labels: List of secondary score values.
        :param figname: string containing desired figure name without extension.

        :return: void.
        """

        figname = "%s/%s/%s" % (self.results_dir, self.results_name, figname)
        plot_scores (G_arr, 'G', bs_arr, ss_arr, bs_labels, ss_labels, figname)


    def plot_BL_scores_all (self, x, bs_arr, ss_arr, bs_labels, ss_labels, figname):
        """
        Plot multiple forecast scores (Brier and secondary). Save plot.

        :param x: List of X-axis values.
        :param bs_arr: 2D numpy array of Brier scores.
        :param ss_arr: 2D numpy array of secondary scores.
        :param bs_labels: List of Brier score labels.
        :param ss_labels: List of secondary score values.
        :param figname: string containing desired figure name without extension.

        :return: void.
        """

        figname = "%s/%s/%s" % (self.results_dir, self.results_name, figname)
        plot_scores_all (x, 'X', bs_arr, ss_arr, bs_labels, ss_labels, figname)


    def plot_RF_M_scores_all (self, M_arr, bs_arr, ss_arr, bs_labels, ss_labels, figname):
        """
        Plot multiple forecast scores (Brier and secondary). Save plot.

        :param M_arr: List of X-axis values.
        :param bs_arr: 2D numpy array of Brier scores.
        :param ss_arr: 2D numpy array of secondary scores.
        :param bs_labels: List of Brier score labels.
        :param ss_labels: List of secondary score values.
        :param figname: string containing desired figure name without extension.

        :return: void.
        """

        figname = "%s/%s/%s" % (self.results_dir, self.results_name, figname)
        plot_scores_all (M_arr, 'M', bs_arr, ss_arr, bs_labels, ss_labels, figname)


    def plot_RF_N_scores_all (self, N_arr, bs_arr, ss_arr, bs_labels, ss_labels, figname):
        """
        Plot multiple forecast scores (Brier and secondary). Save plot.

        :param N_arr: List of X-axis values.
        :param bs_arr: 2D numpy array of Brier scores.
        :param ss_arr: 2D numpy array of secondary scores.
        :param bs_labels: List of Brier score labels.
        :param ss_labels: List of secondary score values.
        :param figname: string containing desired figure name without extension.

        :return: void.
        """

        figname = "%s/%s/%s" % (self.results_dir, self.results_name, figname)
        plot_scores_all (N_arr, 'N', bs_arr, ss_arr, bs_labels, ss_labels, figname)


    def plot_RF_G_scores_all (self, G_arr, bs_arr, ss_arr, bs_labels, ss_labels, figname):
        """
        Plot multiple forecast scores (Brier and secondary). Save plot.

        :param G_arr: List of X-axis values.
        :param bs_arr: 2D numpy array of Brier scores.
        :param ss_arr: 2D numpy array of secondary scores.
        :param bs_labels: List of Brier score labels.
        :param ss_labels: List of secondary score values.
        :param figname: string containing desired figure name without extension.

        :return: void.
        """

        figname = "%s/%s/%s" % (self.results_dir, self.results_name, figname)
        plot_scores_all (G_arr, 'G', bs_arr, ss_arr, bs_labels, ss_labels, figname)


    def plot_RF_optimal_scores_all (self, x, bs_arr, ss_arr, bs_labels, ss_labels, figname):
        """
        Plot multiple forecast scores (Brier and secondary). Save plot.

        :param x: List of X-axis values.
        :param bs_arr: 2D numpy array of Brier scores.
        :param ss_arr: 2D numpy array of secondary scores.
        :param bs_labels: List of Brier score labels.
        :param ss_labels: List of secondary score values.
        :param figname: string containing desired figure name without extension.

        :return: void.
        """

        figname = "%s/%s/%s" % (self.results_dir, self.results_name, figname)
        plot_scores_all (x, 'X', bs_arr, ss_arr, bs_labels, ss_labels, figname)


    def plot_score_comparison (self, x, bs_arr, ss_arr, bs_labels, ss_labels, figname):
        """
        Plot multiple forecast scores (Brier and secondary). Save plot.

        :param x: List of X-axis values.
        :param bs_arr: 2D numpy array of Brier scores.
        :param ss_arr: 2D numpy array of secondary scores.
        :param bs_labels: List of Brier score labels.
        :param ss_labels: List of secondary score values.
        :param figname: string containing desired figure name without extension.

        :return: void.
        """

        figname = "%s/%s/%s" % (self.results_dir, self.results_name, figname)
        plot_score_comparison(x, 'X', bs_arr, ss_arr, bs_labels, ss_labels, figname)



def main():
    x = Log("./results", "2019-12-28x01")
    print ("Finish")


if __name__ == "__main__":
    main()