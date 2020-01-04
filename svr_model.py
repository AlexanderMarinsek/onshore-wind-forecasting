from svr_converter import normalize_data, save_data_as_npz, get_features_and_labels

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from datetime import datetime


class Svr:

    def __init__(self, era5_path, name):
        self.era5_path = era5_path
        self.name = name
        self.set_parameters()
        self.set_vars()


    def set_parameters(self, kernel='rbf', c=0.1, epsilon=1):
        self.kernel = kernel
        self.c = c
        self.epsilon = epsilon


    def set_vars(self, M=1, N=24, G=0):
        self.M = M
        self.N = N
        self.G = G


    def get_vars(self):
        return [self.M, self.N, self.G]


    def run(self, start_loc_dt, stop_loc_dt):


        features, label_V, label_U = get_features_and_labels(
            self.era5_path, start_loc_dt, stop_loc_dt, self.M, self.N, self.G)

        # Calculate speed (abs size) - math.sqrt has problems with numpy arrays
        labels = (label_V ** 2 + label_U ** 2) ** (1 / 2)

        # Split features and labels with regard to time (dictated by N)
        train_features = features[0:-self.N, :]
        test_features = features[-self.N:, :]
        train_labels = labels[0:-self.N]
        test_labels = labels[-self.N:]

        # TODO: combine V and U scalers for scaling the absoulute value
        scaler = StandardScaler()

        # Measure elapsed time
        rf_start_time = datetime.now()

        svr = SVR(kernel=self.kernel, C=self.c, epsilon=self.epsilon)

        svr.fit(train_features, train_labels)
        # forecast = scaler.inverse_transform(svr.predict(test_features))
        forecast = svr.predict(test_features)

        # Measure elapsed time
        rf_stop_time = datetime.now()
        elapsed = rf_stop_time - rf_start_time
        t = elapsed.total_seconds()

        return [t, test_labels, forecast]



def main():
    from pytz import timezone
    from results import Results

    start_dt = datetime(2018, 1, 1, 0)
    stop_dt = datetime(2018, 1, 5, 0)

    tz = timezone("Europe/Bucharest")
    start_loc_dt = tz.localize(start_dt)
    stop_loc_dt = tz.localize(stop_dt)

    date_str = datetime.utcnow().strftime("%F")
    results = Results("./Results", "%sx01" % date_str)

    svr = Svr("../ERA5-Land/Area-44.5-28.5-44.7-28.7", "RF")

    M = 1; N = 24; G = 0;
    svr.set_vars(M, N, G)
    kernel = 'rbf'; c = 0.1; epsilon = 1
    svr.set_parameters(kernel, c, epsilon)

    [t, test_labels, forecast] = svr.run( start_loc_dt, stop_loc_dt )

    # TODO: "%F" might be windows specific
    filename = "SVR-forecast_%s-%s_%s-%f-%f_%d-%d-%d" % (
        start_loc_dt.strftime("%F"),
        stop_loc_dt.strftime("%F"),
        kernel, c, epsilon,
        M, N, G)
    results.plot_forecast(test_labels, forecast, filename)



if __name__ == "__main__":
    main()