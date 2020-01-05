from rf_converter import get_features_and_labels

from datetime import datetime, timedelta


class Bl:

    def __init__(self, era5_path, name):
        self.era5_path = era5_path
        self.name = name
        self.set_parameters()
        self.set_vars()


    def set_parameters(self, nEstimators=100, criterion='mse', max_depth=None, max_features='auto'):
        self.nEstimators = nEstimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.max_features = max_features


    def set_vars(self, M=1, N=24, G=0):
        self.M = M
        self.N = N
        self.G = G


    def get_vars(self):
        return [self.M, self.N, self.G]


    def run(self, start_loc_dt, stop_loc_dt):

        run_time = datetime.now()    # Timer 1

        start_loc_dt = stop_loc_dt - timedelta(hours=48)

        # label is target data column, features are all other data columns
        features, label_V, label_U =  get_features_and_labels(
            self.era5_path, start_loc_dt, stop_loc_dt, self.M, self.N, self.G)

        # Calculate speed (abs size) - math.sqrt has problems with numpy arrays
        labels = (label_V**2 + label_U**2)**(1/2)

        # Split features and labels with regard to time (dictated by N)
        train_features = features[0:-24, :]
        test_features = features[-24:, :]
        train_labels = labels[0:-24]
        test_labels = labels[-24:]

        # Create forecast based on test features
        forecast = train_labels

        # Measure elapsed time
        end_time = datetime.now()

        # Calculate elapsed times [Total time, fitting time, forecasting time]
        t = [
            (end_time - run_time).total_seconds(),
            0,
            0
        ]

        return [t, test_labels, forecast]


def main():
    from pytz import timezone

    start_dt = datetime(2018, 1, 1, 0)
    stop_dt = datetime(2018, 1, 5, 0)

    tz = timezone("Europe/Bucharest")
    start_loc_dt = tz.localize(start_dt)
    stop_loc_dt = tz.localize(stop_dt)

    bl = Bl("../ERA5-Land/Area-44.5-28.5-44.7-28.7", "RF")
    bl.run(start_loc_dt, stop_loc_dt)


if __name__ == "__main__":
    main()