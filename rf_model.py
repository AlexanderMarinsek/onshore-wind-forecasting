from rf_converter import get_features_and_labels

from sklearn.ensemble import RandomForestRegressor
from datetime import datetime


class Rf:

    def __init__(self, era5_path, name):
        self.era5_path = era5_path
        self.name = name
        self.set_parameters()
        self.set_vars()


    def set_parameters(self, nEstimators=100, criterion='mse'):
        self.nEstimators = nEstimators
        self.criterion = criterion


    def set_vars(self, M=1, N=24, G=0):
        self.M = M
        self.N = N
        self.G = G


    def get_vars(self):
        return [self.M, self.N, self.G]


    def run(self, start_loc_dt, stop_loc_dt):

        # label is target data column, features are all other data columns
        features, label_V, label_U =  get_features_and_labels(
            self.era5_path, start_loc_dt, stop_loc_dt, self.M, self.N, self.G)

        # Calculate speed (abs size) - math.sqrt has problems with numpy arrays
        labels = (label_V**2 + label_U**2)**(1/2)

        # Split features and labels with regard to time (dictated by N)
        train_features = features[0:-self.N, :]
        test_features = features[-self.N:, :]
        train_labels = labels[0:-self.N]
        test_labels = labels[-self.N:]

        # Measure elapsed time
        rf_start_time = datetime.now()

        # Initiate random forest
        rf = RandomForestRegressor(
            n_estimators=self.nEstimators, criterion=self.criterion, random_state=1)

        # Build forest of trees based on training data
        rf.fit(train_features, train_labels)

        # Create forecast based on test features
        forecast = rf.predict(test_features)

        # Measure elapsed time
        rf_stop_time = datetime.now()
        elapsed = rf_stop_time - rf_start_time
        t = elapsed.total_seconds()

        return [t, test_labels, forecast]


def main():
    from pytz import timezone

    start_dt = datetime(2018, 1, 1, 0)
    stop_dt = datetime(2018, 1, 5, 0)

    tz = timezone("Europe/Bucharest")
    start_loc_dt = tz.localize(start_dt)
    stop_loc_dt = tz.localize(stop_dt)

    rf = Rf("../ERA5-Land/Area-44.5-28.5-44.7-28.7", "RF")
    rf.run(start_loc_dt, stop_loc_dt)


if __name__ == "__main__":
    main()