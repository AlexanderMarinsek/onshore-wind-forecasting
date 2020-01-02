from rf_converter import get_features_and_labels

from datetime import datetime, timedelta


class Bl:

    def __init__(self, era5_path, name):
        self.era5_path = era5_path
        self.name = name


    def run(self, start_loc_dt, stop_loc_dt):

        start_loc_dt = stop_loc_dt - timedelta(hours=48)

        # label is target data column, features are all other data columns
        features, label_V, label_U =  get_features_and_labels(
            self.era5_path, start_loc_dt, stop_loc_dt, 1, 24, 0)

        # Calculate speed (abs size) - math.sqrt has problems with numpy arrays
        labels = (label_V**2 + label_U**2)**(1/2)

        # Split features and labels with regard to time (dictated by N)
        train_features = features[0:-24, :]
        test_features = features[-24:, :]
        train_labels = labels[0:-24]
        test_labels = labels[-24:]

        # Create forecast based on test features
        forecast = train_labels

        return [0, test_labels, forecast]


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