from rf_converter import get_features_and_labels as get_features_and_labels

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from datetime import datetime


class Lstm:

    def __init__(self, era5_path, name):
        self.era5_path = era5_path
        self.name = name
        self.set_parameters()
        self.set_vars()


    # TODO: add relevant parameters
    def set_parameters(self):
        pass


    # TODO: add relevant parameters
    def get_parameters(self):
        return []


    def set_vars(self, M=1, N=24, G=0):
        self.M = M
        self.N = N
        self.G = G


    def get_vars(self):
        return [self.M, self.N, self.G]


    def run(self, start_loc_dt, stop_loc_dt):

        run_time = datetime.now()   # Timer 1

        features, label_V, label_U = get_features_and_labels(
            self.era5_path, start_loc_dt, stop_loc_dt, self.M, self.N, self.G)

        # Calculate speed (abs size) - math.sqrt has problems with numpy arrays
        labels = (label_V ** 2 + label_U ** 2) ** (1 / 2)

        # Scale features (no need to remember scale factor)
        features = MinMaxScaler(feature_range=(0, 1)).fit_transform(features)

        # Scale features (scaler later used for converting to real values)
        scaler = MinMaxScaler(feature_range=(0, 1))
        labels = scaler.fit_transform(labels.reshape(-1,1))[:,0]

        # Split features and labels with regard to time (dictated by N)
        train_features = features[0:-self.N, :]
        test_features = features[-self.N:, :]
        train_labels = labels[0:-self.N]
        test_labels = labels[-self.N:]

        # Reshape features into 3 dimensions (#samples, #timesteps, #features)
        train_features = train_features.reshape(train_features.shape[0], 1, train_features.shape[1])
        test_features = test_features.reshape(test_features.shape[0], 1, test_features.shape[1])

        # Initiate LSTM
        model = Sequential()
        model.add(LSTM(50, input_shape=(train_features.shape[1], train_features.shape[2])))
        model.add(Dense(1))
        model.compile(loss='mae', optimizer='adam')

        fit_time = datetime.now()    # Timer 2
        # Teach LSTM model
        model.fit(train_features, train_labels, epochs=10, batch_size=72, validation_data=(test_features, test_labels), verbose=2, shuffle=False)

        forecast_time = datetime.now()    # Timer 3
        # Create forecast based on test features
        forecast = scaler.inverse_transform(model.predict(test_features))

        # Measure elapsed time
        end_time = datetime.now()

        # Calculate elapsed times [Total time, fitting time, forecasting time]
        t = [
            (end_time - run_time).total_seconds(),
            (forecast_time - fit_time).total_seconds(),
            (end_time - forecast_time).total_seconds()
        ]

        return [t, scaler.inverse_transform(test_labels.reshape(-1,1))[:,0], forecast[:,0]]



def main():
    from pytz import timezone
    from results import Results

    start_dt = datetime(2017, 6, 1, 0)
    stop_dt = datetime(2018, 6, 1, 0)

    tz = timezone("Europe/Bucharest")
    start_loc_dt = tz.localize(start_dt)
    stop_loc_dt = tz.localize(stop_dt)

    date_str = datetime.utcnow().strftime("%F")
    results = Results("./Results", "%sx01" % date_str)

    # Set up model
    lstm = Lstm("../ERA5-Land/Area-44.5-28.5-44.7-28.7", "LSTM")
    M = 1; N = 24; G = 0;
    lstm.set_vars(M, N, G)
    lstm.set_parameters()

    # Create forecast
    [t, test_labels, forecast] = lstm.run( start_loc_dt, stop_loc_dt )

    # TODO: add parameters to filename
    filename = "SVR-forecast_%s-%s_%d-%d-%d" % (
        start_loc_dt.strftime("%F"),
        stop_loc_dt.strftime("%F"),
        M, N, G)
    results.plot_forecast(test_labels, forecast, filename)

    print (t)

if __name__ == "__main__":
    main()