import os

import numpy as np
from datetime import timedelta
from math import sin, cos
from rf_converter import timestamp_to_UTC, calc_days_in_year, extract_features
from sklearn.preprocessing import StandardScaler

# Constant
hour_delta = timedelta(hours=1)


def save_data_as_npz(data_path, location, data):
    """
    Save numpy array of normalized data as .npz files.

    :param data_path: Path to ERA5 data storage.
    :param location: Location directory name.
    :param data: Numpy array of normalized data.
    """
    path_base = "%s/%s" % (data_path, location)
    idx = 0
    for path_file, dirs, files in os.walk(path_base):
        for file in files:
            if file == 'data.npz':
                temp_data = np.array([])
                for i in range(idx, idx+24):
                    if len(temp_data) == 0:
                        temp_data = data[idx].reshape(1,-1)
                    else:
                        temp_data = np.concatenate((temp_data, data[idx].reshape(1,-1)), axis=0)
                    idx += 1
                np.savez(path_file.replace('\\', '/') + '/data_normalized.npz', temp_data)


def normalize_data(data_path, location, G, scaler_V, scaler_U):
    """
    Go through all data.npz files in data_path and normalize them.

    :param data_path: Path to ERA5 data storage.
    :param location: Location directory name.
    :param G: Feature data points Grid type.
    :param scaler_V: StandardScaler object for V-wind component.
    :param scaler_U: StandardScaler object for U-wind component.

    :return: Numpy array of normalized data
    """
    combined_data = np.array([])
    path_base = "%s/%s" % (data_path, location)
    for path_file, dirs, files in os.walk(path_base):
        for file in files:
            if file == 'data.npz':
                path = path_file.replace('\\', '/') + '/data.npz'
                try:
                    data = np.load(path)
                except FileNotFoundError:
                    print("File '{file}' not found!".format(file=path))

                extracted_data = np.array([])
                for i in range(24):
                    if len(extracted_data) == 0:
                        extracted_data = extract_features(data, i, G).reshape(1, -1)
                    else:
                        extracted_data = np.concatenate((extracted_data, extract_features(data, i, G).reshape(1, -1)),
                                                        axis=0)

                if len(combined_data) == 0:
                    combined_data = extracted_data
                else:
                    combined_data = np.concatenate((combined_data, extracted_data), axis=0)

    wind_V = scaler_V.fit_transform(combined_data[:,0].reshape(-1,1))
    wind_U = scaler_U.fit_transform(combined_data[:,1].reshape(-1,1))
    temp = StandardScaler().fit_transform(combined_data[:,2].reshape(-1,1))
    press = StandardScaler().fit_transform(combined_data[:,3].reshape(-1,1))
    rad = StandardScaler().fit_transform(combined_data[:,4].reshape(-1,1))

    return np.concatenate((wind_V, wind_U, temp, press, rad), axis=1)


# 99% identical to rf_converter counterpart
def get_features_and_labels(data_path, location, start_loc_dt, stop_loc_dt, M, N, G):
    """
    Get features and labels associated with time and other parameters.

    :param data_path: Path to ERA5 data storage.
    :param location: Location directory name.
    :param start_loc_dt: Start time, localized (aware) datetime object.
    :param stop_loc_dt: Stop time, localized (aware) datetime object.
    :param M: Number of preceeding hours used in forecast.
    :param N: Number of forecasted hours.
    :param G: Feature data points Grid type.

    :return: List of numpy array and two lists [features, label_V, label_U]
    """

    features = []
    label_V = []
    label_U = []

    # Copy by value
    tmp_time = start_loc_dt

    # Loop time from start to stop, and extract features
    while (tmp_time < stop_loc_dt):

        # Get broken down UTC
        [year, month, mday, yday, hour] = timestamp_to_UTC(tmp_time)

        # Smooth yearly timestamp with hours (vs. only days)
        days_in_year = calc_days_in_year(start_loc_dt)
        hours_in_year = days_in_year * 24.0

        # Empty temporary features stack
        features_stack = []

        # Loop M parameter (historical data)
        for j in range(0,M):

            # Rewind time for historical feature extraction (dictated by M)
            feat_time = tmp_time - timedelta(hours=j)
            [feat_year, feat_month, feat_mday, feat_yday, feat_hour] = \
                timestamp_to_UTC(feat_time)

            # Open NPZ data file
            path = "%s/%s/%04d/%02d/%02d/data_normalized.npz" % \
                   (data_path, location, feat_year, feat_month, feat_mday)
            try:
                data = np.load(path)
            except FileNotFoundError:
                break

            # Calculate yearly and daily timestamps
            sin_daily_tmstp = np.array([ sin( feat_hour / 24.0 * 2*np.pi ) ])
            cos_daily_tmstp = np.array([ cos( feat_hour / 24.0 * 2*np.pi ) ])
            sin_yearly_tmstp = np.array([ sin( ((feat_yday-1)*24.0+feat_hour)
                / hours_in_year * 2*np.pi ) ])
            cos_yearly_tmstp = np.array([ cos( ((feat_yday-1)*24.0+feat_hour)
                / hours_in_year * 2*np.pi ) ])

            # Add timestamps to features
            tmp = np.concatenate((sin_daily_tmstp, cos_daily_tmstp,
                sin_yearly_tmstp, cos_yearly_tmstp), axis=0)

            # Add ALL environmental data (ERA5) to features
            # TODO: support for all G combinations, currently hardcoded for G = 0
            tmp = np.concatenate((tmp,
                data[data.files[0]][feat_hour]), axis=0)

            # Stack historical feature data ('M' number of sets on the stack)
            if len(features_stack) > 0:
                features_stack = np.concatenate((features_stack, tmp), axis=0)
            else:
                features_stack = tmp

        # Combine all features connected to single hourly label (one column)
        if len(features) > 0:
            features = np.concatenate((features, np.array([features_stack]).T), axis=1)
        else:
            features = np.array([features_stack]).T

        # Calculate label time based on forecast period length (N param)
        label_time = tmp_time + timedelta(hours=N)
        [label_year, label_month, label_mday, label_yday, label_hour] = \
            timestamp_to_UTC(label_time)

        # Fetch NPZ data associated with hourly labels
        path = "%s/%s/%04d/%02d/%02d/data_normalized.npz" % \
               (data_path, location, label_year, label_month, label_mday)
        try:
            data = np.load(path)
        except FileNotFoundError:
            label_V = np.append(label_V, np.zeros((N)), axis=0)
            label_U = np.append(label_U, np.zeros((N)), axis=0)
            break

        # Add label data to V and U list
        if len(label_V) > 0 and len(label_U) > 0:
            tmp = np.array([data[data.files[0]][feat_hour][0]])
            label_V = np.append(label_V, tmp, axis=0)
            tmp = np.array([data[data.files[0]][feat_hour][1]])
            label_U = np.append(label_U, tmp, axis=0)
        else:
            label_V = np.array([data[data.files[0]][feat_hour][0]])
            label_U = np.array([data[data.files[0]][feat_hour][1]])

        # Increment time by one hour (1h)
        tmp_time += hour_delta

    return features.T, label_V, label_U