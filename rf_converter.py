from datetime import datetime, timedelta
import numpy as np
from math import sin, cos


# Constant
hour_delta = timedelta(hours=1)


def timestamp_to_UTC (dt):
    """
    Convert (aware) datetime to UTC array.

    :param dt: (aware) datetime object.

    :return: Array of ints
        [year, month, day of month, day of year, hour of day]
    """

    return [int(dt.utctimetuple().tm_year), \
            int(dt.utctimetuple().tm_mon), \
            int(dt.utctimetuple().tm_mday), \
            int(dt.utctimetuple().tm_yday), \
            int(dt.utctimetuple().tm_hour)]


def calc_days_in_year(dt):
    """
    From datetime object calculate number of days in that year.

    :param dt: datetime object.

    :return: Number of days in year (int).
    """

    return int((datetime(dt.year,12,31) - datetime(dt.year-1,12,31)).days)


def extract_features (data, feat_hour, G):
    """
    Extract features from NPZ file.

    :param data: NPZ data object.
    :param feat_hour: Hour from which features are extracted.
    :param grid: Feature data points grid type (
        0: single point (same as for labels);
        1: +;
        2: x;
        3: full (3x3))

    :return: Extracted numpy data array.
    """

    if G==1:     # grid: +
        d = np.concatenate((
            data[data.files[1]][:, feat_hour, 1, 0],
            data[data.files[1]][:, feat_hour, 0, 1],
            data[data.files[1]][:, feat_hour, 2, 1],
            data[data.files[1]][:, feat_hour, 1, 2],
            data[data.files[1]][:, feat_hour, 1, 1]
        ), axis=0)
    elif G==2:     # grid: X
        d = np.concatenate((
            data[data.files[1]][:, feat_hour, 0, 0],
            data[data.files[1]][:, feat_hour, 2, 0],
            data[data.files[1]][:, feat_hour, 0, 2],
            data[data.files[1]][:, feat_hour, 2, 2],
            data[data.files[1]][:, feat_hour, 1, 1]
        ), axis=0)
    elif G==3:     # grid: full (3x3)
        d = np.concatenate((
            data[data.files[1]][:, feat_hour, 0, 0],
            data[data.files[1]][:, feat_hour, 1, 0],
            data[data.files[1]][:, feat_hour, 2, 0],
            data[data.files[1]][:, feat_hour, 1, 0],
            data[data.files[1]][:, feat_hour, 1, 1],
            data[data.files[1]][:, feat_hour, 1, 2],
            data[data.files[1]][:, feat_hour, 2, 0],
            data[data.files[1]][:, feat_hour, 2, 1],
            data[data.files[1]][:, feat_hour, 2, 2]
        ), axis=0)
    else:          # grid: single point (center)
        d = data[data.files[1]][:, feat_hour, 1, 1]
    return d


#def get_features_and_labels(data_path, location, start_loc_dt, stop_loc_dt, M, N, G):
def get_features_and_labels(era5_path, start_loc_dt, stop_loc_dt, M, N, G):
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

    # TODO: Switch features dimensions (equal to other code)

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
            path = "%s/%04d/%02d/%02d/data.npz" % \
                   (era5_path, feat_year, feat_month, feat_mday)
            # TODO: allow_pickle=True?
            try:
                data = np.load(path)
            #except IOError:
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
            tmp = np.concatenate((tmp,
                extract_features(data, feat_hour, G)), axis=0)

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
        path = "%s/%04d/%02d/%02d/data.npz" % \
               (era5_path, label_year, label_month, label_mday)
        try:
            data = np.load(path)
        except FileNotFoundError:
            label_V = np.append(label_V, np.zeros((N)), axis=0)
            label_U = np.append(label_U, np.zeros((N)), axis=0)
            break

        # Add label data to V and U list
        if len(label_V) > 0 and len(label_U) > 0:
            tmp = np.array([data[data.files[1]][0, label_hour, 1, 1]])
            label_V = np.append(label_V, tmp, axis=0)
            tmp = np.array([data[data.files[1]][1, label_hour, 1, 1]])
            label_U = np.append(label_U, tmp, axis=0)
        else:
            label_V = np.array([data[data.files[1]][0, label_hour, 1, 1]])
            label_U = np.array([data[data.files[1]][1, label_hour, 1, 1]])

        # Increment time by one hour (1h)
        tmp_time += hour_delta

    return features.T, label_V, label_U


"""
data structure:

data[data.files[1]][4][23][2][2]
                a   b  c   d  e

a: 0 - feature names
    0 - 10 metre V wind component
    1 - 10 metre U wind component
    2 - 2 metre temperature
    3 - surface pressure
    4 - surface solar radiation downwards

   1 - feature values

b: 0 ... 4 - feature select

c: 0 ... 23 - day select

d: 0 ... 2 - x coordinate select

e: 0 ... 2 - y coordinate select

examples:
data[data.files[1]][0][:,1,1] -> length 24 array of feature '0' data in point (1,1)
data[data.files[1]][1:][:,:,1,1] -> length 4x24 array of features '1'-'4' data in point (1,1)
"""