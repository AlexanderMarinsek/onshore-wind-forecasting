import numpy as np


# converts number to string with leading zero
def to_string(num):
    num_str = str(num)
    if len(num_str) < 2:
        num_str = '0' + num_str

    return num_str


# converts and aggregates sets of .npz files into numpy arrays
# returns 3 arrays: features, label_V and label_U
# features contains all columns (including wind) for day 'd'
# label_V and label_U contain wind components for day 'd+1'
# note that only values from data point (1,1) are used (from 3x3 grid)
def get_features_and_labels(relative_path, location, years, months, days):
    features = []
    label_V = []
    label_U = []
    for year in years:
        for month in months:
            for day in days:
                # TODO: allow_pickle=True?
                data = np.load(
                    relative_path + '/' + location + '/' + to_string(year) + '/' + to_string(month) + '/' + to_string(
                        day) + '/data.npz')
                if len(features) > 0:
                    features = np.append(features, data[data.files[1]][:][:, :, 1, 1], axis=1)
                else:
                    features = data[data.files[1]][:][:, :, 1, 1]

                try:
                    # try getting labels for next day
                    data = np.load(relative_path + '/' + location + '/' + to_string(year) + '/' + to_string(
                        month) + '/' + to_string(day + 1) + '/data.npz')
                except FileNotFoundError:
                    # if next day is new month, get labels for first day in new month
                    data = np.load(relative_path + '/' + location + '/' + to_string(year) + '/' + to_string(
                        month + 1) + '/' + to_string(1) + '/data.npz')
                # TODO: do same for next year and handle 'no next day' situation

                if len(label_V) > 0 and len(label_U) > 0:
                    label_V = np.append(label_V, data[data.files[1]][0][:, 1, 1], axis=0)
                    label_U = np.append(label_U, data[data.files[1]][1][:, 1, 1], axis=0)
                else:
                    label_V = data[data.files[1]][0][:, 1, 1]
                    label_U = data[data.files[1]][1][:, 1, 1]

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