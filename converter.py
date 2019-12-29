import numpy as np

from datetime import datetime
from math import sin, cos

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

        days_in_year = 1.0 * 365
        hours_in_year = days_in_year * 24

        for month in months:
            for day in days:
                # TODO: allow_pickle=True?
                try:
                    data = np.load(
                        relative_path + '/' + location + '/' + to_string(year) + '/' + to_string(month) + '/' + to_string(
                            day) + '/data.npz')
                #except IOError:
                except FileNotFoundError:
                    break


                day_of_year = datetime(year, month, day).timetuple().tm_yday
                sh = (day_of_year - 1) * 24
                sin_daily_tmpst = np.array([[ sin( (sh+i) / 24.0 * 2*np.pi ) for i in range(0, 24) ]])
                cos_daily_tmpst = np.array([[ cos( (sh+i) / 24.0 * 2*np.pi ) for i in range(0, 24) ]])
                sin_yearly_tmpst = np.array([[ sin( (sh+i) / hours_in_year * 2*np.pi ) for i in range(0, 24) ]])
                cos_yearly_tmpst = np.array([[ cos( (sh+i) / hours_in_year * 2*np.pi ) for i in range(0, 24) ]])

                tmp = np.append(sin_daily_tmpst, cos_daily_tmpst, axis=0)
                tmp = np.append(tmp, sin_yearly_tmpst, axis=0)
                tmp = np.append(tmp, cos_yearly_tmpst, axis=0)

                #tmp = np.append(tmp, data[data.files[1]][:][:, :, 1, 1], axis=0)
                # Exclude wind from features
                tmp = np.append(tmp, data[data.files[1]][:][:, :, 1, 1][2:], axis=0)
                #tmp = data[data.files[1]][:][:, :, 1, 1]

                if len(features) > 0:
                    features = np.append(features, tmp, axis=1)
                else:
                    features = tmp
                """
                try:
                    # try getting labels for next day
                    data = np.load(relative_path + '/' + location + '/' + to_string(year) + '/' + to_string(
                        month) + '/' + to_string(day + 1) + '/data.npz')
                except FileNotFoundError:
                    # if next day is new month, get labels for first day in new month
                    try:
                        data = np.load(relative_path + '/' + location + '/' + to_string(year) + '/' + to_string(
                            month + 1) + '/' + to_string(1) + '/data.npz')
                    except FileNotFoundError:
                        label_V = np.append(label_V, np.zeros((24)), axis=0)
                        label_U = np.append(label_U, np.zeros((24)), axis=0)
                        break
                # TODO: do same for next year and handle 'no next day' situation
                """
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