from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from converter import get_features_and_labels
from plotting import plot_results, plot_features

import numpy as np

# TODO: take all 3x3 points into account
# TODO: use more features (also feature selection for optimal features)
# TODO: use sin and cos to represent temporal features
# TODO: add baseline model for comparison (average? current value?)
# TODO: add better evaluation methods (for accuracy etc.)

# list all days, months and years for getting data
# WARNING: not all months have same day count! (get_features_and_labels assumes this)
days = [1, 2, 3, 4]
months = [1]
years = [2018]

#days = np.arange(1, 32, 1)
#months = np.arange(1, 13, 1)

#days = np.arange(19, 21, 1)
#days = np.arange(1, 31, 1)
#months = np.arange(6, 8, 1)

#days = np.arange(28, 32, 1)
#days = np.arange(1, 3, 1)
#months = np.arange(12, 13, 1)

# set location (as string)
location = 'Area-44.5-28.5-44.7-28.7'

# set relative path to source data file
relative_path = '../ERA5-Land'

# label is target data column, features are all other data columns
features, label_V, label_U = \
    get_features_and_labels(relative_path, location, years, months, days)

plot_features(features)


"""
ts = 24.0/features.shape[0]
# segmenting data on training and test set
train_features_V, test_features_V, train_labels_V, test_labels_V = \
    train_test_split(features, label_V, test_size=ts, random_state=1)
train_features_U, test_features_U, train_labels_U, test_labels_U = \
    train_test_split(features, label_U, test_size=ts, random_state=1)
"""

train_features_V = features[0:-24,:]
test_features_V = features[-24:,:]
#test_features_V = features[-24:,0:2]
train_labels_V = label_V[0:-24]
test_labels_V = label_V[-24:]

train_features_U = features[0:-24,:]
test_features_U = features[-24:,:]
train_labels_U = label_U[0:-24]
test_labels_U = label_U[-24:]

rf = RandomForestRegressor(n_estimators=1000, random_state=1)

# predictions below only useful for evaluation
# to predict wind for day 'd+1', use day 'd' data in 'rf.predict'
rf.fit(train_features_V, train_labels_V)
predictions_V = rf.predict(test_features_V)
rf.fit(train_features_U, train_labels_U)
predictions_U = rf.predict(test_features_U)


plot_results(test_labels_V, predictions_V, test_labels_U, predictions_U)

