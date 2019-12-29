from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from converter import get_features_and_labels
from plotting import plot_results


# TODO: take all 3x3 points into account
# TODO: use more features (also feature selection for optimal features)
# TODO: use sin and cos to represent temporal features
# TODO: add baseline model for comparison (average? current value?)
# TODO: add better evaluation methods (for accuracy etc.)

# list all days, months and years for getting data
days = [1, 2, 3, 4]  # WARNING: not all months have same day count! (get_features_and_labels assumes this)
months = [1]
years = [2018]

# set location (as string)
location = 'Area-44.5-28.5-44.7-28.7'

# set relative path to source data file
relative_path = '../ERA5-Land'

# label is target data column, features are all other data columns
features, label_V, label_U = get_features_and_labels(relative_path, location, years, months, days)

# segmenting data on training and test set
train_features_V, test_features_V, train_labels_V, test_labels_V = train_test_split(features, label_V, test_size=0.25, random_state=1)
train_features_U, test_features_U, train_labels_U, test_labels_U = train_test_split(features, label_U, test_size=0.25, random_state=1)

rf = RandomForestRegressor(n_estimators=1000, random_state=1)

# predictions below only useful for evaluation
# to predict wind for day 'd+1', use day 'd' data in 'rf.predict'
rf.fit(train_features_V, train_labels_V)
predictions_V = rf.predict(test_features_V)
rf.fit(train_features_U, train_labels_U)
predictions_U = rf.predict(test_features_U)

plot_results(test_labels_V, predictions_V, test_labels_U, predictions_U)

