# Onshore Wind Forecasting
Onshore wind forecasting approach using ERA5-Land data and machine learning (ML). The code in this repository was used for the accompanying publication - "Demystifying the use of ERA5-Land and machine learning for wind power forecasting". The following ML approaches are implemented:
- random forest (RF)
- support vector regression (SVR)
- long short-term memory (LSTM)

## Prerequisites
The code was built using Python 3.7. It requires the following Python packages:
- numpy
- keras
- sklearn
- matplotlib

## Usage
For making forecasts, some minor code changes are needed in the *main()* function of the *main.py* file. It is strongly recommended to read the accompanying publication before use.

Training the ML algorithms requires ERA5-Land data. Use [ERA5-Land-Downloader](https://github.com/AlexanderMarinsek/ERA5-Land-Downloader) to download the data. Change the *data_dir* variable to the relative path of the dowloaded data. Example:
```
data_dir = "../ERA5-Land/Area-44.5-28.5-44.7-28.7-10mVw-10mUw-2mt-Sp-Ssrd-2md"
```

To set the validaton and testing data period, set the following variables:
```
valid_start_dt = [datetime(2017, 6, 15, 0)]
valid_stop_dt = [datetime(2017, 7, 15, 0)]
test_start_dt = [datetime(2018, 6, 15, 0)]
test_stop_dt = [datetime(2018, 7, 15, 0)]
```

To tune the ML models' hyperparameters, use the appropriate parameter tuning function. Each function takes as input the ML model and lists of hyperparameter values to test. Values used in the accompanying publication are included. Example for RF:
```
model = Rf(data_dir, "RF")
c_list = [0.01, 1, 100]
epsilon_list = [0.1, 0.2, 0.4, 0.8]
kernel_list = ["poly", "rbf"]
tune_rf_model_parameters(model, valid_start_loc_dt, valid_stop_loc_dt, n_estimators_list, criterion_list, max_features_list, results)
```

To tune the ML models' variables, input the desired ML model and lists of variable values into the *tune_model_vars* function. Values used in the accompanying publication are included. Example for RF:
```
model = Rf(data_dir, "RF")
M = [1, 2, 4]
N = [24]
G = [0, 3]
tune_model_vars(model, valid_start_loc_dt, valid_stop_loc_dt, M, N, G, results)
```

To compare forecasts, use the *compare_models* function. Example for RF and SVR:
```
models = [Rf(data_dir, "RF"), Svr(data_dir, "SVR")]
compare_models(models, test_start_loc_dt, test_stop_loc_dt, results)
```

To produce a power forecast, use the *extrapolate_and_calc_power* function. The function allows for different environment and wind turbine configurations. Values used in the accompanying publication are included. Example for RF and SVR:
```
models = [Rf(data_dir, "RF"), Svr(data_dir, "SVR")]
extrapolate_and_calc_power (models, test_start_loc_dt, test_stop_loc_dt, h0, h, z0, turbine, results)
```

To evaluate the influence of the *N* variable, input the desired ML model into the *eval_model_n_var* function. Example for RF:
```
model = Rf(data_dir, "RF")
N = [n for n in range(1,25)]
eval_model_n_var(model, test_start_loc_dt, test_stop_loc_dt, N, results)
```

All results are saved in the *Results* directory.

### Notes
RMSE values are incorrectly averaged. In order to obtain the real RMSE when iterating multiple dates, calculate the sqrt value of the combined MSE in the output .csv file.

Variable tuning images are based on the total time (including the converter).

In the accompanying publication the RMSE values were manually recalculated. Similarly the variable tuning images were replotted using the sum of fitting and forecast times.
