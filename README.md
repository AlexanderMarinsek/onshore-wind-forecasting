# Onshore Wind Forecasting
Onshore wind forecasting approach using ERA5-Land data and machine learning. The code in this repository was used for the accompanying publication - "Demystifying the use of ERA5-Land and machine learning for wind power forecasting". The following machine learning approaches are implemented:
- random forest (RF)
- support vector regression (SVR)
- long short-term memory (LSTM)

## Prerequisites
The code was built using Python 3.7. It requires the following Python packages:
- numpy
- keras
- sklearn
- matplotlib

### Notes
RMSE values are incorrectly averaged. In order to obtain the real RMSE when iterating multiple dates, calculate the sqrt value of the combined MSE in the output .csv file.

Variable tuning images are based on the total time (including the converter).

In the accompanying publication the RMSE values were manually recalculated. Similarly the variable tuning images were replotted using the sum of fitting and forecast times.
