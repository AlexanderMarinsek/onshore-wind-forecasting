# Onshore Wind Forecasting
Onshore wind forecasting approach using ERA5-Land data and various machine learning algorithms, with emphasis on Random Forests.

RMSE values ar incorrectly averaged. In order to obtain the real RMSE when iterating multiple dates, calculate the sqrt value of the combined MSE in the output .csv file.

Variable tuning images are based on the total time (including the converter).

In the accompanying publication the RMSE values were manually recalculated. Similarly the variable tuning images were replotted using the sum of fitting and forecast times.
