# sarima_model.py

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
from project1 import adf_test,check_acf_and_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.graphics.tsaplots as F
from statsmodels.tsa.stattools import adfuller



# sarima_model.py

from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings("ignore")

def sarima_auto(train_data, feature_name='cnt_dif_log',
                p_range=range(0,3), d_range=range(0,2), q_range=range(0,3),
                P_range=range(0,2), D_range=range(0,2), Q_range=range(0,2),
                s=7):
    """
    Automatically find the best SARIMA(p,d,q)(P,D,Q,s) model based on AIC using train_data.
    Returns the best fitted model and its orders.
    """
    y_train = train_data[feature_name].dropna()

    best_aic = float("inf")
    best_order = None
    best_seasonal_order = None
    best_model = None

    for p in p_range:
        for d in d_range:
            for q in q_range:
                for P in P_range:
                    for D in D_range:
                        for Q in Q_range:
                            try:
                                model = SARIMAX(y_train,
                                                order=(p,d,q),
                                                seasonal_order=(P,D,Q,s),
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)
                                res = model.fit(disp=False)
                                if res.aic < best_aic:
                                    best_aic = res.aic
                                    best_order = (p,d,q)
                                    best_seasonal_order = (P,D,Q,s)
                                    best_model = res
                            except:
                                continue

    print("Best SARIMA model found:")
    print(f"Non-seasonal order (p,d,q): {best_order}")
    print(f"Seasonal order (P,D,Q,s): {best_seasonal_order}")
    print(f"AIC: {best_aic:.2f}")

    return best_model, best_order, best_seasonal_order


def sarima_fit_model(train_data, test_data, feature_name='cnt_dif_log',
                     order=(1,1,1), seasonal_order=(1,0,1,7)):
    """
    Fit SARIMA model with given order and seasonal_order, return fitted model and predictions on test data.
    """
    y_train = train_data[feature_name].dropna()
    y_test = test_data[feature_name].dropna()

    # --- Fit SARIMA ---
    model = SARIMAX(y_train,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    sarima_fit = model.fit(disp=False)

    # --- Predict on test ---
    start = len(y_train)
    end = start + len(y_test) - 1
    predictions = sarima_fit.predict(start=start, end=end, dynamic=False)

    return sarima_fit, predictions


# def sarima_model(train_data, test_data, feature_name='cnt_dif_log',
#                  order=(1,1,1), seasonal_order=(1,0,1,7)):
#     """
#     Fits a SARIMA model, evaluates it, and plots forecasts.

#     Parameters:
#         train_data (pd.Series): Training data (time-indexed)
#         test_data (pd.Series): Test data (time-indexed)
#         feature_name (str): Column name of target series
#         order (tuple): (p,d,q) non-seasonal part
#         seasonal_order (tuple): (P,D,Q,s) seasonal part
#     """
    
#     print("------ SARIMA MODEL STARTED ------")
#     print(f"Non-seasonal order (p,d,q): {order}")
#     print(f"Seasonal order (P,D,Q,s): {seasonal_order}\n")

#     # --- Prepare series ---
#     y_train = train_data[feature_name].dropna()
#     y_test = test_data[feature_name].dropna()

#     # --- Fit SARIMA model ---
#     model = SARIMAX(y_train,
#                     order=order,
#                     seasonal_order=seasonal_order,
#                     enforce_stationarity=False,
#                     enforce_invertibility=False)
#     sarima_fit = model.fit(disp=False)

#     print(sarima_fit.summary())
#     print("\nModel fitted successfully!\n")

#     # --- Forecast ---
#     start = len(y_train)
#     end = start + len(y_test) - 1
#     predictions = sarima_fit.predict(start=start, end=end, dynamic=False)

#     # --- Evaluation metrics ---
#     mse = mean_squared_error(y_test, predictions)
#     mae = mean_absolute_error(y_test, predictions)
#     rmse = math.sqrt(mse)

#     print(f"Evaluation Metrics for SARIMA:")
#     print(f"MAE:  {mae:.3f}")
#     print(f"MSE:  {mse:.3f}")
#     print(f"RMSE: {rmse:.3f}\n")

#     # --- Plot Actual vs Predicted ---
#     plt.figure(figsize=(12, 6))
#     plt.plot(y_train.index, y_train, label="Train", color='blue')
#     plt.plot(y_test.index, y_test, label="Test", color='green')
#     plt.plot(y_test.index, predictions, label="Predicted", color='red', linestyle='--')
#     plt.title("SARIMA Forecast vs Actual")
#     plt.xlabel("Date")
#     plt.ylabel(feature_name)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

#     # --- Future forecast (optional, next 30 days) ---
#     future_forecast = sarima_fit.get_forecast(steps=30)
#     forecast_mean = future_forecast.predicted_mean
#     conf_int = future_forecast.conf_int()

#     plt.figure(figsize=(12, 6))
#     plt.plot(y_train.index, y_train, label="Train")
#     plt.plot(y_test.index, y_test, label="Test")
#     plt.plot(forecast_mean.index, forecast_mean, label="30-Day Forecast", color='orange')
#     plt.fill_between(conf_int.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1],
#                      color='orange', alpha=0.3)
#     plt.title("SARIMA Future 30-Day Forecast")
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

#     print("------ SARIMA MODEL FINISHED ------")
#     return sarima_fit, predictions, forecast_mean
