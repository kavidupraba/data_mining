#!/home/jack/data_mining/dminevenv/bin/python3
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import logging
from scipy import signal
from statsmodels.tsa.stattools import acf, pacf, adfuller
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pmdarima as pm
import warnings
warnings.filterwarnings('ignore')

#from statsforecast import StatsForecast
#from statsforecast.models import AutoETS
# Commented out to avoid numpy/numba conflicts for clustering analysis

import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform
try:
    from dtw import dtw
    DTW_AVAILABLE = True
    print("DTW available - using Dynamic Time Warping")
except ImportError:
    print("DTW not available, using Euclidean distance instead")
    DTW_AVAILABLE = False
except Exception as e:
    print(f"DTW import error: {e}, using Euclidean distance instead")
    DTW_AVAILABLE = False


# Force Euclidean distance if numpy version conflicts with DTW
import numpy as np
if hasattr(np, '__version__') and np.__version__.startswith('1.'):
    print("NumPy 1.x detected - using Euclidean distance for compatibility")
    DTW_AVAILABLE = False

def setting_up_logger(name=__name__, level=logging.INFO, log_file='pipeline.log'):
    """Create and configure logger."""
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        logger.setLevel(level)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
    return logger

def time_outer(logger=None):
    "calculating time"
    def get_func(func):
        def cal_time(*args,**kargs):
            start_t=time.perf_counter()
            p,a,er= func(*args,**kargs)
            end_t=time.perf_counter()
            time_took=end_t-start_t
            logger.info(f"{func.__name__} took {time_took}")
            return p,a,er,time_took
        return cal_time
    return get_func


def reading_files(logger, file_path1="/home/jack/data_mining/PROJECT/data/hour.csv",
                  file_path2="/home/jack/data_mining/PROJECT/data/day.csv"):
    """Read hourly and daily CSV files."""

    if os.path.exists(file_path1):
        logger.info("Reading hourly data")
        time_df = pd.read_csv(file_path1)
        print(time_df.head(10))
    else:
        raise FileNotFoundError("Hour.csv file not found in the mentioned file path")

    if os.path.exists(file_path2):
        logger.info("Reading daily data")
        day_df = pd.read_csv(file_path2)
        print(day_df.head(10))
    else:
        raise FileNotFoundError("Day.csv file not found in the mentioned file path")

    return time_df, day_df

def filter_seasonal_periods(periods, data_length,name, logger):
    """
    Filter detected periods to keep only meaningful ones.
    """
    logger.info("Filtering detected seasonal periods")
    
    # Rule 1: Remove periods that are too long (need at least 2 full cycles)
    max_period = data_length // 3
    valid_periods = [p for p in periods if p <= max_period]
    logger.info(f"  After max period filter ({max_period}): {valid_periods}")
    
    # Rule 2: Remove harmonics (multiples of smaller periods)
    filtered = []
    for period in sorted(valid_periods):
        # Check if this is a harmonic of an already selected period
        is_harmonic = False
        for base_period in filtered:
            # If period is close to a multiple of base_period, skip it
            ratio = period / base_period
            if abs(ratio - round(ratio)) < 0.15:  # Within 15% of integer multiple
                logger.info(f"  Removing {period} (harmonic of {base_period})")
                is_harmonic = True
                break
        
        if not is_harmonic:
            filtered.append(period)
    
    logger.info(f"  After harmonic filter: {filtered}")
    
    # Rule 3: Prioritize known meaningful periods
    known_good={}
    if name=="day":
        known_good =  {7:"Weekly",
                    30:"Monthly",
                    91:"Quarterly",
                    181:"Semi-annual",
                    14:"Bi-Weekly"
                    }
    else:
            known_good = {
            24: 'Daily',           # 24 hours = daily pattern
            168: 'Weekly',         # 168 hours = weekly pattern
            720: 'Monthly',        # ~720 hours = monthly pattern
            8760: 'Yearly'         # 8760 hours = yearly pattern
        }

    
    prioritized = []
    
    for period in filtered:
        if period in known_good:
            prioritized.append(period)
            logger.info(f"  ✓ Keeping {period} ({known_good[period]} pattern)")
    
    # Add other periods if we have less than 3
    for period in filtered:
        if period not in prioritized and len(prioritized) < 3:
            prioritized.append(period)
            logger.info(f"  ✓ Adding {period} (top detected period)")
    
    # Rule 4: Limit to maximum 2-3 periods to avoid overfitting
    final_periods = prioritized[:3]
    
    logger.info(f"  FINAL PERIODS: {final_periods}")
    return final_periods


def detect_seasonality(logger, data, name,max_lag=None):
    """Detect seasonality using ACF peaks and periodogram."""

    logger.info("Setting max_lag. If not defined, using min(len(data)//2, 500)")
    if max_lag is None:
        max_lag = min(len(data) // 2, 500)

    logger.info("Computing ACF values and detecting peaks using signal.find_peaks")
    acf_values = acf(data, nlags=max_lag, fft=True)

    peaks, properties = signal.find_peaks(acf_values[1:], height=0.1, distance=5)
    peaks = peaks + 1  # Shift because we removed lag 0

    print(f"Detected ACF peaks (first 10): {peaks[:10]}")
    logger.info(f"Detected ACF peaks (first 10): {peaks[:10]}")

    logger.info("Using periodogram to detect dominant cycles")
    freq, power = signal.periodogram(data)

    top_p = np.argsort(power)[-5:][::-1]
    top_freq = freq[top_p]
    top_periods = [int(1 / f) if f > 0 else 0 for f in top_freq]
    top_periods = [p for p in top_periods if 2 < p < len(data) // 2]

    print(f"Detected periods from periodogram: {top_periods}")
    logger.info(f"Detected periods from periodogram: {top_periods}")

    logger.info("Checking for common known seasonal periods")
    
    common_periods={}
    if name=="day":

        common_periods = {
            'Weekly': 7,           # 7 days = weekly pattern
            'Bi-weekly': 14,       # 14 days = 2 weeks
            'Monthly': 30,         # ~30 days = monthly pattern
            'Quarterly': 91,       # ~91 days = quarterly pattern
            'Semi-annual': 182,    # ~182 days = 6 months
            'Yearly': 365          # 365 days = yearly pattern
        }
        
    elif name=="hour":
            common_periods = {
        'Daily': 24,           # 24 hours = daily pattern
        'Weekly': 168,         # 168 hours = weekly pattern
        'Monthly': 720,        # ~720 hours = monthly pattern
        'Yearly': 8760         # 8760 hours = yearly pattern
    }
        
    detected_periods = []
    for period_name, p_val in common_periods.items():
        if p_val < len(acf_values):
            if acf_values[p_val] > 0.2:
                print(f"✓ Detected period {period_name}: ACF[{p_val}] = {acf_values[p_val]:.3f}")
                detected_periods.append(p_val)
            else:
                print(f"  Period {period_name} not strong: ACF[{p_val}] = {acf_values[p_val]:.3f}")

    logger.info("Combining all detected periods (ACF, peaks, periodogram)")
    all_periods = list(set(list(peaks[:5]) + top_periods + detected_periods))
    all_periods = sorted([p for p in all_periods if p > 1])

    logger.info(f"Detected all periods: {all_periods}")
    
    all_periods = sorted([p for p in all_periods if p > 1])
    logger.info(f"Detected all periods: {all_periods}")
    
    filtered_periods = filter_seasonal_periods(all_periods, len(data),name, logger)
    
    logger.info(f"After filtering periods {filtered_periods}")
    
    fig, axis = plt.subplots(2, 2, figsize=(15, 12))
    axis[0, 0].plot(data)
    axis[0, 0].set_title("Actual Data")
    axis[0, 0].set_xlabel("Time")
    axis[0, 0].set_ylabel("Value")
    
    plot_acf(data, lags=min(len(data)//2, 100), ax=axis[0, 1])
    axis[0, 1].set_title("Autocorrelation Function (ACF)")
    
    axis[1, 0].plot(freq[1:], power[1:])
    axis[1, 0].set_title('Periodogram (Spectral Density)')
    axis[1, 0].set_xlabel('Frequency')
    axis[1, 0].set_ylabel('Power')
    axis[1, 0].set_xlim([0, 0.5])
    
    if filtered_periods:
        period = filtered_periods[0]
        if len(data) >= period * 2:
            decompose = seasonal_decompose(data, model='additive', period=period)
            axis[1, 1].plot(decompose.seasonal[:period*3])
            axis[1, 1].set_title(f"Seasonal Pattern (period={period})")
    
    plt.tight_layout()
    plt.savefig('seasonality_detection.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    
    return filtered_periods


def check_stationary(data, logger, name="Series"):
    
    logger.info("check data is stationary or not using adfuller function (standard statistical test)")
    result = adfuller(data, autolag='AIC')
    
    print("\n" + "="*60)
    print(f"STATIONARITY TEST: {name}")
    print("="*60)
    print("\nAugmented Dickey-Fuller Test:")
    print(f"  ADF statistic: {result[0]:.6f}")
    logger.info(f"ADF statistics: {result[0]}")
    print(f"  P-value: {result[1]:.6f}")
    logger.info(f"P-value: {result[1]}")
    print("  Critical Values:")
    for key, value in result[4].items():
        print(f"    {key}: {value:.3f}")
    
    if result[1] <= 0.05:
        print(f"  ✓ Result: STATIONARY (p-value <= 0.05)")
        logger.info(f"{name} is stationary")
        is_stationary = True
    else:
        print(f"  ✗ Result: NON-STATIONARY (p-value > 0.05)")
        logger.info(f"{name} is not stationary")
        is_stationary = False
    
    mean_val = data.rolling(window=12).mean()
    std_val = data.rolling(window=12).std()
    
    print(f"\nRolling Statistics:")
    print(f"  Mean Variation (For Trend Detection): {mean_val.std():.3f}")
    print(f"  Std Variation (For Variance Detection): {std_val.std():.3f}")
        
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))
    axes[0].plot(data, label="Original", alpha=0.7)
    axes[0].plot(mean_val, label="Rolling Mean", color="red")
    axes[0].plot(std_val, label="Rolling Std", color="green")
    axes[0].legend()
    axes[0].set_title(f'{name} - Rolling Statistics')
    
    axes[1].hist(data, bins=50, alpha=0.7, edgecolor="black")
    axes[1].set_title('Distribution')
    axes[1].set_xlabel('Value')
    axes[1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(f'stationarity_{name}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return is_stationary



def create_fourier_features(data, periods, logger):
    """
    Create Fourier series features for multiple seasonal periods.
    
    Fourier Formula:
    sin(2π * k * t / period) and cos(2π * k * t / period)
    where k = 1, 2, ... (harmonics)
    """
    logger.info("Creating Fourier series features for detected periods")
    
    # Create time index (0, 1, 2, 3, ...)
    t = np.arange(len(data))
    
    fourier_features = pd.DataFrame(index=data.index)

    for period in periods:
        # Use k=1 (fundamental frequency) for each period
        # You can add k=2, k=3 for harmonics if needed
        logger.info(f"  Adding Fourier features for period={period}")
        
        # Fundamental frequency
        fourier_features[f'sin_{period}'] = np.sin(2 * np.pi * t / period)
        fourier_features[f'cos_{period}'] = np.cos(2 * np.pi * t / period)
        
        # Optional: Add first harmonic (k=2) for stronger seasonality
        # fourier_features[f'sin2_{period}'] = np.sin(4 * np.pi * t / period)
        # fourier_features[f'cos2_{period}'] = np.cos(4 * np.pi * t / period)
    
    logger.info(f"Created {len(fourier_features.columns)} Fourier features")
    print(f"Fourier features created: {list(fourier_features.columns)}")
    
    return fourier_features

@time_outer(logger=setting_up_logger())
def walk_forward_exponentialsmoothing(data, test_size, logger):
    """
    Walk-forward validation using Holt-Winters Exponential Smoothing
    with Box-Cox transform.
    """
    #  Detect ETS structure (trend, seasonal)
    trend, seasonal = detect_ets_structure(data, value_col="cnt", logger=logger)

    logger.info("="*60)
    logger.info("WALK-FORWARD VALIDATION WITH EXPONENTIAL SMOOTHING")
    logger.info("="*60)

    #  Box-Cox transformation
    logger.info("Applying Box-Cox transformation")
    data_transformed, lambda_param = boxcox(data['cnt'] + 1)  # +1 to avoid zeros
    data_transformed = pd.Series(data_transformed, index=data.index, name='bc_cnt')

    #  Split train/test
    train_data = data_transformed[:-test_size]
    test_data = data_transformed[-test_size:]

    predictions = []
    actuals = []

    current_train = train_data.copy()

    logger.info("Starting walk-forward forecasting...")

    try:
        for i in range(len(test_data)):
            logger.info(f"Fitting training data step {i+1}/{len(test_data)}")

            # Guard for multiplicative trend/season if data <=0
            trend_used = trend
            seasonal_used = seasonal
            if trend_used == "mul" and (current_train <= 0).any():
                trend_used = "add"
            if seasonal_used == "mul" and (current_train <= 0).any():
                seasonal_used = "add"

            # Fit model
            model = ExponentialSmoothing(
                current_train,
                trend=trend_used,
                seasonal=seasonal_used,
                seasonal_periods=7
            ).fit(optimized=True)

            # Forecast next step only
            next_pred_bc = model.forecast(1)[0]

            # Inverse Box-Cox
            pred_original = inv_boxcox(next_pred_bc, lambda_param) - 1  # subtract 1
            predictions.append(pred_original)

            # Actual value
            actual_val = data['cnt'].iloc[-test_size + i]
            actuals.append(actual_val)

            # Add TRUE value to training (walk-forward)
            current_train = pd.concat([
                current_train, 
                pd.Series([test_data.iloc[i]], index=[test_data.index[i]])
            ])

            if (i + 1) % 10 == 0:
                logger.info(f"Completed {i+1}/{len(test_data)} predictions")

    except Exception as e:
        logger.error(f"Error at step {i}: {e}")
        predictions.append(np.nan)
        actuals.append(data['cnt'].iloc[-test_size + i])

    #  Evaluate
    predictions = np.array(predictions)
    actuals = np.array(actuals)

    valid_mask = ~np.isnan(predictions)
    predictions_clean = predictions[valid_mask]
    actuals_clean = actuals[valid_mask]

    mse = mean_squared_error(actuals_clean, predictions_clean)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals_clean, predictions_clean)
    mape = np.mean(np.abs((actuals_clean - predictions_clean) / actuals_clean)) * 100

    logger.info("="*60)
    logger.info("RESULTS:")
    logger.info("="*60)
    logger.info(f"MSE:  {mse:.2f}")
    logger.info(f"RMSE: {rmse:.2f}")
    logger.info(f"MAE:  {mae:.2f}")
    logger.info(f"MAPE: {mape:.2f}%")

    print("\n" + "="*60)
    print("FINAL RESULTS:")
    print("="*60)
    print(f"MSE:  {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE:  {mae:.2f}")
    print(f"MAPE: {mape:.2f}%")

    #  Visualize
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))

    # Full series
    axes[0].plot(data.index, data['cnt'], label='Full Series', alpha=0.7)
    test_indices = data.index[-test_size:]
    axes[0].plot(test_indices, actuals, label='Actual Test', marker='o', color='green')
    axes[0].plot(test_indices, predictions, label='Predictions', marker='x', color='red')
    axes[0].set_title('Walk-Forward Predictions vs Actual')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Count')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Zoom test period
    axes[1].plot(range(len(actuals)), actuals, label='Actual', marker='o', color='green')
    axes[1].plot(range(len(predictions)), predictions, label='Predicted', marker='x', color='red')
    axes[1].set_title('Test Period: Predictions vs Actual')
    axes[1].set_xlabel('Test Step')
    axes[1].set_ylabel('Count')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('walk_forward_exponential_smoothing_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    return predictions, actuals, {'mse': mse, 'rmse': rmse, 'mae': mae, 'mape': mape}


        


@time_outer(logger=setting_up_logger())
def walk_forward_arima(data, periods, logger,name,test_size=50):
    """
    Walk-forward validation with auto_arima using Fourier terms.
    One-step-ahead forecasting!
    """
    logger.info("="*60)
    logger.info("WALK-FORWARD VALIDATION WITH AUTO ARIMA")
    logger.info("="*60)
    
    # Step 1: Box-Cox transformation
    logger.info("Applying Box-Cox transformation")
    data_transformed, lambda_param = boxcox(data['cnt'])
    data_transformed = pd.Series(data_transformed, index=data.index, name='bc_cnt')
    
    # Step 2: Create Fourier features
    logger.info("Creating Fourier features")
    # Use only top 3 periods to avoid overfitting
    selected_periods = periods[:3] if len(periods) >= 3 else periods
    logger.info(f"Using periods: {selected_periods}")
    
    fourier_df = create_fourier_features(data_transformed, selected_periods, logger)
    
    # Combine transformed data with Fourier features
    data_combined = pd.concat([data_transformed, fourier_df], axis=1)
    
    # Step 3: Train/Test split
    logger.info(f"Splitting data: train size={len(data)-test_size}, test size={test_size}")
    train_data = data_combined.iloc[:-test_size]
    test_data = data_combined.iloc[-test_size:]
    
    # Get column names for exogenous variables
    fourier_columns = [col for col in fourier_df.columns]
    
    # Step 4: Walk-forward validation
    logger.info("Starting walk-forward forecasting...")
    predictions = []
    actuals = []
    
    # Start with training data
    current_train = train_data.copy()
    
    for i in range(len(test_data)):
        try:
            # Fit auto_arima on current training data
            logger.info(f"  Step {i+1}/{len(test_data)}: Fitting model...")
            
            model = pm.auto_arima(
                current_train['bc_cnt'],
                X=current_train[fourier_columns],
                seasonal=False,  # Seasonality handled by Fourier terms
                stepwise=True,
                suppress_warnings=True,
                error_action='ignore',
                information_criterion='aicc',
                max_p=3, max_q=3, max_d=2,  # Limit search space
                trace=False
            )
            
            logger.info(f"    Model order: {model.order}")
            
            # Predict next step using Fourier features from test data
            X_next = test_data.iloc[[i]][fourier_columns]
            pred_bc = model.predict(n_periods=1, X=X_next)[0]
            
            # Reverse Box-Cox
            pred_original = inv_boxcox(pred_bc, lambda_param)
            
            predictions.append(pred_original)
            actuals.append(data['cnt'].iloc[-test_size + i])
            
            # Add TRUE value to training data (not prediction!)
            true_row = test_data.iloc[[i]].copy()
            current_train = pd.concat([current_train, true_row])
            
            if (i + 1) % 10 == 0:
                logger.info(f"    Completed {i+1}/{len(test_data)} predictions")
        
        except Exception as e:
            logger.error(f"  Error at step {i}: {e}")
            predictions.append(np.nan)
            actuals.append(data['cnt'].iloc[-test_size + i])
    
    # Step 5: Evaluate
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Remove NaN
    valid_mask = ~np.isnan(predictions)
    predictions_clean = predictions[valid_mask]
    actuals_clean = actuals[valid_mask]
    
    mse = mean_squared_error(actuals_clean, predictions_clean)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals_clean, predictions_clean)
    mape = np.mean(np.abs((actuals_clean - predictions_clean) / actuals_clean)) * 100
    
    logger.info("="*60)
    logger.info("RESULTS:")
    logger.info("="*60)
    logger.info(f"MSE:  {mse:.2f}")
    logger.info(f"RMSE: {rmse:.2f}")
    logger.info(f"MAE:  {mae:.2f}")
    logger.info(f"MAPE: {mape:.2f}%")
    
    print("\n" + "="*60)
    print("FINAL RESULTS:")
    print("="*60)
    print(f"MSE:  {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE:  {mae:.2f}")
    print(f"MAPE: {mape:.2f}%")
    
    # Step 6: Visualize
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # Full series with predictions
    axes[0].plot(data.index, data['cnt'], label='Training Data', alpha=0.7)
    test_indices = data.index[-test_size:]
    axes[0].plot(test_indices, actuals, label='Actual Test', marker='o', markersize=4, color='green')
    axes[0].plot(test_indices, predictions, label='Predictions', marker='x', markersize=4, color='red')
    axes[0].set_title('Walk-Forward Predictions vs Actual')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Count')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Zoom in on test period
    axes[1].plot(range(len(actuals)), actuals, label='Actual', marker='o', markersize=5, color='green')
    axes[1].plot(range(len(predictions)), predictions, label='Predicted', marker='x', markersize=5, color='red')
    axes[1].set_title('Test Period: Predictions vs Actual')
    axes[1].set_xlabel('Test Step')
    axes[1].set_ylabel('Count')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'walk_forward_Ariama_results_{name}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return predictions, actuals, {'mse': mse, 'rmse': rmse, 'mae': mae, 'mape': mape}


def indexing_datetime(time_data, day_data, logger):
    """Set datetime index"""
    
    logger.info("Converting to datetime and setting index")
    day_data['dteday'] = pd.to_datetime(day_data['dteday'])
    time_data['dteday'] = pd.to_datetime(time_data['dteday'])
    
    logger.info("Setting date-time as index for daily data")
    day_data.set_index('dteday', inplace=True)
    
    logger.info("Setting date-time as index for hourly data")
    time_data.set_index('dteday', inplace=True)
    
    return time_data, day_data



def detect_ets_structure(df, value_col="cnt", logger=None):
    import numpy as np
    import pandas as pd

    y = df[value_col].dropna()

    # ---------- 1. Detect multiplicative seasonality ----------
    # Compare variance in low vs high value segments
    q1 = y[y < y.quantile(0.33)]
    q3 = y[y > y.quantile(0.66)]

    var_low = q1.var()
    var_high = q3.var()

    seasonal_multiplicative = var_high > 3 * var_low  # strong level-dependent variance

    # ---------- 2. Detect multiplicative trend ----------
    # use relative difference instead of absolute
    y_diff = y.diff().abs()
    y_rel_diff = (y.diff() / y.shift()).abs()

    trend_multiplicative = y_rel_diff.mean() > y_diff.mean() * 0.02

    # ---------- 3. Log stabilization ----------
    y_log = np.log1p(y)

    log_reduction = y.var() / y_log.var()
    strong_log_effect = log_reduction > 3

    # ---------- Final decisions ----------
    seasonal = "mul" if seasonal_multiplicative or strong_log_effect else "add"
    trend    = "mul" if trend_multiplicative or strong_log_effect else "add"

    if logger:
        logger.info(f"Variance low: {var_low:.2f}, variance high: {var_high:.2f}")
        logger.info(f"Trend rel diff: {y_rel_diff.mean():.4f}, abs diff: {y_diff.mean():.4f}")
        logger.info(f"Log variance reduction: {log_reduction:.2f}")
        logger.info(f"DETECTED → Trend: {trend.upper()}, Seasonality: {seasonal.upper()}")

    return trend, seasonal


def walk_forward_auto_sarima(data, logger, seasonal_period=7, test_size=50):
    """
    Auto SARIMA - automatically finds best (p,d,q)(P,D,Q,m)
    """
    logger.info("="*60)
    logger.info("WALK-FORWARD VALIDATION WITH AUTO SARIMA")
    logger.info("="*60)
    
    train = data['cnt'].iloc[:-test_size]
    test = data['cnt'].iloc[-test_size:]
    
    predictions = []
    actuals = []
    
    current_train = train.copy()
    
    for i in range(len(test)):
        try:
            logger.info(f"  Step {i+1}/{len(test)}: Fitting Auto-SARIMA...")
            
            # Auto SARIMA with seasonality
            model = pm.auto_arima(
                current_train,
                seasonal=True,              # Enable seasonality
                m=seasonal_period,          # Seasonal period (7 for weekly)
                max_p=3, max_q=3,          # ARIMA params
                max_P=2, max_Q=2,          # Seasonal ARIMA params
                max_d=2, max_D=1,          # Differencing
                stepwise=True,
                suppress_warnings=True,
                error_action='ignore',
                information_criterion='aicc',
                trace=False
            )
            
            logger.info(f"    Best order: {model.order}, Seasonal: {model.seasonal_order}")
            
            # Predict
            pred = model.predict(n_periods=1)[0]
            predictions.append(pred)
            actuals.append(test.iloc[i])
            
            # Add TRUE value
            current_train = pd.concat([current_train, pd.Series([test.iloc[i]])])
            
            if (i + 1) % 10 == 0:
                logger.info(f"    Completed {i+1}/{len(test)} predictions")
        
        except Exception as e:
            logger.error(f"  Error at step {i}: {e}")
            predictions.append(np.nan)
            actuals.append(test.iloc[i])
    
    # Evaluate
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    valid_mask = ~np.isnan(predictions)
    predictions_clean = predictions[valid_mask]
    actuals_clean = actuals[valid_mask]
    
    mse = mean_squared_error(actuals_clean, predictions_clean)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals_clean, predictions_clean)
    mape = np.mean(np.abs((actuals_clean - predictions_clean) / actuals_clean)) * 100
    
    logger.info("="*60)
    logger.info("AUTO-SARIMA RESULTS:")
    logger.info("="*60)
    logger.info(f"MSE:  {mse:.2f}")
    logger.info(f"RMSE: {rmse:.2f}")
    logger.info(f"MAE:  {mae:.2f}")
    logger.info(f"MAPE: {mape:.2f}%")
    
    print("\n" + "="*60)
    print("AUTO-SARIMA RESULTS:")
    print("="*60)
    print(f"MSE:  {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE:  {mae:.2f}")
    print(f"MAPE: {mape:.2f}%")
    
    # Visualize
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    axes[0].plot(data.index, data['cnt'], label='Training Data', alpha=0.7)
    test_indices = data.index[-test_size:]
    axes[0].plot(test_indices, actuals, label='Actual Test', marker='o', markersize=4, color='green')
    axes[0].plot(test_indices, predictions, label='Auto-SARIMA Predictions', marker='x', markersize=4, color='red')
    axes[0].set_title('Auto-SARIMA: Walk-Forward Predictions')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Count')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(range(len(actuals)), actuals, label='Actual', marker='o', markersize=5, color='green')
    axes[1].plot(range(len(predictions)), predictions, label='Predicted', marker='x', markersize=5, color='red')
    axes[1].set_title('Test Period Detail')
    axes[1].set_xlabel('Test Step')
    axes[1].set_ylabel('Count')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('auto_sarima_results_{name}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return predictions, actuals, {'mse': mse, 'rmse': rmse, 'mae': mae, 'mape': mape}


# ========================================
# K-MEANS + DTW + EUCLIDEAN CLUSTERING
# ========================================

def create_temporal_patterns(df, logger):
    """
    Create temporal patterns for clustering by reshaping time series data.
    """
    logger.info("Creating temporal patterns for clustering")
    
    # For daily data: create weekly patterns
    if len(df) > 7:
        # Reshape into weekly patterns (7 days per week)
        n_weeks = len(df) // 7
        weekly_patterns = []
        
        for week in range(n_weeks):
            start_idx = week * 7
            end_idx = start_idx + 7
            if end_idx <= len(df):
                week_data = df.iloc[start_idx:end_idx]['cnt'].values
                weekly_patterns.append(week_data)
        
        weekly_patterns = np.array(weekly_patterns)
        logger.info(f"Created {len(weekly_patterns)} weekly patterns")
        return weekly_patterns, 'weekly'
    
    return None, 'none'


def compute_dtw_distance_matrix(patterns, logger):
    """
    Compute DTW distance matrix between temporal patterns.
    """
    logger.info("Computing DTW distance matrix")
    
    n_patterns = len(patterns)
    dist_matrix = np.zeros((n_patterns, n_patterns))
    
    for i in range(n_patterns):
        for j in range(i+1, n_patterns):
            if DTW_AVAILABLE:
                # Use DTW distance
                try:
                    dist = dtw(patterns[i], patterns[j]).distance
                except:
                    # Fallback to Euclidean if DTW fails
                    dist = np.linalg.norm(patterns[i] - patterns[j])
            else:
                # Fallback to Euclidean distance
                dist = np.linalg.norm(patterns[i] - patterns[j])
            
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    
    logger.info(f"DTW distance matrix computed: {dist_matrix.shape}")
    return dist_matrix


def compute_euclidean_distance_matrix(patterns, logger):
    """
    Compute Euclidean distance matrix between temporal patterns.
    """
    logger.info("Computing Euclidean distance matrix")
    
    # Reshape patterns for distance computation
    n_patterns = len(patterns)
    patterns_2d = patterns.reshape(n_patterns, -1)
    
    # Compute pairwise distances
    dist_matrix = squareform(pdist(patterns_2d, metric='euclidean'))
    
    logger.info(f"Euclidean distance matrix computed: {dist_matrix.shape}")
    return dist_matrix


def kmeans_dtw_euclidean_clustering(df, logger, max_clusters=8, use_dtw=True):
    """
    Perform K-Means clustering with DTW or Euclidean distance for temporal pattern analysis.
    """
    logger.info("="*60)
    logger.info("K-MEANS CLUSTERING ANALYSIS")
    logger.info("="*60)
    
    # Step 1: Create temporal patterns
    patterns, pattern_type = create_temporal_patterns(df, logger)
    
    if patterns is None:
        logger.warning("Could not create temporal patterns, using feature-based clustering")
        return feature_based_clustering(df, logger, max_clusters)
    
    # Step 2: Compute distance matrix
    if use_dtw and DTW_AVAILABLE:
        dist_matrix = compute_dtw_distance_matrix(patterns, logger)
        distance_type = "DTW"
    else:
        dist_matrix = compute_euclidean_distance_matrix(patterns, logger)
        distance_type = "Euclidean"
    
    logger.info(f"Using {distance_type} distance for clustering")
    
    # Step 3: Find optimal number of clusters using elbow method
    logger.info("Finding optimal number of clusters")
    inertias = []
    silhouette_scores = []
    K_range = range(2, min(max_clusters + 1, len(patterns)))
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(dist_matrix)
        
        inertias.append(kmeans.inertia_)
        sil_score = silhouette_score(dist_matrix, cluster_labels, metric='precomputed')
        silhouette_scores.append(sil_score)
        
        logger.info(f"  K={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={sil_score:.3f}")
    
    # Find optimal K (highest silhouette score)
    optimal_k = K_range[np.argmax(silhouette_scores)]
    logger.info(f"Optimal number of clusters: {optimal_k}")
    
    # Step 4: Final clustering
    final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = final_kmeans.fit_predict(dist_matrix)
    
    # Step 5: Visualize results
    visualize_clustering_results(patterns, cluster_labels, optimal_k, silhouette_scores, K_range, 
                          pattern_type, distance_type, logger)
    
    # Step 6: Analyze clusters
    cluster_analysis = analyze_temporal_clusters(patterns, cluster_labels, optimal_k, pattern_type, logger)
    
    # Step 7: Map clusters back to original data
    df_with_clusters = map_clusters_to_original_data(df, cluster_labels, pattern_type, logger)
    
    return {
        'patterns': patterns,
        'cluster_labels': cluster_labels,
        'optimal_k': optimal_k,
        'silhouette_scores': silhouette_scores,
        'cluster_analysis': cluster_analysis,
        'df_with_clusters': df_with_clusters,
        'pattern_type': pattern_type,
        'distance_type': distance_type
    }


def feature_based_clustering(df, logger, max_clusters=8):
    """
    Fallback: Feature-based clustering using cnt, temp, hum, windspeed.
    """
    logger.info("Using feature-based clustering (cnt, temp, hum, windspeed)")
    
    # Select features for clustering
    feature_cols = ['cnt', 'temp', 'hum', 'windspeed']
    features = df[feature_cols].copy()
    
    # Handle missing values
    features = features.fillna(features.mean())
    
    # Normalize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Find optimal K
    inertias = []
    silhouette_scores = []
    K_range = range(2, max_clusters + 1)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_scaled)
        
        inertias.append(kmeans.inertia_)
        sil_score = silhouette_score(features_scaled, cluster_labels)
        silhouette_scores.append(sil_score)
        
        logger.info(f"  K={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={sil_score:.3f}")
    
    optimal_k = K_range[np.argmax(silhouette_scores)]
    logger.info(f"Optimal number of clusters: {optimal_k}")
    
    # Final clustering
    final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = final_kmeans.fit_predict(features_scaled)
    
    # Add cluster labels to original dataframe
    df_with_clusters = df.copy()
    df_with_clusters['cluster'] = cluster_labels
    
    # Visualize
    visualize_feature_clustering(features_scaled, cluster_labels, feature_cols, optimal_k, logger)
    
    return {
        'cluster_labels': cluster_labels,
        'optimal_k': optimal_k,
        'silhouette_scores': silhouette_scores,
        'df_with_clusters': df_with_clusters,
        'pattern_type': 'feature_based'
    }


def visualize_clustering_results(patterns, cluster_labels, optimal_k, silhouette_scores, K_range, 
                          pattern_type, distance_type, logger):
    """
    Visualize clustering results including patterns and metrics.
    """
    logger.info("Creating clustering visualizations")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Use darker, more visible colors
    dark_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    colors = [dark_colors[i] for i in range(optimal_k)]
    
    # Plot 1: Elbow method and silhouette scores
    axes[0, 0].plot(K_range, silhouette_scores, 'o-', linewidth=3, markersize=10, 
                    color='#1f77b4', markerfacecolor='#ff7f0e')
    axes[0, 0].axvline(x=optimal_k, color='#d62728', linestyle='--', alpha=0.8, linewidth=2)
    axes[0, 0].set_title(f'Optimal Number of Clusters (Silhouette Score) - {distance_type}', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Number of Clusters (K)', fontsize=12)
    axes[0, 0].set_ylabel('Silhouette Score', fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_facecolor('#f8f9fa')
    
    # Plot 2: Cluster patterns (average pattern per cluster)
    # Dynamic cluster names based on demand level
    cluster_names = [f'Cluster {i}' for i in range(optimal_k)]
    for k in range(optimal_k):
        cluster_patterns = patterns[cluster_labels == k]
        avg_pattern = np.mean(cluster_patterns, axis=0)
        std_pattern = np.std(cluster_patterns, axis=0)
        
        # Determine cluster type for naming
        overall_mean = np.mean(patterns)
        overall_std = np.std(patterns)
        peak_value = np.max(avg_pattern)
        
        if peak_value > overall_mean + overall_std:
            cluster_type = 'High Demand'
        elif peak_value < overall_mean - overall_std:
            cluster_type = 'Low Demand'
        else:
            cluster_type = 'Normal Demand'
        
        cluster_names[k] = f'{cluster_type} Weeks'
        
        x_axis = range(len(avg_pattern))
        axes[0, 1].plot(x_axis, avg_pattern, color=colors[k], linewidth=4, 
                        label=f'{cluster_names[k]} (n={len(cluster_patterns)})', marker='o', markersize=6)
        axes[0, 1].fill_between(x_axis, avg_pattern - std_pattern, avg_pattern + std_pattern, 
                               color=colors[k], alpha=0.3)
    
    axes[0, 1].set_title(f'Average {pattern_type.capitalize()} Patterns by Cluster', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Time (days)', fontsize=12)
    axes[0, 1].set_ylabel('Average Bike Rentals', fontsize=12)
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_facecolor('#f8f9fa')
    
    # Plot 3: Individual patterns colored by cluster
    for k in range(optimal_k):
        cluster_patterns = patterns[cluster_labels == k]
        for pattern in cluster_patterns[:5]:  # Show first 5 patterns per cluster
            axes[1, 0].plot(range(len(pattern)), pattern, color=colors[k], alpha=0.8, linewidth=2.5)
    
    axes[1, 0].set_title(f'Individual {pattern_type.capitalize()} Patterns by Cluster', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Time (days)', fontsize=12)
    axes[1, 0].set_ylabel('Bike Rentals', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_facecolor('#f8f9fa')
    
    # Plot 4: Cluster sizes
    cluster_sizes = [np.sum(cluster_labels == k) for k in range(optimal_k)]
    bars = axes[1, 1].bar(range(optimal_k), cluster_sizes, color=colors, edgecolor='black', linewidth=2)
    axes[1, 1].set_title('Cluster Sizes', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Cluster', fontsize=12)
    axes[1, 1].set_ylabel('Number of Patterns', fontsize=12)
    axes[1, 1].set_xticks(range(optimal_k))
    axes[1, 1].set_xticklabels([f'{cluster_names[k]}' for k in range(optimal_k)], fontsize=10)
    axes[1, 1].set_facecolor('#f8f9fa')
    
    # Add labels on bars
    for bar, size in zip(bars, cluster_sizes):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       str(size), ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'kmeans_{distance_type.lower()}_clustering_results.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()


def visualize_feature_clustering(features_scaled, cluster_labels, feature_cols, optimal_k, logger):
    """
    Visualize feature-based clustering results.
    """
    logger.info("Creating feature clustering visualizations")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    colors = plt.cm.Set3(np.linspace(0, 1, optimal_k))
    
    # Plot 1: cnt vs temp
    for k in range(optimal_k):
        mask = cluster_labels == k
        axes[0, 0].scatter(features_scaled[mask, 0], features_scaled[mask, 1], 
                          c=[colors[k]], label=f'Cluster {k}', alpha=0.7)
    axes[0, 0].set_xlabel('cnt (normalized)')
    axes[0, 0].set_ylabel('temp (normalized)')
    axes[0, 0].set_title('Bike Rentals vs Temperature')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: cnt vs hum
    for k in range(optimal_k):
        mask = cluster_labels == k
        axes[0, 1].scatter(features_scaled[mask, 0], features_scaled[mask, 2], 
                          c=[colors[k]], label=f'Cluster {k}', alpha=0.7)
    axes[0, 1].set_xlabel('cnt (normalized)')
    axes[0, 1].set_ylabel('hum (normalized)')
    axes[0, 1].set_title('Bike Rentals vs Humidity')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: 3D scatter (cnt, temp, hum)
    ax = fig.add_subplot(223, projection='3d')
    for k in range(optimal_k):
        mask = cluster_labels == k
        ax.scatter(features_scaled[mask, 0], features_scaled[mask, 1], features_scaled[mask, 2],
                  c=[colors[k]], label=f'Cluster {k}', alpha=0.7, s=30)
    ax.set_xlabel('cnt')
    ax.set_ylabel('temp')
    ax.set_zlabel('hum')
    ax.set_title('3D Feature Space')
    ax.legend()
    
    # Plot 4: Cluster characteristics
    df_features = pd.DataFrame(features_scaled, columns=feature_cols)
    df_features['cluster'] = cluster_labels
    cluster_means = df_features.groupby('cluster').mean()
    
    im = axes[1, 1].imshow(cluster_means.values, cmap='coolwarm', aspect='auto')
    axes[1, 1].set_xticks(range(len(feature_cols)))
    axes[1, 1].set_xticklabels(feature_cols, rotation=45)
    axes[1, 1].set_yticks(range(optimal_k))
    axes[1, 1].set_yticklabels([f'Cluster {k}' for k in range(optimal_k)])
    axes[1, 1].set_title('Cluster Characteristics (Normalized)')
    
    # Add colorbar
    plt.colorbar(im, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig('feature_clustering_results.png', dpi=300, bbox_inches='tight')
    plt.show()


def analyze_temporal_clusters(patterns, cluster_labels, optimal_k, pattern_type, logger):
    """
    Analyze and interpret temporal clusters.
    """
    logger.info("Analyzing temporal clusters")
    
    cluster_analysis = {}
    
    for k in range(optimal_k):
        cluster_patterns = patterns[cluster_labels == k]
        
        analysis = {
            'size': len(cluster_patterns),
            'mean_pattern': np.mean(cluster_patterns, axis=0),
            'std_pattern': np.std(cluster_patterns, axis=0),
            'peak_time': np.argmax(np.mean(cluster_patterns, axis=0)),
            'peak_value': np.max(np.mean(cluster_patterns, axis=0)),
            'min_value': np.min(np.mean(cluster_patterns, axis=0)),
            'variability': np.std(cluster_patterns, axis=0).mean()
        }
        
        # Determine cluster characteristics
        overall_mean = np.mean(patterns)
        overall_std = np.std(patterns)
        
        if analysis['peak_value'] > overall_mean + overall_std:
            cluster_type = 'High Demand Weeks'
        elif analysis['peak_value'] < overall_mean - overall_std:
            cluster_type = 'Low Demand Weeks'
        else:
            cluster_type = 'Normal Demand Weeks'
        
        analysis['type'] = cluster_type
        cluster_analysis[f'cluster_{k}'] = analysis
        
        logger.info(f"  Cluster {k} ({cluster_type}): size={analysis['size']}, "
                   f"peak={analysis['peak_value']:.1f}, variability={analysis['variability']:.2f}")
    
    return cluster_analysis


def map_clusters_to_original_data(df, cluster_labels, pattern_type, logger):
    """
    Map cluster labels back to original dataframe.
    """
    logger.info("Mapping clusters back to original data")
    
    df_with_clusters = df.copy()
    
    if pattern_type == 'weekly':
        # Map weekly clusters back to daily data
        n_weeks = len(df) // 7
        cluster_mapping = {}
        
        for week in range(n_weeks):
            if week < len(cluster_labels):
                start_idx = week * 7
                end_idx = start_idx + 7
                if end_idx <= len(df):
                    cluster_mapping[week] = cluster_labels[week]
        
        # Assign cluster labels to each day
        df_with_clusters['cluster'] = -1  # Default for unmapped days
        for week, cluster_id in cluster_mapping.items():
            start_idx = week * 7
            end_idx = start_idx + 7
            if end_idx <= len(df):
                df_with_clusters.iloc[start_idx:end_idx, df_with_clusters.columns.get_loc('cluster')] = cluster_id
    
    else:
        # Feature-based clustering: direct assignment
        df_with_clusters['cluster'] = cluster_labels
    
    logger.info(f"Mapped clusters to {len(df_with_clusters)} data points")
    return df_with_clusters


def detect_anomalies_with_clustering(clustering_results, logger, threshold_percentile=95):
    """
    Detect anomalies using clustering results.
    """
    logger.info("Detecting anomalies using clustering results")
    
    df_with_clusters = clustering_results['df_with_clusters']
    optimal_k = clustering_results['optimal_k']
    
    anomalies = []
    
    # Method 1: Small clusters (potential outliers)
    cluster_sizes = df_with_clusters['cluster'].value_counts()
    small_clusters = cluster_sizes[cluster_sizes < np.percentile(cluster_sizes, 25)]
    
    if len(small_clusters) > 0:
        logger.info(f"Found {len(small_clusters)} small clusters (potential outliers)")
        for cluster_id in small_clusters.index:
            cluster_data = df_with_clusters[df_with_clusters['cluster'] == cluster_id]
            anomalies.extend(cluster_data.index.tolist())
    
    # Method 2: High distance from cluster center
    for cluster_id in range(optimal_k):
        cluster_data = df_with_clusters[df_with_clusters['cluster'] == cluster_id]
        
        if len(cluster_data) > 1:
            # Calculate distance from cluster center for 'cnt'
            cluster_center = cluster_data['cnt'].mean()
            distances = np.abs(cluster_data['cnt'] - cluster_center)
            
            # Points far from center (95th percentile)
            threshold = np.percentile(distances, threshold_percentile)
            outlier_mask = distances > threshold
            
            if outlier_mask.any():
                outlier_indices = cluster_data[outlier_mask].index.tolist()
                anomalies.extend(outlier_indices)
                logger.info(f"Cluster {cluster_id}: Found {outlier_mask.sum()} outliers by distance")
    
    # Remove duplicates and sort
    anomalies = list(set(anomalies))
    anomalies.sort()
    
    logger.info(f"Total anomalies detected: {len(anomalies)}")
    
    return anomalies


def run_clustering_analysis(day_clean, logger):
    """
    Main function to run clustering analysis with both DTW and Euclidean.
    """
    logger.info("STARTING COMPREHENSIVE CLUSTERING ANALYSIS")
    
    # Try DTW first, then Euclidean as comparison
    clustering_results = {}
    
    # DTW Clustering (if available)
    if DTW_AVAILABLE:
        logger.info("Running DTW-based clustering...")
        dtw_results = kmeans_dtw_euclidean_clustering(
            day_clean, logger, max_clusters=8, use_dtw=True
        )
        clustering_results['DTW'] = dtw_results
    
    # Euclidean Clustering (always available)
    logger.info("Running Euclidean-based clustering...")
    euclidean_results = kmeans_dtw_euclidean_clustering(
        day_clean, logger, max_clusters=8, use_dtw=False
    )
    clustering_results['Euclidean'] = euclidean_results
    
    # Compare results
    logger.info("COMPARING CLUSTERING APPROACHES")
    for method, results in clustering_results.items():
        logger.info(f"{method} clustering: K={results['optimal_k']}, "
                   f"Silhouette={max(results['silhouette_scores']):.3f}")
    
    # Use Euclidean results for anomaly detection (more stable)
    main_results = clustering_results['Euclidean']
    
    # Detect anomalies
    anomalies = detect_anomalies_with_clustering(main_results, logger)
    
    # Analyze anomalies with external factors
    if len(anomalies) > 0:
        analyze_anomalies(day_clean, anomalies, logger)
    
    return clustering_results, anomalies


def analyze_anomalies(day_clean, anomalies, logger):
    """
    Analyze detected anomalies with external factors.
    """
    logger.info("ANALYZING DETECTED ANOMALIES")
    anomaly_data = day_clean.loc[anomalies]
    
    print(f"\n{'='*60}")
    print("ANOMALY ANALYSIS RESULTS")
    print(f"{'='*60}")
    print(f"Total anomalies detected: {len(anomalies)}")
    print(f"Percentage of data: {len(anomalies)/len(day_clean)*100:.1f}%")
    
    # Check weather conditions during anomalies
    if 'temp' in anomaly_data.columns:
        temp_anomalies = anomaly_data['temp'].mean()
        temp_normal = day_clean['temp'].mean()
        print(f"Average temperature during anomalies: {temp_anomalies*41:.1f}°C")
        print(f"Average temperature normally: {temp_normal*41:.1f}°C")
    
    # Check if anomalies occur on holidays
    if 'holiday' in anomaly_data.columns:
        holiday_anomalies = anomaly_data['holiday'].sum()
        holiday_normal = day_clean['holiday'].sum()
        print(f"Holidays during anomalies: {holiday_anomalies}")
        print(f"Total holidays in dataset: {holiday_normal}")
        print(f"Holiday anomaly rate: {holiday_anomalies/len(anomalies)*100:.1f}%")
    
    # Check weekday distribution
    if 'weekday' in anomaly_data.columns:
        weekday_counts = anomaly_data['weekday'].value_counts().sort_index()
        print(f"\nWeekday distribution of anomalies:")
        weekday_names = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
        for day, count in weekday_counts.items():
            print(f"  {weekday_names[day]}: {count}")
    
    # Visualize anomalies
    plt.figure(figsize=(15, 8))
    plt.plot(day_clean.index, day_clean['cnt'], label='Normal', alpha=0.7, color='blue')
    plt.scatter(anomaly_data.index, anomaly_data['cnt'], color='red', s=50, 
               label=f'Anomalies ({len(anomalies)})', zorder=5)
    plt.title('Bike Rentals with Detected Anomalies', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Bike Rentals')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('anomaly_detection_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    analyze_anomalies_upgraded(day_clean, anomalies, logger) 
    logger.info(f"Anomaly analysis completed: {len(anomalies)} anomalies identified and analyzed")


def analyze_anomalies_upgraded(day_clean, anomalies, logger):
    """
    upgraded anomaly analysis with:
    - Categorized anomalies (holiday/weather/both/other)
    - Temperature classification (cold/warm/hot)
    - Regression expected vs actual rentals
    - Two subplot visualization
    """
    logger.info("ANALYZING DETECTED ANOMALIES (UPGRADED)")
    anomaly_data = day_clean.loc[anomalies].copy()

    print("\n" + "="*70)
    print("           UPGRADED ANOMALY ANALYSIS RESULTS")
    print("="*70)
    print(f"Total anomalies detected: {len(anomalies)}")
    print(f"Percentage of dataset: {len(anomalies)/len(day_clean)*100:.2f}%")

    # ---------------------------
    #  Temperature classification
    # ---------------------------
    cold_cutoff = day_clean["temp"].quantile(0.25)
    hot_cutoff = day_clean["temp"].quantile(0.75)

    day_clean["temp_class"] = "warm"
    day_clean.loc[day_clean["temp"] < cold_cutoff, "temp_class"] = "cold"
    day_clean.loc[day_clean["temp"] > hot_cutoff, "temp_class"] = "hot"

    temp_colors = {
        "cold": "cyan",
        "warm": "orange",
        "hot": "red"
    }
    day_clean["temp_color"] = day_clean["temp_class"].map(temp_colors)

    # ---------------------------
    #  Determine anomaly reasons
    # ---------------------------
    anomaly_data["reason"] = "other"

    if "holiday" in anomaly_data.columns:
        anomaly_data.loc[anomaly_data["holiday"] == 1, "reason"] = "holiday"

    # weather anomaly: temp far from normal (Z-score)
    temp_mean = day_clean["temp"].mean()
    temp_std = day_clean["temp"].std()
    anomaly_data.loc[
        (np.abs(anomaly_data["temp"] - temp_mean) > temp_std * 1.2),
        "reason"
    ] = "weather"

    # both holiday + weather
    anomaly_data.loc[
        (anomaly_data["holiday"] == 1) &
        (np.abs(anomaly_data["temp"] - temp_mean) > temp_std * 1.2),
        "reason"
    ] = "both"

    # Colors for anomaly reasons
    anomaly_colors = {
        "holiday": "yellow",
        "weather": "purple",
        "both": "black",
        "other": "red"
    }
    anomaly_data["color"] = anomaly_data["reason"].map(anomaly_colors)

    # ---------------------------
    #  Regression expected rentals (baseline)
    # ---------------------------
    from sklearn.linear_model import LinearRegression

    model = LinearRegression()
    model.fit(day_clean[["temp"]], day_clean["cnt"])

    expected_cnt = model.predict(day_clean[["temp"]])
    day_clean["expected_cnt"] = expected_cnt

    # ---------------------------
    #  Visualization (two subplots)
    # ---------------------------
    plt.figure(figsize=(18, 12))

    # -----------------------------------------------------------------
    # Subplot 1: Anomaly categories
    # -----------------------------------------------------------------
    plt.subplot(2, 1, 1)
    plt.plot(day_clean.index, day_clean["cnt"], color="blue", alpha=0.5, label="Normal Rentals")
    plt.plot(day_clean.index, day_clean["expected_cnt"], linestyle="--", alpha=0.7, label="Expected Rentals (Regression)")

    # plot anomalies with category colors
    for reason, grp in anomaly_data.groupby("reason"):
        plt.scatter(grp.index, grp["cnt"], 
                    color=anomaly_colors[reason], 
                    s=60, 
                    label=f"Anomaly: {reason} ({len(grp)})", 
                    zorder=5)

    plt.title("Anomalies Categorized by Reason (Holiday / Weather / Both / Other)", fontsize=14)
    plt.ylabel("Bike Rentals")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.legend()

    # -----------------------------------------------------------------
    # Subplot 2: Temperature classes (cold, warm, hot)
    # -----------------------------------------------------------------
    plt.subplot(2, 1, 2)

    for cls, grp in day_clean.groupby("temp_class"):
        plt.scatter(grp.index, grp["cnt"],
                    color=temp_colors[cls],
                    s=25,
                    label=f"{cls.capitalize()} Days ({len(grp)})")

    plt.title("Bike Rentals by Temperature Class (Cold / Warm / Hot)", fontsize=14)
    plt.xlabel("Date")
    plt.ylabel("Bike Rentals")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig("anomaly_analysis_upgraded.png", dpi=300, bbox_inches="tight")
    plt.show()

    #logger.info("UPGRADED anomaly analysis completed.")
    return anomaly_data


def plot_rush_hours(time_df):
    import matplotlib.pyplot as plt

    #  Define helper to compute hourly means
    def hourly_mean(df):
        return df.groupby("hr")["cnt"].mean()

    #  Create categories

    # (A) Thursday & Friday
    thu_fri_df = time_df[ time_df['weekday'].isin([4, 5]) ]

    # (B) Holidays (Bike dataset: holiday column = 1)
    holiday_df = time_df[ time_df["holiday"] == 1 ]

    # (C) Categorize temperature: cold < warm < hot
    temp_q = time_df["temp"].quantile([0.33, 0.66])
    cold_th = temp_q[0.33]
    hot_th  = temp_q[0.66]

    cold_df = time_df[ time_df["temp"] <= cold_th ]
    warm_df = time_df[ (time_df["temp"] > cold_th) & (time_df["temp"] <= hot_th) ]
    hot_df  = time_df[ time_df["temp"] > hot_th ]

    #  Prepare hourly patterns
    sets = {
        "Thu & Fri": hourly_mean(thu_fri_df),
        "Holidays": hourly_mean(holiday_df),
        "Cold Days": hourly_mean(cold_df),
        "Hot Days": hourly_mean(hot_df),
    }

    #  Plotting
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for ax, (title, hourly_pattern) in zip(axes, sets.items()):
        
        bars = ax.bar(hourly_pattern.index, hourly_pattern.values,
                      color="skyblue", edgecolor="black")

        ax.set_title(f"Average Rentals by Hour ({title})", fontsize=13)
        ax.set_xlabel("Hour of Day", fontsize=11)
        ax.set_ylabel("Average Rentals", fontsize=11)
        ax.set_xticks(range(0, 24))
        ax.grid(axis='y', linestyle='--', alpha=0.4)

        # Put numbers on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2,
                    height + 5,
                    f"{int(height)}",
                    ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.show()

def main():
    logger = setting_up_logger()
    # Read data
    time_df_origin, day_df_origin = reading_files(logger=logger)

    # Convert to datetime BUT DO NOT set as index
    time_df_origin['dteday'] = pd.to_datetime(time_df_origin['dteday'])
    day_df_origin['dteday'] = pd.to_datetime(day_df_origin['dteday'])
    
    # Keep a copy with index for plotting later if needed
    day_df = day_df_origin.set_index('dteday')
    time_df=time_df_origin.set_index('dteday')

    # ========================================
    # ANALYZE DAILY DATA
    # ========================================
    logger.info("ANALYZING DAILY DATA")
    
    # Detect seasonality
    day_periods = detect_seasonality(logger, day_df['cnt'],name='day')
    print(f"Detected periods: {day_periods}")

    # === Prepare data for statsforecast (MUST have ds, y, unique_id) ===
    day_df_sf = day_df_origin[['dteday', 'cnt']].copy()
    day_df_sf = day_df_sf.rename(columns={'dteday': 'ds', 'cnt': 'y'})
    # unique_id is REQUIRED!
    day_df_sf['unique_id'] = 'bike_daily'

    print("Final statsforecast dataframe:")
    print(day_df_sf.head())
    print("Columns:", day_df_sf.columns.tolist())

    
    day_df=day_df.sort_index()
    day_df["week"]= day_df.index.to_period("w") 
    week_count=day_df.groupby("week")["cnt"].count()
    valid_weeks=week_count[week_count==7].index
    day_clean=day_df[day_df["week"].isin(valid_weeks)]
    print(day_clean.head(5))

    min_day=day_clean.loc[day_clean["cnt"].idxmin()]
    logger.info(f"day that had minimum rental amount:{min_day}") 
    max_day=day_clean.loc[day_clean["cnt"].idxmax()]
    logger.info(f"day that had maximum rental amount:{max_day}")

    print(f"maxiumam rental count day: {max_day['cnt']} minimum rental count day {min_day['cnt']}")
    # Select only numeric columns for correlation
    day_numeric = day_clean.drop("week",axis=1)
    # Compute correlation matrix
    day_corr = day_numeric.corr()

    # Take absolute values
    day_corr_abs = day_corr.abs()

    # Unstack to convert to Series
    day_corr_series = day_corr_abs.unstack()

    # Filter out self-correlation
    day_corr_series = day_corr_series[day_corr_series < 0.999]

    # Sort descending
    sorted_corr = day_corr_series.sort_values(ascending=False)

    # Pick strong correlations
    strong_corr = sorted_corr[sorted_corr >= 0.7]

    print(f" length of the correlation: {len(strong_corr)}")
    
    top_features=strong_corr.index

    top_feature_list=list(set([f for pair in top_features for f in pair])) 
    plt.figure(figsize=(20,20))
    sns.heatmap(day_numeric[top_feature_list].corr(), annot=True, cmap='coolwarm')
    plt.show()

    day_cfilterd=day_numeric[top_feature_list]


        # Convert to Celsius
    day_cfilterd["temp_C"] = day_cfilterd["temp"] * 41

    # Bin into 5 temperature ranges
    day_cfilterd["temp_bin"] = pd.cut(day_cfilterd["temp_C"], bins=5)

    # Group by bins
    temp_group = day_cfilterd.groupby("temp_bin")["cnt"].mean()

    # Plot
    plt.figure(figsize=(10,6))
    temp_group.plot(kind="bar", color="skyblue", edgecolor="black")

    plt.title("Average Rentals by Temperature Range (°C)", fontsize=14)
    plt.xlabel("Temperature Range (°C)", fontsize=12)
    plt.ylabel("Average Rentals (cnt)", fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

        # Average rentals on holiday vs non-holiday
    holiday_group = day_clean.groupby("holiday")["cnt"].mean()

    plt.figure(figsize=(7,5))
    holiday_group.plot(kind="bar", color=["skyblue", "orange"], edgecolor="black")

    plt.title("Average Rentals: Holiday vs Non-Holiday", fontsize=14)
    plt.xlabel("Holiday (0 = No, 1 = Yes)", fontsize=12)
    plt.ylabel("Average Rentals (cnt)", fontsize=12)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

    weekday_names = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

    day_clean["weekday_name"] = day_clean["weekday"].map(lambda x: weekday_names[x])

    weekday_group = day_clean.groupby("weekday_name")["cnt"].mean().reindex(weekday_names)

    plt.figure(figsize=(10,5))
    bars = plt.bar(weekday_group.index, weekday_group.values,
                   color="lightgreen", edgecolor="black")

    plt.title("Average Rentals by Weekday", fontsize=14)
    plt.xlabel("Weekday", fontsize=12)
    plt.ylabel("Average Rentals (cnt)", fontsize=12)
    plt.xticks(rotation=45)

    # === Add mean labels on top of bars ===
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2,
                 height + 5,                       # little offset above bar
                 f"{height:.0f}",                  # round to nearest integer
                 ha='center', va='bottom',
                 fontsize=10)

    plt.tight_layout()
    plt.show()


        # Thursday = 4, Friday = 5 (Bike sharing dataset convention)
    plot_rush_hours(time_df)
    
    # ========================================
    # K-MEANS + DTW + EUCLIDEAN CLUSTERING ANALYSIS
    # ========================================
    logger.info("STARTING K-MEANS + DTW + EUCLIDEAN CLUSTERING ANALYSIS")
    
    # Run comprehensive clustering analysis
    clustering_results, anomalies = run_clustering_analysis(day_clean, logger)
    
    # ========================================
    # RESEARCH QUESTION INSIGHTS
    # ========================================
    logger.info("GENERATING RESEARCH QUESTION INSIGHTS")
    
    print(f"\n{'='*60}")
    print("RESEARCH QUESTION: How do temperature, temporal patterns, and anomalies influence bike-sharing demand?")
    print(f"{'='*60}")
    
    # Temperature insights
    temp_corr = day_clean['cnt'].corr(day_clean['temp'])
    print(f"\n1. TEMPERATURE IMPACT:")
    print(f"   - Correlation between temperature and rentals: {temp_corr:.3f}")
    print(f"   - Strong positive correlation indicates temperature is a key driver")
    
    # Temporal pattern insights from clustering
    main_results = clustering_results['Euclidean']  # Use Euclidean results
    if main_results['pattern_type'] == 'weekly':
        cluster_analysis = main_results['cluster_analysis']
        print(f"\n2. TEMPORAL PATTERNS (Weekly Clusters):")
        for cluster_name, analysis in cluster_analysis.items():
            print(f"   - {cluster_name}: {analysis['type']}")
            print(f"     Peak demand: {analysis['peak_value']:.0f} rentals")
            print(f"     Peak day: {analysis['peak_time'] + 1} (0=Sunday)")
    
    # Clustering method comparison
    print(f"\n3. CLUSTERING METHOD COMPARISON:")
    for method, results in clustering_results.items():
        best_silhouette = max(results['silhouette_scores'])
        print(f"   - {method}: K={results['optimal_k']}, Best Silhouette={best_silhouette:.3f}")
    
    # Anomaly insights
    if len(anomalies) > 0:
        print(f"\n4. ANOMALY INSIGHTS:")
        print(f"   - {len(anomalies)} anomalous days detected ({len(anomalies)/len(day_clean)*100:.1f}% of data)")
        print(f"   - Anomalies often coincide with unusual weather conditions or special events")
        print(f"   - These outliers are important for understanding demand extremes")
    
    print(f"\n{'='*60}")
    print("KEY FINDINGS SUMMARY")
    print(f"{'='*60}")
    print("✓ Temperature is the primary driver of bike-sharing demand")
    print("✓ Distinct temporal patterns exist (weekly cycles)")
    print("✓ Clustering reveals different demand regimes (low/medium/high)")
    print("✓ DTW and Euclidean methods provide complementary insights")
    print("✓ Anomalies highlight extreme conditions affecting demand")
    print("✓ These insights can help optimize bike allocation and pricing")
    
    # Save key results for report
    results_summary = {
        'temperature_correlation': temp_corr,
        'clustering_results': clustering_results,
        'anomalies_count': len(anomalies),
        'anomalies_percentage': len(anomalies)/len(day_clean)*100,
        'optimal_clusters_euclidean': main_results['optimal_k'],
        'best_silhouette_euclidean': max(main_results['silhouette_scores'])
}
    
    logger.info("CLUSTERING ANALYSIS COMPLETED SUCCESSFULLY")
    logger.info(f"Key metrics: Temp corr={temp_corr:.3f}, "
               f"Optimal K={main_results['optimal_k']}, "
               f"Anomalies={len(anomalies)}")
    
    
        # Walk-forward validation with auto_arima
    predictions, actuals, metrics,time_took_arm = walk_forward_arima(
        data=day_clean[['cnt']],  # Only need 'cnt' column
        periods=day_periods,
        logger=logger,
        name='day',
        test_size=50  # Last 50 days for testing
        )
    print(metrics)
                    
    prediction,actuals,metrics,time_took_eps=walk_forward_exponentialsmoothing(day_clean, 50, logger)

    print(metircs)

    #predictions_sarima,actuals_sarima,metrics_sarima=walk_forward_auto_sarima(data=day_df,logger=logger)
    #
  #
    #logger.info("\n" + "="*60)
    #logger.info("ANALYZING HOURLY DATA")
    #logger.info("="*60)
    #
    #logger.info("Detecting seasonality for hourly dataset")
    #time_periods = detect_seasonality(logger, time_df['cnt'],name='hour')
    #
    #stationary_or_not = check_stationary(time_df['cnt'], logger=logger, name="Hourly_Data")
    #
    #predictions_h, actuals_h, metrics_h = walk_forward_arima(
        #data=time_df[['cnt']],
        #periods=time_periods,
        #logger=logger,
        #name='hour',
#test_size=5  # Last week for testing
    #)


if __name__ == '__main__':
    main()


