import os
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pmdarima import auto_arima
import yaml
from tqdm import tqdm

# Clean Zone Finder Class
class CleanZoneFinder:
    def __init__(self, time_column='time', min_clean_length=20):
        self.time_column = time_column
        self.min_clean_length = min_clean_length

    def find_clean_zones(self, df, label_column):
        clean_intervals = []
        clean_start = None
        clean_length = 0

        for i in range(len(df)):
            if df[label_column].iloc[i] == 0:
                if clean_start is None:
                    clean_start = df.index[i]
                clean_length += 1
            else:
                if clean_start is not None and clean_length >= self.min_clean_length:
                    clean_end = df.index[i - 1]
                    clean_intervals.append([clean_start, clean_end])
                clean_start = None
                clean_length = 0

        if clean_start is not None and clean_length >= self.min_clean_length:
            clean_end = df.index[-1]
            clean_intervals.append([clean_start, clean_end])

        return clean_intervals

# ARIMA Outlier Detector
class ARIMAOutlierDetector:
    def __init__(self, order=(5, 1, 0)):
        self.order = order

    def fit_predict(self, df, feature_column):
        model = ARIMA(df[feature_column], order=self.order)
        fit = model.fit()
        df['arima_residual'] = fit.resid
        threshold = 3 * np.std(df['arima_residual'])
        df['arima_residual_anomaly'] = np.where(np.abs(df['arima_residual']) > threshold, 1, 0)
        return df['arima_residual_anomaly']

# ARIMA Parameter Finder with Timeout
def find_best_arima_params(data):
    try:
        with ThreadPoolExecutor() as executor:
            future = executor.submit(
                auto_arima, 
                data.sample(frac=0.03, random_state=42),
                seasonal=False,
                start_p=1,
                start_q=0,
                max_p=5,
                max_q=5,
                d=2,
                trace=True,
                error_action='ignore',
                suppress_warnings=True,
                stepwise=True
            )
            return future.result(timeout=120).order
    except TimeoutError:
        logging.warning("Time limit exceeded, using default parameters (1, 1, 1)")
        return (1, 1, 1)

# Outlier Processor Class
class OutlierProcessor:
    def __init__(self, base_dir, isolation_dir):
        self.base_dir = base_dir
        self.isolation_dir = isolation_dir
        self.clean_zone_finder = CleanZoneFinder()

        if not os.path.exists(isolation_dir):
            os.makedirs(isolation_dir)

    def process_data(self, config_path):
        csv_files = [f for f in os.listdir(self.base_dir) if f.endswith('.csv')]
        arima_params = {}

        for file_name in tqdm(csv_files, desc="Processing CSV files", unit='file'):
            file_path = os.path.join(self.base_dir, file_name)
            logging.info(f"Processing: {file_path}")

            try:
                df = pd.read_csv(file_path)
                df['time'] = pd.to_datetime(df['time'], unit='ms')
                df.set_index('time', inplace=True)
                data = df['value']

                # Finding ARIMA Parameters
                p, d, q = find_best_arima_params(data)
                arima_params[file_name] = {'p': p, 'd': d, 'q': q}
                logging.info(f"Best ARIMA params for {file_name}: (p={p}, d={d}, q={q})")

                # Detecting Anomalies with ARIMA
                arima_detector = ARIMAOutlierDetector(order=(p, d, q))
                df['arima_residual_anomaly'] = arima_detector.fit_predict(df, 'value')

                # Clean Zone Detection
                clean_intervals = self.clean_zone_finder.find_clean_zones(df, 'arima_residual_anomaly')
                df['clean_zone'] = 0
                for interval in clean_intervals:
                    mask = (df.index >= interval[0]) & (df.index <= interval[1])
                    df.loc[mask, 'clean_zone'] = 1

                # Saving the processed CSV
                # Saving the processed CSV with only 'time' and 'anomaly' columns
                output_file = os.path.join(self.isolation_dir, f"{file_name}_arima_label.csv")
                df_reset = df.reset_index()[['time', 'clean_zone']].rename(columns={'clean_zone': 'anomaly'})
                df_reset.to_csv(output_file, index=False)
                logging.info(f"Saved labeled data to {output_file}")

                # Monthly Plotting in a 3x5 Grid
                df['month'] = df.index.to_period("M")
                unique_months = df['month'].unique()[:15]  # Up to 15 months for 3x5 layout

                fig, axs = plt.subplots(3, 5, figsize=(20, 12), sharex=False)
                fig.suptitle(f"{file_name} - Monthly ARIMA Outliers and Clean Zones", fontsize=16)

                for i, month in enumerate(unique_months):
                    ax = axs[i // 5, i % 5]
                    monthly_data = df[df['month'] == month]

                    # Plot each month's data
                    ax.plot(monthly_data.index, monthly_data['value'], label='Value')
                    ax.scatter(monthly_data[monthly_data['arima_residual_anomaly'] == 1].index,
                               monthly_data[monthly_data['arima_residual_anomaly'] == 1]['value'],
                               color='red', label='Outliers')
                    ax.fill_between(monthly_data.index, monthly_data['value'], 
                                    where=(monthly_data['clean_zone'] == 1), color='blue', alpha=0.3, label='Clean Zone')
                    ax.set_title(f"Month: {month.strftime('%Y-%m')}", fontsize=10)
                    ax.legend(fontsize=8)

                    # Set x-axis limits to cover only the current month's data
                    ax.set_xlim(monthly_data.index.min(), monthly_data.index.max())
                    ax.tick_params(axis='x', rotation=45)  # Rotate x-ticks for readability

                plt.tight_layout(rect=[0, 0, 1, 0.96])
                plt.savefig(os.path.join(self.isolation_dir, f"{file_name}_monthly_arima_outliers.png"))
                plt.close()

            except Exception as e:
                logging.error(f"Error processing {file_path}: {e}")
                continue

        with open(config_path, 'w') as config_file:
            yaml.dump(arima_params, config_file, default_flow_style=False)
        logging.info(f"Saved ARIMA parameters to {config_path}")

# 실행 예시
if __name__ == "__main__":
    base_dir = "C:/Users/ansck/Documents/ybigta/DA_25/project/AD_detection/clean_detection/data"
    isolation_dir = "C:/Users/ansck/Documents/ybigta/DA_25/project/AD_detection/clean_detection/output"
    config_path = "C:/Users/ansck/Documents/ybigta/DA_25/project/AD_detection/clean_detection/utils/config.yaml"
    
    processor = OutlierProcessor(base_dir=base_dir, isolation_dir=isolation_dir)
    processor.process_data(config_path)
