import os
import pandas as pd
import numpy as np
import re
import logging
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
import json
from tqdm import tqdm

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CleanZoneFinder:
    def __init__(self, time_column='time', label_column='arima_residual_anomaly', min_clean_length=10):  
        self.time_column = time_column
        self.label_column = label_column
        self.min_clean_length = min_clean_length

    def find_clean_zones(self, df):
        clean_intervals = []
        clean_start = None
        clean_length = 0

        for i in range(len(df)):
            if df[self.label_column].iloc[i] == 0:  # 이상치가 아닌 구간
                if clean_start is None:
                    clean_start = df[self.time_column].iloc[i]
                clean_length += 1
            else:
                if clean_start is not None and clean_length >= self.min_clean_length:
                    clean_end = df[self.time_column].iloc[i - 1]
                    clean_intervals.append([str(clean_start), str(clean_end)])
                clean_start = None
                clean_length = 0

        if clean_start is not None and clean_length >= self.min_clean_length:
            clean_end = df[self.time_column].iloc[-1]
            clean_intervals.append([str(clean_start), str(clean_end)])

        return clean_intervals

class ARIMAOutlierDetector:
    def __init__(self, order=(5, 1, 0)):
        self.order = order  

    def fit_predict(self, df, feature_column):
        # ARIMA 모델 피팅
        model = ARIMA(df[feature_column], order=self.order)
        fit = model.fit()

        # 잔차(residual) 계산
        df['arima_residual'] = fit.resid

        # 잔차의 이상치 탐지 (평균에서 3 표준편차를 벗어나는 데이터)
        threshold = 3 * np.std(df['arima_residual'])
        df['arima_residual_anomaly'] = np.where(np.abs(df['arima_residual']) > threshold, 1, 0)

        return df['arima_residual_anomaly']

class OutlierProcessor:
    def __init__(self, base_dir, isolation_dir, arima_order):
        self.base_dir = base_dir
        self.isolation_dir = isolation_dir
        self.arima_order = arima_order
        self.date_pattern = re.compile(r'\d{6}')
        self.clean_zone_finder = CleanZoneFinder()

        if not os.path.exists(isolation_dir):
            os.makedirs(isolation_dir)

    def process_data(self, folder_list, months):
        overall_clean_zones = {}
        for folder in folder_list:
            folder_path = os.path.join(self.base_dir, folder)

            csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
            folder_clean_zones = {}

            for file_name in tqdm(csv_files, desc=f"Processing folder {folder}", unit='file'):
                match = self.date_pattern.search(file_name)
                if match:
                    file_date = match.group(0)

                    if file_date in months:
                        file_path = os.path.join(folder_path, file_name)
                        logging.info(f"Processing: {file_path}")

                        try:
                            # 데이터 읽기
                            df = pd.read_csv(file_path)

                            # 데이터 전처리
                            df['time'] = pd.to_datetime(df['time'], unit='ms')
                            df['value'] = df['value'].interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')

                            decomposition = seasonal_decompose(df['value'], model='additive', period=30)
                            df['residual'] = decomposition.resid.bfill().ffill()

                            # ARIMA 모델을 사용한 이상치 탐지
                            arima_detector = ARIMAOutlierDetector(order=self.arima_order)
                            df['arima_residual_anomaly'] = arima_detector.fit_predict(df, 'value')

                            # 청정구역 찾기
                            clean_intervals = self.clean_zone_finder.find_clean_zones(df)
                            df['clean_zone'] = 0
                            for interval in clean_intervals:
                                mask = (df['time'] >= interval[0]) & (df['time'] <= interval[1])
                                df.loc[mask, 'clean_zone'] = 1

                            # 정규화
                            scaler = StandardScaler()
                            df['value_scaled'] = scaler.fit_transform(df[['value']])

                            # 시각화 및 결과 저장
                            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12))

                            # Original Value 시각화
                            ax1.plot(df['time'], df['value'], label='Original Value', color='gray', alpha=0.5)
                            outliers_value = df[df['arima_residual_anomaly'] == 1]
                            ax1.scatter(outliers_value['time'], outliers_value['value'], color='red', label='Outliers', zorder=3)
                            ax1.fill_between(df['time'], df['value'], where=(df['clean_zone'] == 1), color='blue', alpha=0.2, label='Clean Zone')
                            ax1.set_title(f"Original Data with ARIMA Outliers (order={self.arima_order})")
                            ax1.set_xlabel("Time")
                            ax1.set_ylabel("Value")

                            # Scaled Value 시각화
                            ax2.plot(df['time'], df['value_scaled'], label='Scaled Value', color='orange', alpha=0.5)
                            outliers_scaled = df[df['arima_residual_anomaly'] == 1]
                            ax2.scatter(outliers_scaled['time'], outliers_scaled['value_scaled'], color='red', label='Outliers', zorder=3)
                            ax2.set_title(f"Scaled Data with ARIMA Outliers (order={self.arima_order})")
                            ax2.set_xlabel("Time")
                            ax2.set_ylabel("Scaled Value")

                            # Residual 시각화
                            ax3.plot(df['time'], df['residual'], label='Residual', color='blue', alpha=0.5)
                            outliers_res = df[df['arima_residual_anomaly'] == 1]
                            ax3.scatter(outliers_res['time'], outliers_res['residual'], color='red', label='Outliers', zorder=3)
                            ax3.set_title(f"Residual with ARIMA Outliers (order={self.arima_order})")
                            ax3.set_xlabel("Time")
                            ax3.set_ylabel("Residual")

                            # 시각화 파일 저장
                            plot_filename = f"{folder}_{file_date}_arima.png"
                            contamination_dir = os.path.join(self.isolation_dir, folder)
                            if not os.path.exists(contamination_dir):
                                os.makedirs(contamination_dir)
                            plt.tight_layout()
                            plt.savefig(os.path.join(contamination_dir, plot_filename))
                            plt.close()

                            # CSV 파일 저장
                            csv_output_filename = f"{folder}_{file_date}_arima.csv"
                            df.to_csv(os.path.join(contamination_dir, csv_output_filename), index=False)

                            # 청정구역 정보를 JSON 파일로 저장
                            json_output_filename = os.path.join(contamination_dir, f"{file_name}.json")
                            folder_clean_zones[file_name] = clean_intervals  # 폴더별 청정구역 저장

                            with open(json_output_filename, 'w') as json_file:
                                json.dump({folder: {file_name: clean_intervals}}, json_file, indent=4)

                        except Exception as e:
                            logging.error(f"Error processing {file_path}: {e}")
                            continue

            overall_clean_zones[folder] = folder_clean_zones
        logging.info("Processing complete and files saved.")
