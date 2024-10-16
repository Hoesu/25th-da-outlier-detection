import os
import pandas as pd
import numpy as np
import re
import logging
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
import json
from tqdm import tqdm
from merge import JSONMerger
class CleanZoneFinder:
    def __init__(self, time_column='time', label_column='iso_label_value_scaled', min_clean_length=10):
        """
        초기화 메서드. 청정 구역을 찾기 위한 파라미터를 설정합니다.
        
        :param time_column: 시간 컬럼 이름 (기본값 'time')
        :param label_column: 이상치 라벨 컬럼 이름 (기본값 'iso_label_value_scaled') -> 정규화된 값 사용
        :param min_clean_length: 최소 청정 구역의 길이 (기본값 10)
        """
        self.time_column = time_column
        self.label_column = label_column
        self.min_clean_length = min_clean_length

    def find_clean_zones(self, df):
        """
        청정 구역을 찾는 메서드.
        
        :param df: 데이터프레임 (시간 및 라벨이 포함된 데이터)
        :return: 청정 구역의 리스트 (시작 시간, 종료 시간)
        """
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

        # 마지막 구간 처리
        if clean_start is not None and clean_length >= self.min_clean_length:
            clean_end = df[self.time_column].iloc[-1]
            clean_intervals.append([str(clean_start), str(clean_end)])

        return clean_intervals

class OutlierDetector:
    def __init__(self, n_estimators=200, max_samples=0.8, contamination=0.01):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            random_state=42
        )

    def fit_predict(self, df, feature_columns):
        """
        주어진 feature_columns에 대해 이상치 탐지를 수행하고, 이상치 라벨을 반환합니다.
        
        :param df: 입력 데이터프레임
        :param feature_columns: 이상치 탐지에 사용할 컬럼들
        :return: 이상치 라벨 (1: 이상치, 0: 정상)
        """
        return np.where(self.model.fit_predict(df[feature_columns]) == -1, 1, 0)

class OutlierProcessor:
    def __init__(self, base_dir, isolation_dir, n_estimators, max_samples, contamination_list):
        self.base_dir = base_dir
        self.isolation_dir = isolation_dir
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination_list = contamination_list
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
                            df['residual'] = decomposition.resid.fillna(method='bfill').fillna(method='ffill')

                            scaler = StandardScaler()
                            df['value_scaled'] = scaler.fit_transform(df[['value']])

                            # contamination 값을 순회하면서 각각 처리
                            for contamination in self.contamination_list:
                                outlier_detector = OutlierDetector(n_estimators=self.n_estimators, max_samples=self.max_samples, contamination=contamination)

                                # 이상치 탐지
                                df['iso_label_value_scaled'] = outlier_detector.fit_predict(df, ['value_scaled'])
                                df['iso_label_value'] = outlier_detector.fit_predict(df, ['value'])
                                df['iso_label_residual'] = outlier_detector.fit_predict(df, ['residual'])

                                # 청정구역 찾기
                                clean_intervals = self.clean_zone_finder.find_clean_zones(df)
                                df['clean_zone'] = 0
                                for interval in clean_intervals:
                                    mask = (df['time'] >= interval[0]) & (df['time'] <= interval[1])
                                    df.loc[mask, 'clean_zone'] = 1

                                # 시각화 및 결과 저장
                                fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12))

                                # value, value_scaled, residual에 대해 각각 이상치와 청정구역 시각화
                                ax1.plot(df['time'], df['value'], label='Original Value', color='gray', alpha=0.5)
                                outliers_value = df[df['iso_label_value_scaled'] == 1]
                                ax1.scatter(outliers_value['time'], outliers_value['value'], color='red', label='Outliers', zorder=3)
                                ax1.fill_between(df['time'], df['value'], where=(df['clean_zone'] == 1), color='blue', alpha=0.2, label='Clean Zone')
                                ax1.set_title(f"Original Data with Outliers (contamination={contamination})")
                                ax1.set_xlabel("Time")
                                ax1.set_ylabel("Value")

                                ax2.plot(df['time'], df['value_scaled'], label='Scaled Value', color='orange', alpha=0.5)
                                outliers_scaled = df[df['iso_label_value_scaled'] == 1]
                                ax2.scatter(outliers_scaled['time'], outliers_scaled['value_scaled'], color='red', label='Outliers', zorder=3)
                                ax2.set_title(f"Scaled Data with Outliers (contamination={contamination})")
                                ax2.set_xlabel("Time")
                                ax2.set_ylabel("Scaled Value")

                                ax3.plot(df['time'], df['residual'], label='Residual', color='blue', alpha=0.5)
                                outliers_res = df[df['iso_label_residual'] == 1]
                                ax3.scatter(outliers_res['time'], outliers_res['residual'], color='red', label='Outliers', zorder=3)
                                ax3.set_title(f"Residual with Outliers (contamination={contamination})")
                                ax3.set_xlabel("Time")
                                ax3.set_ylabel("Residual")

                                # 시각화 파일 저장
                                plot_filename = f"{folder}_{file_date}_contamination_{contamination}.png"
                                contamination_dir = os.path.join(self.isolation_dir, f"contamination_{contamination}", folder)
                                if not os.path.exists(contamination_dir):
                                    os.makedirs(contamination_dir)
                                plt.tight_layout()
                                plt.savefig(os.path.join(contamination_dir, plot_filename))
                                plt.close()

                                # CSV 파일 저장
                                csv_output_filename = f"{folder}_{file_date}_contamination_{contamination}.csv"
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