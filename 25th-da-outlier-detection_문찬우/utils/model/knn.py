import os
import pandas as pd
import numpy as np
import re
import logging
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
import json
from tqdm import tqdm

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CleanZoneFinder:
    def __init__(self, time_column='time', label_column='knn_label', min_clean_length=10):
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

class KNNOutlierDetector:
    def __init__(self, n_neighbors_list=[30, 50, 70], window_size=30):  # 이동 윈도우 크기 추가
        self.n_neighbors_list = n_neighbors_list
        self.window_size = window_size

    def fit_predict(self, df, feature_columns):
        knn_labels = {}
        for n_neighbors in self.n_neighbors_list:
            if n_neighbors > len(df):  # n_neighbors가 샘플 수보다 크면 조정
                n_neighbors = len(df)
                logging.warning(f"n_neighbors is greater than the number of samples. Setting n_neighbors to {n_neighbors}")

            labels = np.zeros(len(df))  # 이상치 라벨 저장을 위한 배열

            for start in range(0, len(df) - self.window_size + 1):
                window_data = df.iloc[start:start + self.window_size]  # 이동 윈도우 데이터 선택
                knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
                knn_model.fit(window_data[feature_columns], np.zeros(len(window_data)))  # 정상 데이터(0)를 가정하여 훈련
                distances, indices = knn_model.kneighbors(window_data[feature_columns])

                # KNN 거리를 기반으로 임의로 이상치를 정의 (최대 거리 상위 5% 이상을 이상치로 정의)
                distance_threshold = np.percentile(distances[:, -1], 95)
                window_labels = np.where(distances[:, -1] > distance_threshold, 1, 0)  # 1: 이상치, 0: 정상
                labels[start:start + self.window_size] = window_labels  # 라벨 저장

            knn_labels[f'knn_{n_neighbors}'] = labels
        return knn_labels

class OutlierProcessor:
    def __init__(self, base_dir, isolation_dir, n_neighbors_list, window_size=30):
        self.base_dir = base_dir
        self.isolation_dir = isolation_dir
        self.n_neighbors_list = n_neighbors_list
        self.window_size = window_size
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

                            # 시계열 분해
                            decomposition = seasonal_decompose(df['value'], model='additive', period=30)
                            df['residual'] = decomposition.resid.bfill().ffill()

                            # 정규화
                            scaler = StandardScaler()
                            df['value_scaled'] = scaler.fit_transform(df[['value']])

                            # KNN 모델을 사용한 이상치 탐지
                            knn_detector = KNNOutlierDetector(n_neighbors_list=self.n_neighbors_list, window_size=self.window_size)
                            knn_labels = knn_detector.fit_predict(df, ['value_scaled'])

                            for key, labels in knn_labels.items():
                                df[key] = labels

                            # 청정구역 찾기
                            clean_intervals = self.clean_zone_finder.find_clean_zones(df)
                            df['clean_zone'] = 0
                            for interval in clean_intervals:
                                mask = (df['time'] >= interval[0]) & (df['time'] <= interval[1])
                                df.loc[mask, 'clean_zone'] = 1

                            # CSV 파일 저장
                            csv_output_filename = f"{folder}_{file_date}_knn.csv"
                            contamination_dir = os.path.join(self.isolation_dir, folder)
                            if not os.path.exists(contamination_dir):
                                os.makedirs(contamination_dir)
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

import os
import pandas as pd
import numpy as np
import re
import logging
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
import json
from tqdm import tqdm

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CleanZoneFinder:
    def __init__(self, time_column='time', label_column='knn_label', min_clean_length=10):
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

class KNNOutlierDetector:
    def __init__(self, n_neighbors_list=[30, 50, 70], window_size=30):  # 이동 윈도우 크기 추가
        self.n_neighbors_list = n_neighbors_list
        self.window_size = window_size

    def fit_predict(self, df, feature_columns):
        knn_labels = {}
        for n_neighbors in self.n_neighbors_list:
            if n_neighbors > len(df):  # n_neighbors가 샘플 수보다 크면 조정
                n_neighbors = len(df)
                logging.warning(f"n_neighbors is greater than the number of samples. Setting n_neighbors to {n_neighbors}")

            labels = np.zeros(len(df))  # 이상치 라벨 저장을 위한 배열

            for start in range(0, len(df) - self.window_size + 1):
                window_data = df.iloc[start:start + self.window_size]  # 이동 윈도우 데이터 선택
                knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
                knn_model.fit(window_data[feature_columns], np.zeros(len(window_data)))  # 정상 데이터(0)를 가정하여 훈련
                distances, indices = knn_model.kneighbors(window_data[feature_columns])

                # KNN 거리를 기반으로 임의로 이상치를 정의 (최대 거리 상위 5% 이상을 이상치로 정의)
                distance_threshold = np.percentile(distances[:, -1], 95)
                window_labels = np.where(distances[:, -1] > distance_threshold, 1, 0)  # 1: 이상치, 0: 정상
                labels[start:start + self.window_size] = window_labels  # 라벨 저장

            knn_labels[f'knn_{n_neighbors}'] = labels
        return knn_labels

class OutlierProcessor:
    def __init__(self, base_dir, isolation_dir, n_neighbors_list, window_size=30):
        self.base_dir = base_dir
        self.isolation_dir = isolation_dir
        self.n_neighbors_list = n_neighbors_list
        self.window_size = window_size
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

                            # 시계열 분해
                            decomposition = seasonal_decompose(df['value'], model='additive', period=30)
                            df['residual'] = decomposition.resid.bfill().ffill()

                            # 정규화
                            scaler = StandardScaler()
                            df['value_scaled'] = scaler.fit_transform(df[['value']])

                            # KNN 모델을 사용한 이상치 탐지
                            knn_detector = KNNOutlierDetector(n_neighbors_list=self.n_neighbors_list, window_size=self.window_size)
                            knn_labels = knn_detector.fit_predict(df, ['value_scaled'])

                            for key, labels in knn_labels.items():
                                df[key] = labels

                            # 청정구역 찾기
                            clean_intervals = self.clean_zone_finder.find_clean_zones(df)
                            df['clean_zone'] = 0
                            for interval in clean_intervals:
                                mask = (df['time'] >= interval[0]) & (df['time'] <= interval[1])
                                df.loc[mask, 'clean_zone'] = 1

                            # CSV 파일 저장
                            csv_output_filename = f"{folder}_{file_date}_knn.csv"
                            contamination_dir = os.path.join(self.isolation_dir, folder)
                            if not os.path.exists(contamination_dir):
                                os.makedirs(contamination_dir)
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

