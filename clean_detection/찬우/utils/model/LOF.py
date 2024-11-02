import os
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import yaml
from tqdm import tqdm

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 청정구역 탐색 모델 정의
class CleanZoneFinder:
    def __init__(self, time_column='time', min_clean_length=30):
        self.time_column = time_column
        self.min_clean_length = min_clean_length

    def find_clean_zones(self, df, label_column):
        clean_intervals = []
        clean_start = None
        clean_length = 0

        for i in range(len(df)):
            if df[label_column].iloc[i] == 0:
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

# LOF 이상치 탐지 클래스
class LOFOutlierDetector:
    def __init__(self, n_neighbors_list):
        self.n_neighbors_list = n_neighbors_list

    def fit_predict(self, df, feature_columns):
        for n_neighbors in self.n_neighbors_list:
            lof_model = LocalOutlierFactor(n_neighbors=n_neighbors)
            labels = lof_model.fit_predict(df[feature_columns])
            df[f'lof_outlier_{n_neighbors}'] = np.where(labels == -1, 1, 0)  # -1을 이상치로 간주
        return df

# LOFProcessor 클래스 - 데이터 처리 및 이상치 탐지
class LOFProcessor:
    def __init__(self, base_dir, isolation_dir, n_neighbors_list, min_clean_length):
        self.base_dir = base_dir
        self.isolation_dir = isolation_dir
        self.n_neighbors_list = n_neighbors_list
        self.min_clean_length = min_clean_length
        self.clean_zone_finder = CleanZoneFinder(min_clean_length=self.min_clean_length)

        # 결과 저장 디렉토리 생성
        if not os.path.exists(isolation_dir):
            os.makedirs(isolation_dir)
                
    def process_data(self):
        # data 폴더의 모든 CSV 파일을 처리
        csv_files = [f for f in os.listdir(self.base_dir) if f.endswith('.csv')]

        logging.info(f"CSV files in {self.base_dir}:")
        for file_name in csv_files:
            logging.info(file_name)

        for file_name in tqdm(csv_files, desc="Processing CSV files", unit='file'):
            file_path = os.path.join(self.base_dir, file_name)
            logging.info(f"Processing: {file_path}")

            try:
                # 데이터 읽기
                df = pd.read_csv(file_path)
                logging.info(f"Data read successfully: {file_path}")

                # NaN 값 처리
                if df.isnull().values.any():
                    logging.warning(f"NaN values found in {file_name}, filling NaNs with forward fill method.")
                    df.fillna(method='ffill', inplace=True)
                    df.fillna(method='bfill', inplace=True)

                df['time'] = pd.to_datetime(df['time'], unit='ms')

                # 스케일링
                scaler = StandardScaler()
                df['value_scaled'] = scaler.fit_transform(df[['value']])

                # LOF 모델을 사용해 이상치를 탐지
                lof_detector = LOFOutlierDetector(n_neighbors_list=self.n_neighbors_list)
                df = lof_detector.fit_predict(df, ['value_scaled'])

                for n_neighbors in self.n_neighbors_list:
                    # 청정구역 찾기
                    clean_intervals = self.clean_zone_finder.find_clean_zones(df, f'lof_outlier_{n_neighbors}')
                    df[f'clean_zone_{n_neighbors}'] = 1  # 기본값을 1로 설정 (이상치)
                    for interval in clean_intervals:
                        mask = (df['time'] >= interval[0]) & (df['time'] <= interval[1])
                        df.loc[mask, f'clean_zone_{n_neighbors}'] = 0  # 청정구역을 0으로 설정

                    # 결과 저장 (time과 anomaly 컬럼만 포함)
                    output_file = os.path.join(self.isolation_dir, f"{file_name}_LOF_{n_neighbors}.csv")
                    df_reset = df.reset_index()[['time', f'clean_zone_{n_neighbors}']].rename(columns={f'clean_zone_{n_neighbors}': 'anomaly'})
                    df_reset.to_csv(output_file, index=False)
                    logging.info(f"Saved labeled data to {output_file}")

                    # 월별 데이터 시각화 - 3x5 서브플롯 생성
                    df['month'] = df['time'].dt.to_period("M")
                    unique_months = df['month'].unique()[:15]  # 최대 15개월만 표시

                    fig, axs = plt.subplots(3, 5, figsize=(20, 12), sharex=False)
                    fig.suptitle(f"{file_name} - Monthly LOF Outliers and Clean Zones (n_neighbors={n_neighbors})", fontsize=16)

                    for i, month in enumerate(unique_months):
                        ax = axs[i // 5, i % 5]
                        monthly_data = df[df['month'] == month]

                        # 각 월별 데이터 시각화
                        ax.plot(monthly_data['time'], monthly_data['value'], label='Value')
                        ax.scatter(monthly_data[monthly_data[f'lof_outlier_{n_neighbors}'] == 1]['time'], 
                                   monthly_data[monthly_data[f'lof_outlier_{n_neighbors}'] == 1]['value'], 
                                   color='red', label='Outliers')
                        ax.fill_between(monthly_data['time'], monthly_data['value'], 
                                        where=(monthly_data[f'clean_zone_{n_neighbors}'] == 1), color='blue', alpha=0.3, label='Clean Zone')
                        ax.set_title(f"Month: {month.strftime('%Y-%m')}", fontsize=10)
                        ax.legend(fontsize=8)

                        # 각 서브플롯의 x축을 월별 데이터로 설정
                        ax.set_xlim(monthly_data['time'].min(), monthly_data['time'].max())
                        ax.tick_params(axis='x', rotation=45)  # x축 눈금 회전

                    plt.tight_layout(rect=[0, 0, 1, 0.96])
                    plt.savefig(os.path.join(self.isolation_dir, f"{file_name}_monthly_outliers_{n_neighbors}.png"))
                    plt.close()

            except Exception as e:
                logging.error(f"Error processing {file_path}: {e}")
                continue

# 실행 예시
if __name__ == "__main__":
    base_dir = "C:/Users/ansck/Documents/ybigta/DA_25/project/AD_detection/clean_detection/data"
    isolation_dir = "C:/Users/ansck/Documents/ybigta/DA_25/project/AD_detection/clean_detection/output"

    processor = LOFProcessor(base_dir=base_dir, isolation_dir=isolation_dir, n_neighbors_list=[20, 30, 40], min_clean_length=30)
    processor.process_data()
