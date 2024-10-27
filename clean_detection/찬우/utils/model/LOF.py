import os
import pickle
import pandas as pd
import numpy as np
import re
import logging
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from tqdm import tqdm

# 로깅 설정 - 로그 메시지 형식을 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 데이터 전처리 클래스 (Preheater)<- 350~380 사이 정상범위 이미 preheter.py에서 처리해서 이를 불러오는 부분임
class Preheater:
    def __init__(self, interpolate_method):
        # 결측치 처리 방법 설정
        self.interpolate_method = interpolate_method

    # 데이터 전처리 메서드
    def preprocess(self, df, time_column='time', value_column='value'):
        # 시간 데이터를 datetime 형식으로 변환
        df[time_column] = pd.to_datetime(df[time_column], unit='ms')
        # 결측치를 지정한 방식으로 보간 및 채움
        df[value_column] = df[value_column].interpolate(method=self.interpolate_method).ffill().bfill()
        return df








# 청정구역 탐색 모델 정의 
class CleanZoneFinder:
    def __init__(self, time_column='time', min_clean_length=20):
        # 시간 열과 청정구역 최소 길이 설정(20으로 초기값 설정)
        self.time_column = time_column
        self.min_clean_length = min_clean_length

    # 청정구역을 찾는 메서드
    def find_clean_zones(self, df, label_column):
        clean_intervals = []
        clean_start = None
        clean_length = 0

        # 연속된 정상 상태를 청정구역으로 판별
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

# LOF 이상치 탐지 클래스(LOFOutlierDetector) 이상치 라벨링 부분

class LOFOutlierDetector:
    def __init__(self, n_neighbors_list):
        # 사용할 n_neighbors 목록 설정
        self.n_neighbors_list = n_neighbors_list

    # LOF 모델을 사용해 이상치를 탐지하고 결과를 반환
    def fit_predict(self, df, feature_columns):
        for n_neighbors in self.n_neighbors_list:
            lof_model = LocalOutlierFactor(n_neighbors=n_neighbors)
            labels = lof_model.fit_predict(df[feature_columns])
            df[f'lof_outlier_{n_neighbors}'] = np.where(labels == -1, 1, 0)  # -1을 이상치로 간주
        return df













# LOFProcessor 클래스 - 데이터 처리 및 이상치 탐지-> input파일을 어떻게 받을건지 저장은 어떻게 할것인지 정의하는 부분이여서 그냥 
class LOFProcessor:
    def __init__(self, base_dir, isolation_dir, n_neighbors_list, combined_outliers_file, interpolate_method='linear'):
        # 기본 설정 및 경로 설정
        self.base_dir = base_dir
        self.isolation_dir = isolation_dir
        self.n_neighbors_list = n_neighbors_list
        self.date_pattern = re.compile(r'\d{6}')
        self.preheater = Preheater(interpolate_method=interpolate_method)
        self.clean_zone_finder = CleanZoneFinder()
        self.existing_outliers = self.load_existing_outliers(combined_outliers_file)

        # 결과 저장 디렉토리 생성
        if not os.path.exists(isolation_dir):
            os.makedirs(isolation_dir)



#############################################################################################

    # 기존 이상치 정보를 로드하는 메서드<- preheater에서 불러옴
    def load_existing_outliers(self, file_path):
        if os.path.exists(file_path):
            combined_df = pd.read_csv(file_path)
            return set(combined_df[combined_df['is_outlier'] == 1]['time'])
        else:
            logging.warning(f"기존 이상치 파일이 존재하지 않습니다: {file_path}")
            return set()


##################################################################################################



    # 데이터 처리 메서드 - 폴더별로 파일을 처리    파일 처리알고리즘 정의 ( 단일파일로 할거면 for문을 빼고 try부분 하나만 처리하면 됨)
    def process_data(self, folder_list, months):
        overall_results = []  # 전체 폴더의 결과를 저장할 리스트

        for folder in folder_list:
            folder_path = os.path.join(self.base_dir, folder)
            csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

            folder_results = []  # 폴더별 결과 저장
            folder_output_dir = os.path.join(self.isolation_dir, folder)
            os.makedirs(folder_output_dir, exist_ok=True)

            # 서브플롯을 위한 설정
            fig, axs = plt.subplots(len(csv_files), 1, figsize=(14, 5 * len(csv_files)), sharex=True)
            fig.suptitle(f"Folder: {folder} - Combined Subplots")

            for idx, file_name in enumerate(tqdm(csv_files, desc=f"Processing folder {folder}", unit='file')):
                match = self.date_pattern.search(file_name)
                if match:
                    file_date = match.group(0)
                    if file_date in months:
                        file_path = os.path.join(folder_path, file_name)
                        logging.info(f"Processing: {file_path}")

                        try:
                            
                            
                            
                            
                            
########################################################################################################################
# 실제로 preheater+ LOFOutlierDetector를 사용해 이상치 탐지하는 부분 

                            # 데이터 읽기 및 전처리
                            df = pd.read_csv(file_path)
                            df = self.preheater.preprocess(df)

                            # 기존 이상치 라벨링
                            df['predefined_outlier'] = df['time'].apply(lambda x: 1 if x in self.existing_outliers else 0)

                            # 시계열 분해 및 정규화
                            decomposition = seasonal_decompose(df['value'], model='additive', period=30)
                            df['residual'] = decomposition.resid.ffill().bfill()
                            scaler = StandardScaler()
                            df['value_scaled'] = scaler.fit_transform(df[['value']])

                            # LOF 모델을 사용해 이상치를 탐지
                            lof_detector = LOFOutlierDetector(n_neighbors_list=self.n_neighbors_list)
                            df = lof_detector.fit_predict(df, ['value_scaled'])

                            # 모든 이상치 라벨의 합집합을 기준으로 최종 이상치 라벨링
                            lof_outlier_columns = [f'lof_outlier_{n}' for n in self.n_neighbors_list]
                            df['final_outlier'] = df[lof_outlier_columns].max(axis=1).combine(df['predefined_outlier'], max)

                            # 청정구역 찾기
                            clean_intervals = self.clean_zone_finder.find_clean_zones(df, 'final_outlier')
                            df['clean_zone'] = 0
                            for interval in clean_intervals:
                                mask = (df['time'] >= interval[0]) & (df['time'] <= interval[1])
                                df.loc[mask, 'clean_zone'] = 1
# 결과 리스트에 추가
                            folder_results.append(df)
                            overall_results.append(df)
################################################################################3#################################################





                            

                            # 개별 시각화 생성
                            axs[idx].plot(df['time'], df['value'], label='Value')
                            axs[idx].scatter(df[df['final_outlier'] == 1]['time'], df[df['final_outlier'] == 1]['value'], color='red', label='Final Outliers')
                            axs[idx].fill_between(df['time'], df['value'], where=(df['clean_zone'] == 1), color='blue', alpha=0.3, label='Clean Zone')
                            axs[idx].set_title(f"{file_name}")
                            axs[idx].legend()

                            # 개별 파일 시각화 저장
                            individual_plot_filename = f"{file_name}_outliers.png"
                            plt.figure(figsize=(14, 7))
                            plt.plot(df['time'], df['value'], label='Value')
                            plt.scatter(df[df['final_outlier'] == 1]['time'], df[df['final_outlier'] == 1]['value'], color='red', label='Final Outliers')
                            plt.fill_between(df['time'], df['value'], where=(df['clean_zone'] == 1), color='blue', alpha=0.3, label='Clean Zone')
                            plt.title(f"{file_name} - Outliers and Clean Zones")
                            plt.legend()
                            plt.savefig(os.path.join(folder_output_dir, individual_plot_filename))
                            plt.close()

                        except Exception as e:
                            logging.error(f"Error processing {file_path}: {e}")
                            continue

            # 폴더별 서브플롯 저장
            combined_plot_filename = f"{folder}_combined_outliers.png"
            plt.tight_layout()
            plt.savefig(os.path.join(folder_output_dir, combined_plot_filename))
            plt.close(fig)

            # 폴더별 PKL 파일 저장
            if folder_results:
                folder_df = pd.concat(folder_results, ignore_index=True)
                folder_pkl_path = os.path.join(self.isolation_dir, f"{folder}_results.pkl")
                with open(folder_pkl_path, "wb") as f:
                    pickle.dump(folder_df, f)
                logging.info(f"Folder results saved to {folder_pkl_path}")

        # 전체 데이터를 포함한 통합 PKL 파일 저장
        if overall_results:
            overall_df = pd.concat(overall_results, ignore_index=True)
            overall_pkl_path = os.path.join(self.isolation_dir, "overall_results.pkl")
            with open(overall_pkl_path, "wb") as f:
                pickle.dump(overall_df, f)
            logging.info("Overall results saved to overall_results.pkl")
