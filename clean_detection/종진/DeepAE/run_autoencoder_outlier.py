import pandas as pd
import os
from tqdm import tqdm
from AutoEncoder_clean_detection import detect_and_visualize_outliers_autoencoder, time_convert


def process_data_with_autoencoder(data_directory, save_directory, window_sizes, autoencoder):


    # 데이터 디렉토리 및 저장 경로 설정
    combined_dir = os.path.join(data_directory, 'combined_data')
    if not os.path.exists(combined_dir):
        raise FileNotFoundError(f"[ERROR] 데이터 디렉토리 '{combined_dir}'가 존재하지 않습니다. 경로를 확인하세요.")

    csv_files = [f for f in os.listdir(combined_dir) if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError(f"[ERROR] '{combined_dir}'에 CSV 파일이 없습니다. 데이터를 확인하세요.")

    # 각 window_size에 대해 반복 실행
    for window_size in tqdm(window_sizes, desc="Processing window sizes", unit="window"):
        # window_size별 결과 저장 폴더 생성
        window_save_dir = os.path.join(save_directory, f'window_size_{window_size}')
        os.makedirs(window_save_dir, exist_ok=True)

        # 각 CSV 파일 처리
        threshold_signal = 0
        # 각 CSV 파일 처리
        for file_name in tqdm(csv_files, desc=f"Processing CSV files (window size: {window_size})", unit="file"):
            file_path = os.path.join(combined_dir, file_name)
            if file_name in ['423_BE1_IZ1_combined_data.csv', '433_CG1_ZC1_RWIN_combined_data.csv', '433_CG2_ZC1_RWIN_combined_data.csv', '463_KH1_TC1_RXEX_combined_data.csv']:
                threshold_signal == 1
            elif file_name in ['463_KL1_SC1_RWIN_combined_data.csv'] :
                threshold_signal == 2
            elif file_name in ['433_DU4_UZ1_combined_data.csv'] :
                threshold_signal == 3
            else :
                threshold_signal == 4
            try:
                df = pd.read_csv(file_path)
                print(f"[INFO] 파일 읽기 시작: {file_name}, 데이터 크기: {df.shape}")

                # 시간 변환 및 정렬
                df = time_convert(df)

                # AutoEncoder 이상치 탐지 및 시각화 (window_size 폴더에 결과 저장)
                detect_and_visualize_outliers_autoencoder(df, file_name, window_save_dir, window_size, autoencoder, threshold_signal)

                print(f"[INFO] 파일 처리 완료: {file_name}")

            except Exception as e:
                print(f"[ERROR] {file_path} 처리 중 오류 발생: {e}")
                continue
