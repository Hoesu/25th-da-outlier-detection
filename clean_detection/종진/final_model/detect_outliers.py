import argparse
import yaml
import pandas as pd
import torch
from autoencoder_models import simple_AutoEncoder, DeepAutoEncoder_3layer, DeepAutoEncoder_5layer
from utils import load_model, time_convert, visualize_outliers, normalize_data, calculate_z_score_reconstruction_error, calculate_mae_outliers, calculate_mae_percentile_outliers, calculate_mse_percentile_outliers, calculate_mse_outliers
import os
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity



with open("config.yaml") as f:
    config = yaml.safe_load(f)

def detect_outliers(variable_folder, data_folder, config):
    var_name = os.path.splitext(os.path.basename(variable_folder))[0]
    # 파일 목록 가져오기
    model_files = [f for f in os.listdir(variable_folder) if f.endswith('.pkl')]
    variable_output_folder = os.path.join('output_result', var_name)
    os.makedirs(variable_output_folder, exist_ok=True)

    for model_file in model_files:
        # 모델 경로와 이름
        model_path = os.path.join(variable_folder, model_file)

        # 파일명에서 window_size 추출
        match = re.search(r'ws(\d+)', model_file)
        if match:
            input_size = int(match.group(1))
        else:
            print(f"[ERROR] 파일 이름에서 window_size를 찾을 수 없습니다: {model_file}")
            continue
        hidden_size = 32
        
        # 모델 종류에 따라 클래스 설정
        if "DeepAutoEncoder_3layer" in model_file:
            model_class = DeepAutoEncoder_3layer
        elif "DeepAutoEncoder_5layer" in model_file:
            model_class = DeepAutoEncoder_5layer
        else:
            model_class = simple_AutoEncoder

        # 모델과 스케일러 로드
        model, scaler = load_model(model_path, model_class, input_size)
        model.eval()

        # 데이터 불러오기 및 정규화
        df = pd.read_csv(data_folder)
        df = time_convert(df)
        df['normalized_value'] = scaler.transform(df[['value']])
        
        # 탐지 파라미터 설정
        X = []

        # 모델에 입력할 데이터 준비 (window_size 만큼씩 슬라이싱)
        for i in range(input_size, len(df)):
            X.append(df['normalized_value'].iloc[i-input_size:i].values)
        
        X = torch.tensor(X, dtype=torch.float32).to(next(model.parameters()).device)
        
        # 모델 예측
        with torch.no_grad():
            X_reconstructed = model(X).cpu().numpy()
        
        # Reconstruction error 계산
        X_original = X.cpu().numpy()
        reconstruction_error = np.zeros(len(X_original))  # 기본값으로 초기화
        threshold = None  # 기본값

        if config['model']['reconstruction_error'] == 'mse' :
            if config['model']['anomaly_threshold'] == 'std' :
                reconstruction_error, threshold = calculate_mse_outliers(X_original, X_reconstructed)
            elif config['model']['anomaly_threshold'] == 'percentile' :
                reconstruction_error, threshold = calculate_mse_percentile_outliers(X_original, X_reconstructed)
            elif config['model']['anomaly_threshold'] == 2 :
                reconstruction_error = np.mean((X_original - X_reconstructed) ** 2, axis=1)
                threshold = 2
            else :
                print('지정된 임계값 없음')

            outliers = reconstruction_error > threshold

        elif config['model']['reconstruction_error'] == 'mae':
            if config['model']['anomaly_threshold'] == 'std' :
                reconstruction_error, threshold = calculate_mae_outliers(X_original, X_reconstructed)
            elif config['model']['anomaly_threshold'] == 'percentile' :
                reconstruction_error, threshold = calculate_mae_percentile_outliers(X_original, X_reconstructed)
            elif config['model']['anomaly_threshold'] == 2 :
                reconstruction_error = np.mean(np.abs(X_original - X_reconstructed), axis=1)
                threshold = 2
            outliers = reconstruction_error > threshold

        elif config['model']['reconstruction_error'] == 'z-score':
            threshold , z_scores = calculate_z_score_reconstruction_error(X_original, X_reconstructed)
            outliers = z_scores > threshold

        elif config['model']['reconstruction_error'] == 'cosine_sim':
            cosine_sim = cosine_similarity(X_original, X_reconstructed)
            cosine_sim = cosine_sim.diagonal()  # 각 샘플의 유사도 (대각 성분만 추출)
            threshold = 0.8
            outliers = cosine_sim < threshold

        else :
            print('지정된 재구성 error 없음')
            

        print(f"[INFO] 예측 및 reconstruction error 계산 완료")

        # Threshold 설정
        df['outlier'] = False

        df = df.iloc[input_size:].copy()
        df['reconstruction_error'] = reconstruction_error
        df['outlier'] = outliers
        df['clean_zone'] = ~df['outlier']
        print(f"[INFO] 결과 DataFrame 업데이트 완료: {var_name}")

        print(f"사용된 임계치 : {threshold}")
        print(f"[INFO] 이상치 탐지 완료, 총 이상치 수: {np.sum(outliers)}")

        recons = config['model']['reconstruction_error']

        # CSV 파일 저장
        csv_output_path = os.path.join(variable_output_folder,  f"{recons}_{var_name}_{model.__class__.__name__}, ws{input_size}_outliers.csv")
        df.to_csv(csv_output_path, index=False)
        print(f"[INFO] 이상치 탐지 결과 저장 완료: {csv_output_path}")
        # 전체 기간에 대한 시각화 저장
        png_file = f'{recons}_{var_name}_{model.__class__.__name__}_ws{input_size}'
        #os.makedirs(png_file, exist_ok=True)  # 변수 이름 폴더 생성
        visualize_outliers(df, variable_output_folder, config, png_file)


        print(f"[INFO] 모든 이상치 탐지 및 시각화 저장 완료: {var_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect Outliers with Trained Models")
    parser.add_argument('-m', "--model_folder", type=str, required=True, help="Path to the folder containing the trained models for a variable")
    parser.add_argument('-d', '--data_folder', type=str, required=True)
    args = parser.parse_args()

    detect_outliers(args.model_folder, args.data_folder, config)