import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import time
import yaml
import os
from autoencoder_models import simple_AutoEncoder
import joblib

# config.yaml 파일에서 설정 읽기
with open("config.yaml") as f:
    config = yaml.safe_load(f)

# 데이터 정규화 함수
def normalize_data(df):
    scaler = StandardScaler()
    df['normalized_value'] = scaler.fit_transform(df[['value']])
    return df, scaler

# 시간 변환 및 정렬 함수
def time_convert(df):
    if pd.api.types.is_datetime64_any_dtype(df['time']):
        print("[INFO] 'time' 열이 이미 datetime 형식입니다.")
        return df

    try:
        df['time'] = pd.to_datetime(df['time'], unit='ms', errors='coerce')
        if df['time'].isnull().any():
            print("[INFO] 밀리초 변환 실패, 일반적인 datetime 형식으로 변환을 시도합니다.")
            df['time'] = pd.to_datetime(df['time'], errors='coerce')
        df = df.sort_values(by='time')
        df = df.set_index('time')
        return df

    except Exception as e:
        print(f"[ERROR] 'time' 변환 중 오류 발생: {e}")
        raise



import matplotlib.pyplot as plt
import os
import pandas as pd
import math

import matplotlib.pyplot as plt
import os
import pandas as pd
import math

def visualize_outliers(df, save_path, config, variable_name):
    """
    전체 기간과 월별 이상치 탐지 결과를 각각 별도의 이미지로 시각화하여 저장하는 함수.

    Args:
        df (pd.DataFrame): time, value, outlier 열이 포함된 데이터프레임.
        save_path (str): 이미지 저장 경로.
        config (dict): 시각화 설정을 포함한 config 딕셔너리.
        variable_name (str): 변수명으로, 파일명에 포함됨.
    """
    # 설정 불러오기
    line_color = config['visualization']['line_color']
    outlier_color = config['visualization']['outlier_color']
    
    # 전체 기간 시각화
    plt.figure(figsize=(15, 6))
    plt.plot(df.index, df['value'], color=line_color, label='Value')
    plt.scatter(df[df['outlier']].index, df[df['outlier']]['value'], color=outlier_color, label='Outliers', marker='x')
    plt.title(f'Outlier Detection - Full Period ({variable_name})')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.xticks(rotation=45)
    
    # 전체 기간 이미지 저장
    full_period_image_path = os.path.join(save_path, f'{variable_name}_outlier_full_period.png')
    plt.tight_layout()
    plt.savefig(full_period_image_path)
    plt.close()
    print(f"[INFO] 전체 기간 이상치 시각화 이미지 저장 완료: {full_period_image_path}")

    # 월별 시각화
    df['month'] = df.index.to_period('M')
    months = df['month'].unique()
    num_rows = math.ceil(len(months) / 3)  # 3열로 배치
    fig, axs = plt.subplots(num_rows, 3, figsize=(15, num_rows * 4), sharex=False)
    
    for i, month in enumerate(months):
        month_df = df[df['month'] == month]
        row = i // 3
        col = i % 3
        ax = axs[row, col]
        
        ax.plot(month_df.index, month_df['value'], color=line_color, label='Value')
        ax.scatter(month_df[month_df['outlier']].index, month_df[month_df['outlier']]['value'], 
                   color=outlier_color, label='Outliers', marker='x')
        ax.set_title(f'{month} Outliers')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.legend()
        ax.tick_params(axis='x', rotation=45)
    
    # 월별 이미지 저장
    monthly_image_path = os.path.join(save_path, f'{variable_name}_outlier_monthly.png')
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5)
    plt.savefig(monthly_image_path)
    plt.close()
    print(f"[INFO] 월별 이상치 시각화 이미지 저장 완료: {monthly_image_path}")


def save_model(model, scaler, path):
    model_data = {
        'model_state_dict': model.state_dict(),
        'scaler': scaler
    }
    torch.save(model_data, path)


def load_model(path, model_class, input_size):
    model = model_class(input_size)
    model_data = torch.load(path)
    model.load_state_dict(model_data['model_state_dict'])
    scaler = model_data['scaler']
    return model, scaler

def calculate_mae_percentile_outliers(X, X_reconstructed, percentile=95):
    """
    MAE의 상위 몇 퍼센타를 기준으로 이상치를 탐지합니다.

    Args:
        X (np.array): 원본 데이터.
        X_reconstructed (np.array): 재구성된 데이터.
        percentile (int): 이상치로 간주할 퍼센타일 (기본값: 상위 5%인 95 퍼센타일).

    Returns:
        np.array: 이상치 여부가 표시된 boolean 배열.
    """
    mae_errors = np.mean(np.abs(X - X_reconstructed), axis=1)
    threshold = np.percentile(mae_errors, percentile)
    return mae_errors, threshold

def calculate_mae_outliers(X, X_reconstructed, n_std=2):
    """
    MAE와 표준편차를 사용해 임계치를 설정하여 이상치를 탐지합니다.

    Args:
        X (np.array): 원본 데이터.
        X_reconstructed (np.array): 재구성된 데이터.
        n_std (float): 표준편차 배수를 임계치로 사용 (기본값: 2 표준편차).

    Returns:
        np.array: 이상치 여부가 표시된 boolean 배열.
    """
    mae_errors = np.mean(np.abs(X - X_reconstructed), axis=1)
    threshold = np.mean(mae_errors) + n_std * np.std(mae_errors)
    return mae_errors, threshold


def calculate_mse_percentile_outliers(X, X_reconstructed, percentile=80):
    """
    MAE의 상위 몇 퍼센타를 기준으로 이상치를 탐지합니다.

    Args:
        X (np.array): 원본 데이터.
        X_reconstructed (np.array): 재구성된 데이터.
        percentile (int): 이상치로 간주할 퍼센타일 (기본값: 상위 5%인 95 퍼센타일).

    Returns:
        np.array: 이상치 여부가 표시된 boolean 배열.
    """
    mse_errors = np.mean((X - X_reconstructed) ** 2, axis=1)
    threshold = np.percentile(mse_errors, percentile)
    return mse_errors, threshold

def calculate_mse_outliers(X, X_reconstructed, n_std=2):
    """
    MAE와 표준편차를 사용해 임계치를 설정하여 이상치를 탐지합니다.

    Args:
        X (np.array): 원본 데이터.
        X_reconstructed (np.array): 재구성된 데이터.
        n_std (float): 표준편차 배수를 임계치로 사용 (기본값: 2 표준편차).

    Returns:
        np.array: 이상치 여부가 표시된 boolean 배열.
    """
    mse_errors = np.mean((X - X_reconstructed) ** 2, axis=1)
    threshold = np.mean(mse_errors) + n_std * np.std(mse_errors)
    return mse_errors, threshold


def calculate_z_score_reconstruction_error(X, X_reconstructed, threshold=2):
    """
    Z-score를 사용하여 이상치를 탐지합니다.

    Args:
        X (np.array): 원본 데이터.
        X_reconstructed (np.array): 재구성된 데이터.
        threshold (float): 이상치로 간주할 Z-score 임계값.

    Returns:
        np.array: 이상치 여부가 표시된 boolean 배열.
    """
    reconstruction_error = np.mean((X - X_reconstructed) ** 2, axis=1)
    mean_error = np.mean(reconstruction_error)
    std_error = np.std(reconstruction_error)

    z_scores = (reconstruction_error - mean_error) / std_error
    return threshold, z_scores