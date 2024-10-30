import argparse
import yaml
from autoencoder_models import simple_AutoEncoder, DeepAutoEncoder_3layer, DeepAutoEncoder_5layer
import torch
import pandas as pd
from utils import normalize_data, save_model, time_convert
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
import os
# config.yaml 파일에서 설정 읽기
with open("config.yaml") as f:
    config = yaml.safe_load(f)

def train_autoencoder(file_path):
    # 데이터 불러오기 및 정규화    
    start_time = time.time()
    df = pd.read_csv(file_path)
    df = time_convert(df)
    df, scaler = normalize_data(df)
    print(f"[INFO] 데이터 정규화 완료: {file_path}")
    
    # 모델 설정
    input_size = config['model']['window_size']
    layers = config['model']['layers']
    hidden_size = config['model']['hidden_size']
    epochs = config['model']['epochs']
    learning_rate = config['model']['learning_rate']

    if input_size >= len(df):
        print(f"[ERROR] Window size {input_size}가 데이터 길이보다 큽니다: {file_path}")
        return

    # 모델 초기화
    if config['model']['type'] == "deep_autoencoder" and layers == 3:
        model = DeepAutoEncoder_3layer(input_size)
    elif config['model']['type'] == 'deep_autoencoder' and layers == 5:
        model = DeepAutoEncoder_5layer(input_size)
    else:
        model = simple_AutoEncoder(input_size, hidden_size)
        print('심플한 오토인코더가 사용됩니다')

    # 학습 파라미터 설정
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()
    
    # 시계열 데이터 생성
    X = np.array([df['normalized_value'].iloc[i-input_size:i].values for i in range(input_size, len(df))])
    X = torch.tensor(X, dtype=torch.float32).to(config['device'])
    model.train()

    # 모델 학습
    for epoch in range(epochs):
        outputs = model(X)
        loss = criterion(outputs, X)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"[INFO] Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")

    print(f"[INFO] 모델 학습 완료: {file_path}, 총 학습 시간: {time.time() - start_time:.2f}초")
    
    # 모델과 스케일러 저장
    variable_name = os.path.splitext(os.path.basename(file_path))[0]
    variable_folder = os.path.join("trained_models", variable_name)  # 변수 이름별 폴더 생성 경로
    os.makedirs(variable_folder, exist_ok=True)  # 변수 이름 폴더 생성

    output_path = os.path.join(variable_folder, f"{variable_name}_{model.__class__.__name__}_ws{input_size}.pkl")
    save_model(model, scaler, output_path)
    print(f"[INFO] 모델 및 스케일러 저장 완료: {output_path}")

def train_autoencoder_for_folder(folder_path):
    # 폴더 내 모든 CSV 파일 처리
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):
            file_path = os.path.join(folder_path, file_name)
            print(f"\n[INFO] 파일 처리 중: {file_name}")
            train_autoencoder(file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AutoEncoder Models for All Files in a Folder")
    parser.add_argument('-f', "--folder_path", type=str, required=True, help="Path to the folder containing CSV files for training")
    args = parser.parse_args()
    train_autoencoder_for_folder(args.folder_path)
