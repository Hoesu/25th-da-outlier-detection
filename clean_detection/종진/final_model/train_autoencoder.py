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

def train_autoencoder(file_path, ws):
    # 데이터 불러오기 및 정규화    
    start_time = time.time()
    
    # 모델 선택 및 설정
    df = pd.read_csv(file_path)
    df = time_convert(df)
    df, scaler = normalize_data(df)
    print(f"[INFO] 데이터 정규화 완료: {file_path}")
    
    # 모델 설정
    input_size = ws
    layers = config['model']['layers']
    hidden_size = 32
    epochs = config['model']['epochs']
    learning_rate = config['model']['learning_rate']

    if input_size >= len(df):
        print(f"[ERROR] Window size {input_size}가 데이터 길이보다 큽니다: {file_path}")
        return

    if config['model']['type'] == "deep_autoencoder" and layers == 3 :
        model = DeepAutoEncoder_3layer(input_size)
    elif config['model']['type'] == 'deep_autoencoder' and layers == 5 :
        model =  DeepAutoEncoder_5layer(input_size)
    else :
        model = simple_AutoEncoder(input_size,hidden_size)
        print('심플한 오토인코더가 사용됩니다')


    # 학습 파라미터
    X, y = [], []
    for i in range(input_size, len(df)):
        X.append(df['normalized_value'].iloc[i-input_size:i].values)
        y.append(df['normalized_value'].iloc[i])

    X = np.array(X)
    y = np.array(y)
    print(f"[INFO] 시계열 데이터 생성 완료: {file_path}, Window Size: {input_size}, X 크기: {X.shape}, y 크기: {y.shape}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 데이터를 장치로 이동
    X = torch.tensor(X, dtype=torch.float32).to(device)
    y = torch.tensor(y, dtype=torch.float32).to(device)

    # 모델을 장치로 이동
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()
    
    # 학습 루프
    model.train()
    for epoch in range(epochs):
        outputs = model(X)
        loss = criterion(outputs, X)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"[INFO] Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")

    training_time = time.time() - start_time
    print(f"[INFO] 모델 학습 완료: {file_path}, 총 학습 시간: {training_time:.2f}초")

    var_name = os.path.splitext(os.path.basename(file_path))[0]
    variable_folder = os.path.join("trained_models", var_name)  # 변수 이름별 폴더 생성 경로
    os.makedirs(variable_folder, exist_ok=True)
    
    output_path = os.path.join(variable_folder, f"{var_name}_{model.__class__.__name__}_ws{input_size}.pkl")
    save_model(model, scaler, output_path)
    print(f"[INFO] 모델 및 스케일러 저장 완료: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AutoEncoder Model")
    parser.add_argument('-i', "--input_path", type=str, required=True, help="Path to the input CSV file for training")
    parser.add_argument('-w', "--window_sizes", type=int, nargs='+', required=True, help="List of window sizes for training")
    args = parser.parse_args()
    
    # 여러 개의 window_size 값 처리
    for w in args.window_sizes:
        train_autoencoder(args.input_path, w)