import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import time

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
        df['time'] = pd.to_datetime(df['time'], unit='ms', errors='coerce')  # 밀리초 단위로 변환
        if df['time'].isnull().any():
            print("[INFO] 밀리초 변환 실패, 일반적인 datetime 형식으로 변환을 시도합니다.")
            df['time'] = pd.to_datetime(df['time'], errors='coerce')  # 일반적인 datetime 형식으로 변환

        df = df.sort_values(by='time')
        df = df.set_index('time')
        return df

    except Exception as e:
        print(f"[ERROR] 'time' 변환 중 오류 발생: {e}")
        raise

# DeepAutoEncoder 모델 정의
class DeepAutoEncoder(nn.Module):
    def __init__(self, input_size):
        super(DeepAutoEncoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(True),
            nn.Linear(64, 32),
            nn.ReLU(True),
            nn.Linear(32, 16),
            nn.ReLU(True)  # 인코더의 출력은 16차원으로 축소
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(True),
            nn.Linear(32, 64),
            nn.ReLU(True),
            nn.Linear(64, input_size),  # 디코더의 출력은 원래 입력 차원으로 복원
            nn.Sigmoid()  # 출력이 0~1 사이의 값으로 제한되도록 sigmoid 함수 적용
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# DeepAutoEncoder를 사용한 이상치 탐지 및 시각화 함수
def detect_and_visualize_outliers_deep_autoencoder(df, file_name, save_path, custom_window_size=5, threshold_signal= None):
    # 시작 시간 기록
    start_time = time.time()

    # 데이터 정규화
    df, scaler = normalize_data(df)
    print(f"[INFO] 데이터 정규화 완료: {file_name}")

    # 시계열 데이터를 학습에 사용할 수 있도록 이전 값을 특성으로 사용
    window_size = custom_window_size
    if window_size >= len(df):
        print(f"[ERROR] Window size {window_size}가 데이터 길이보다 큽니다: {file_name}")
        return

    X, y = [], []
    for i in range(window_size, len(df)):
        X.append(df['normalized_value'].iloc[i-window_size:i].values)
        y.append(df['normalized_value'].iloc[i])

    X = np.array(X)
    y = np.array(y)
    print(f"[INFO] 시계열 데이터 생성 완료: {file_name}, Window Size: {window_size}, X 크기: {X.shape}, y 크기: {y.shape}")

    # PyTorch Tensor로 변환 및 GPU 사용 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X = torch.tensor(X, dtype=torch.float32).to(device)

    # DeepAutoEncoder 모델 정의 및 학습 설정
    input_size = window_size
    model = DeepAutoEncoder(input_size).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # 모델 학습
    num_epochs = 60
    model.train()
    for epoch in range(num_epochs):
        outputs = model(X)
        loss = criterion(outputs, X)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"[INFO] Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}")

    # 학습 완료 시간 기록
    training_time = time.time() - start_time
    print(f"[INFO] 모델 학습 완료: {file_name}, 총 학습 시간: {training_time:.2f}초")

    # 예측 (복원)
    model.eval()
    with torch.no_grad():
        X_reconstructed = model(X).cpu().numpy()
    X_original = X.cpu().numpy()
    reconstruction_error = np.mean((X_original - X_reconstructed) ** 2, axis=1)
    print(f"[INFO] 예측 및 reconstruction error 계산 완료: {file_name}")

    if threshold_signal == 1 :
        threshold = 2
    elif threshold_signal == 2:
        threshold = 1.5
    elif threshold_signal == 3 :
        threshold = 3
    else :
        threshold = np.mean(reconstruction_error) + 2 * np.std(reconstruction_error)

    # 이상치 탐지 (reconstruction error 기준)
    outliers = reconstruction_error > threshold
    print(f"사용한 임계치 : {threshold}")
    print(f"[INFO] 이상치 탐지 완료: {file_name}, 총 이상치 수: {np.sum(outliers)}")

    # 결과를 DataFrame에 추가
    df = df.iloc[window_size:].copy()
    df['reconstruction_error'] = reconstruction_error
    df['outlier'] = outliers
    df['clean_zone'] = ~df['outlier']
    print(f"[INFO] 결과 DataFrame 업데이트 완료: {file_name}")

    # CSV 파일 저장
    csv_output_path = f"{save_path}/{file_name}_deep_autoencoder_results_{window_size}.csv"
    df.to_csv(csv_output_path)
    print(f"[INFO] 결과 CSV 파일 저장 완료: {csv_output_path}")

    # 시각화
    plt.figure(figsize=(12, 10))

    # 원본 값과 복원 값
    plt.subplot(3, 1, 1)
    plt.plot(df.index, df['normalized_value'], label='Normalized Value', color='blue')
    plt.plot(df.index, X_reconstructed.mean(axis=1), label='Reconstructed', color='red')
    plt.title(f'Normalized vs Reconstructed - {file_name} (window size: {window_size})')
    plt.legend()

    # Reconstruction Error 및 이상치 표시
    plt.subplot(3, 1, 2)
    plt.plot(df.index, df['reconstruction_error'], label='Reconstruction Error', color='gray')
    plt.scatter(df.index[df['outlier']], df['reconstruction_error'][df['outlier']], color='red', label='Outliers')
    plt.title(f'Reconstruction Error with Outliers - {file_name} (window size: {window_size})')
    plt.legend()

    # 이상치 시각화 (원본 데이터에서의 위치)
    plt.subplot(3, 1, 3)
    plt.plot(df.index, df['value'], label='Original Value', color='blue')
    plt.scatter(df.index[df['outlier']], df['value'][df['outlier']], color='red', label='Outliers')
    plt.title(f'Original Value with Outliers - {file_name} (window size: {window_size})')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'{save_path}/{file_name}_deep_autoencoder_visualization_{window_size}.png')
    plt.close()
    print(f"[INFO] 결과 시각화 이미지 저장 완료: {save_path}/{file_name}_deep_autoencoder_visualization_{window_size}.png")
