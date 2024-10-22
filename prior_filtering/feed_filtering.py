import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import joblib  # 모델 저장 및 로드

def load_and_merge_csv(feed1, feed2):
    # 두 파일을 읽어서 DataFrame 생성
    data1 = pd.read_csv(feed1)
    data2 = pd.read_csv(feed2)

    # 두 DataFrame을 time 열 기준으로 내부 조인 (inner join)
    merged_data = pd.merge(data1, data2, on='time', suffixes=('_1', '_2'))

    # value 열끼리 더하기
    merged_data['value'] = merged_data['value_1'] + merged_data['value_2']

    # 원본 value 열 제거 후 필요한 열만 유지
    merged_data = merged_data[['time', 'value', 'feeder_1', 'tag_1']]

    # 열 이름 정리
    merged_data = merged_data.rename(columns={'feeder_1': 'feeder', 'tag_1': 'tag'})

    # 시간 기준으로 정렬
    merged_data = merged_data.sort_values(by='time').reset_index(drop=True)

    return merged_data

def train_and_save_model(data, model_path='isolation_forest_model.pkl'):
    # Isolation Forest 모델 생성 및 학습
    model = IsolationForest(contamination=0.01, random_state=42)
    model.fit(data[['value']])

    # 학습된 모델을 파일로 저장
    joblib.dump(model, model_path)
    print(f"Model saved at: {model_path}")

def load_model_and_predict(data, model_path='isolation_forest_model.pkl'):
    # 저장된 모델 로드
    model = joblib.load(model_path)
    print(f"Model loaded from: {model_path}")

    # 이상치 예측
    data['outlier'] = model.predict(data[['value']])  # -1: 이상치, 1: 정상치
    
    # 값이 0인 경우도 이상치로 처리
    zero_value_condition = data['value'] == 0
    data.loc[zero_value_condition, 'outlier'] = -1
    
    # 이상치인 시간대 추출
    outlier_times = data[data['outlier'] == -1]['time'].tolist()
    print(f"Outlier times: {outlier_times}")

    return data, outlier_times

def visualize_outliers(data):
    # 시각화 설정
    plt.figure(figsize=(12, 6))

    # 원본 데이터를 검은색 선으로 표시
    plt.plot(data['time'], data['value'], color='black', label='Original', linewidth=1.5)

    # 이상치를 빨간색 점으로 표시
    outlier_data = data[data['outlier'] == -1]
    plt.scatter(outlier_data['time'], outlier_data['value'], color='red', label='Outlier', s=50)

    # 그래프 설정
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Isolation Forest Outlier Detection')
    plt.legend()
    plt.grid(True)
    plt.show()

'''
# 예제 
feed1 = r"C:\Users\김소민\Desktop\이상치탐지\DB\423_FM1_FC1_RWIN.csv"
feed2 = r"C:\Users\김소민\Desktop\이상치탐지\DB\423_FM2_FC1_RWIN.csv"

# 데이터 로드 및 병합
merged_data = load_and_merge_csv(feed1, feed2)

# 모델 학습 및 저장
train_and_save_model(merged_data)

# 저장된 모델 로드 및 이상치 예측, 이상치 시간대 반환
result_data, outlier_times = load_model_and_predict(merged_data)

# 시각화
visualize_outliers(result_data)
'''
