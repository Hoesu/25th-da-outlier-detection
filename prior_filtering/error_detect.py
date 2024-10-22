import pandas as pd
import numpy as np

def error_detect(csv_file, threshold_factor_doubtful=3, threshold_factor_frozen=0.5):
    
    # CSV 파일 읽기
    data = pd.read_csv(csv_file)
    
    # 'value' 열이 있는지 확인
    if 'value' not in data.columns:
        raise ValueError("CSV 파일에 'value' 열이 없습니다.")
    
    # 'value' 열에서 기울기(gradient) 계산
    data['gradient'] = np.gradient(data['value'])
    
    # 상태 초기화
    data['status'] = 'Normal'
    
    # Frozen 상태 분류
    frozen_condition = data['gradient'] == 0
    data.loc[frozen_condition, 'status'] = 'Frozen'
    
    # 2. Hole 상태 정의: 값이 0인 경우
    hole_condition = data['value'] == 0
    data.loc[hole_condition, 'status'] = 'Hole'
    data = data.drop(columns=['gradient'])
    return data
