import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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


def visualize_results(data, num_parts=5):
    # 데이터 길이에 따라 나눌 크기 계산
    part_size = len(data) // num_parts

    # 각 부분별로 그래프를 그리기
    for i in range(num_parts):
        start = i * part_size
        end = (i + 1) * part_size if i < num_parts - 1 else len(data)
        subset = data.iloc[start:end]

        plt.figure(figsize=(30, 10))  # 그래프 크기 설정

        # 1. 전체 데이터를 선으로 표시 (해당 구간)
        plt.plot(subset.index, subset['value'], color='blue', linewidth=2, label='Value (All)')

        # 2. Frozen과 Hole 상태를 점으로 덧씌우기
        color_map = {'Frozen': 'red', 'Hole': 'green'}
        for status, color in color_map.items():
            status_subset = subset[subset['status'] == status]
            plt.scatter(status_subset.index, status_subset['value'], 
                        color=color, label=status, s=50)

        # 그래프 설정
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.title(f'Error Detection Results (Part {i + 1}/{num_parts})')
        plt.legend()
        plt.grid(True)
        plt.show()

#사용 예제
'''
csv_file = r"C:\Users\김소민\Desktop\이상치탐지\DB\443_SK1_UZ1.csv"
data = error_detect(csv_file)
visualize_results(data)
'''
