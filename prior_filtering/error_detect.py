import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

def error_detect_interpolate(csv_file):
    # CSV 파일 읽기
    data = pd.read_csv(csv_file)

    # 필수 열 확인
    if 'value' not in data.columns or 'time' not in data.columns:
        raise ValueError("CSV 파일에 'value'와 'time' 열이 필요합니다.")

    # 기울기 계산 및 상태 초기화
    data['gradient'] = np.gradient(data['value'])
    data['status'] = 'Normal'

    # Frozen 및 Hole 상태 분류
    frozen_condition = data['gradient'] == 0
    hole_condition = data['value'] == 0
    data.loc[frozen_condition, 'status'] = 'Frozen'
    data.loc[hole_condition, 'status'] = 'Hole'

    # Frozen과 Hole의 합이 과반수인지 체크
    frozen_hole_count = data[frozen_condition | hole_condition].shape[0]
    total_count = data.shape[0]

    if frozen_hole_count > total_count / 2:
        # Frozen과 Hole 데이터가 과반수일 경우 time 열의 값을 리스트로 반환
        frozen_hole_times = data.loc[frozen_condition | hole_condition, 'time'].tolist()
        print("Frozen과 Hole 상태가 과반수를 초과하여 원본 데이터를 반환합니다.")
        return data.drop(columns=['gradient', 'status'])

    # 과반수가 아닌 경우: NaN 처리 및 스플라인 보간 수행
    data.loc[frozen_condition | hole_condition, 'value'] = np.nan

    # 유효한 값의 인덱스와 값 추출
    valid_idx = data[data['value'].notna()].index
    valid_values = data['value'].dropna().values

    # 스플라인 보간 적용
    spline = CubicSpline(valid_idx, valid_values)
    interpolated_values = pd.Series(spline(data.index), index=data.index)

    # 보간된 값 대입
    data['interpolated_value'] = data['value'].combine_first(interpolated_values)

    # 시각화: 원본 값과 보간된 값 비교
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['value'], 'o', label='Original (with NaN)', markersize=5, alpha=0.7)
    plt.plot(data.index, data['interpolated_value'], '-', label='Interpolated (Cubic Spline)', linewidth=2)
    plt.title('Comparison of Original and Interpolated Values')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 불필요한 열 삭제
    data = data.drop(columns=['gradient', 'status'])

    # 보간된 DataFrame 반환
    return data
'''
#활용 예재
csv_file = r"C:\Users\김소민\Desktop\이상치탐지\DB1\DB\423_FM1_FC1_RWIN\KFEMS.HALLA.01.DC01.IP001_202211_feeder_00_tag_IO_5SEC_MIX_423_FM1_FC1_RWIN.csv"
error_detect_and_spline_interpolate_with_condition(csv_file)
'''
