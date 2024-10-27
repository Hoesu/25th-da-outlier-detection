import os
import pandas as pd
import re

# 기본 경로 설정
base_dir = r'C:\Users\ansck\Documents\ybigta\DA_25\project\AD_detection\clean_detection\data'
folders = ['423_FM1_FC1_RWIN', '423_FM2_FC1_RWIN']  # 대상 폴더

# 연도-월을 추출하는 함수
def get_year_month_from_filename(filename):
    match = re.search(r'_([0-9]{6})_', filename)
    return match.group(1) if match else None

# 폴더별 연도-월별 파일을 저장할 딕셔너리 생성
folder_files = {folder: {} for folder in folders}

# 폴더에서 파일을 가져와서 연도-월별로 저장
for folder in folders:
    folder_path = os.path.join(base_dir, folder)
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            year_month = get_year_month_from_filename(file_name)
            if year_month:
                folder_files[folder][year_month] = os.path.join(folder_path, file_name)

# 연도-월이 같은 파일을 찾아 value 합산
combined_df = pd.DataFrame()
for year_month in folder_files[folders[0]].keys():
    if year_month in folder_files[folders[1]]:
        file1_path = folder_files[folders[0]][year_month]
        file2_path = folder_files[folders[1]][year_month]

        # 두 파일 로드
        df1 = pd.read_csv(file1_path)
        df2 = pd.read_csv(file2_path)
        
        # `time`을 기준으로 `value` 컬럼을 합산
        df1 = df1[['time', 'value']].rename(columns={'value': 'value1'})
        df2 = df2[['time', 'value']].rename(columns={'value': 'value2'})
        merged_df = pd.merge(df1, df2, on='time')
        merged_df['combined_value'] = merged_df['value1'] + merged_df['value2']
        
        # 합산된 결과를 combined_df에 추가
        combined_df = pd.concat([combined_df, merged_df[['time', 'combined_value']]], ignore_index=True)

# 밀리초를 datetime으로 변환
combined_df['datetime'] = pd.to_datetime(combined_df['time'], unit='ms')

# 이상치 라벨링: 350에서 380 사이가 정상, 그 외는 이상치
combined_df['is_outlier'] = combined_df['combined_value'].apply(lambda x: 0 if 350 <= x <= 380 else 1)

# 결과 확인
print(combined_df[['datetime', 'time', 'combined_value', 'is_outlier']].head())

# 결과를 CSV로 저장
output_file = os.path.join(base_dir, 'combined_df.csv')
combined_df.to_csv(output_file, index=False)
print(f"CSV 파일로 저장되었습니다: {output_file}")
