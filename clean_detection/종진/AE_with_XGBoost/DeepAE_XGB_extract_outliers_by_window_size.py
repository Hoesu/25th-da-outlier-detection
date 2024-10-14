import os
import pandas as pd
from tqdm import tqdm

# DeepAutoEncoder 결과에서 이상치를 추출하는 함수 정의
def extract_outliers(result_directory, output_directory):
    # 결과 저장 폴더 생성
    os.makedirs(output_directory, exist_ok=True)

    # 빈 데이터프레임 생성
    all_outliers_df = pd.DataFrame()

    # result 디렉토리 내 모든 window_size 폴더 탐색
    window_folders = [f for f in os.listdir(result_directory) if f.startswith('window_size_')]

    # 각 window_size 폴더 순회 (tqdm 추가)
    for window_folder in tqdm(window_folders, desc="Processing window size folders", unit="folder"):
        window_size = int(window_folder.split('_')[-1])  # window_size 추출
        window_folder_path = os.path.join(result_directory, window_folder)
        
        # window_size 폴더 내 모든 CSV 파일 탐색 (tqdm 추가)
        csv_files = [f for f in os.listdir(window_folder_path) if f.endswith('.csv')]
        for csv_file in tqdm(csv_files, desc=f"Processing CSV files in {window_folder}", unit="file"):
            csv_path = os.path.join(window_folder_path, csv_file)

            try:
                # CSV 파일 읽기
                df = pd.read_csv(csv_path)
                print(f"[INFO] CSV 파일 읽기 완료: {csv_file}, 데이터 크기: {df.shape}")

                # 이상치 데이터만 필터링
                outliers_df = df[df['outlier'] == True].copy()  # 경고 해결을 위해 .copy() 사용
                if outliers_df.empty:
                    print(f"[INFO] 이상치가 없는 파일: {csv_file}")
                    continue

                # window_size 정보 추가
                outliers_df.loc[:, 'window_size'] = window_size  # .loc[] 사용하여 명시적으로 할당

                # 전체 데이터프레임에 추가
                all_outliers_df = pd.concat([all_outliers_df, outliers_df], ignore_index=True)
                print(f"[INFO] 이상치 데이터프레임에 추가 완료: {csv_file}")

            except Exception as e:
                print(f"[ERROR] {csv_file} 처리 중 오류 발생: {e}")
                continue

    # 변수(tag)별로 분류하기 위해 데이터프레임에서 'tag' 컬럼을 사용해 분류
    if 'tag' in all_outliers_df.columns:
        tags = all_outliers_df['tag'].unique()  # 각 변수(tag) 값 찾기
    else:
        print("[ERROR] 'tag' 열을 찾을 수 없습니다. CSV 파일에 'tag' 열이 포함되어 있는지 확인하세요.")
        return

    # 각 변수별로 데이터를 분류하고 저장
    for tag in tqdm(tags, desc="Saving outliers by tag", unit="tag"):
        tag_df = all_outliers_df[all_outliers_df['tag'] == tag]
        print(f"[INFO] 저장 중: {tag}, 데이터 크기: {tag_df.shape}")

        # 변수별 CSV 파일 저장
        tag_output_path = os.path.join(output_directory, f'{tag}_outliers.csv')
        tag_df.to_csv(tag_output_path, index=False)
        print(f"[INFO] CSV 파일 저장 완료: {tag_output_path}")

    print("[INFO] 모든 이상치 데이터 저장 완료.")
