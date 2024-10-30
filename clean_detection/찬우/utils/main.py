import os
import yaml
import logging
from model.arima import OutlierProcessor as ARIMAProcessor
from model.LOF import LOFProcessor
import pandas as pd
import re

# 기본 경로 설정
base_dir = r'C:\Users\ansck\Documents\ybigta\DA_25\project\AD_detection\clean_detection\data'

# 연도-월을 추출하는 함수
def get_year_month_from_filename(filename):
    match = re.search(r'_([0-9]{6})_', filename)
    return match.group(1) if match else None

# 특정 폴더 내의 모든 CSV 파일을 병합하여 수직으로 결합하는 함수
def combine_folder_files():
    for root, dirs, files in os.walk(base_dir):
        folder = os.path.basename(root)
        if files:
            combined_df = pd.DataFrame()  # 각 폴더별로 병합된 데이터를 저장할 DataFrame
            
            # 폴더 내의 모든 CSV 파일을 연도-월 순서로 정렬
            csv_files = sorted(
                [f for f in files if f.endswith('.csv') and get_year_month_from_filename(f) is not None],
                key=lambda x: get_year_month_from_filename(x)
            )
            
            if not csv_files:
                print(f"폴더 '{folder}'가 비어있습니다.")
                continue

            # 모든 CSV 파일을 순서대로 병합 (수직 병합)
            for file_name in csv_files:
                file_path = os.path.join(root, file_name)
                
                # 필요한 컬럼만 읽기
                df = pd.read_csv(file_path, usecols=['time', 'value', 'feeder', 'tag'])
                
                # 병합
                combined_df = pd.concat([combined_df, df], ignore_index=True)

            # 파일 저장 (각 하위 폴더에 저장)
            output_file = os.path.join(base_dir, f'{folder}_combined.csv')
            combined_df.to_csv(output_file, index=False)
            print(f"폴더 '{folder}'의 데이터를 수직으로 병합하여 '{output_file}'에 저장했습니다.")

# 모든 폴더의 병합 파일이 생성된 후 연산을 수행하는 함수
def perform_calculations():
    try:
        # 첫 번째 연산: (433_CG1_ZC1_RWIN - 433_CG2_ZC1_RWIN)
        df1 = pd.read_csv(os.path.join(base_dir, '433_CG1_ZC1_RWIN_combined.csv'), usecols=['time', 'value']).set_index('time')
        df2 = pd.read_csv(os.path.join(base_dir, '433_CG2_ZC1_RWIN_combined.csv'), usecols=['time', 'value']).set_index('time')
        
        if df1.empty or df2.empty:
            print("433_diff_result.csv 계산에 필요한 파일 중 하나가 비어 있습니다.")
            pd.DataFrame(columns=['time', 'calculated_value']).to_csv(os.path.join(base_dir, '433_diff_result.csv'), index=False)
        else:
            diff_df = df1 - df2
            diff_df.reset_index().to_csv(os.path.join(base_dir, '433_diff_result.csv'), index=False)
            print("433_diff_result.csv 파일이 생성되었습니다.")

        # 두 번째 연산: (423_FM1_FC1_RWIN + 423_FM2_FC1_RWIN) / 463_KL1_SC1_RXEX
        df3 = pd.read_csv(os.path.join(base_dir, '423_FM1_FC1_RWIN_combined.csv'), usecols=['time', 'value']).set_index('time')
        df4 = pd.read_csv(os.path.join(base_dir, '423_FM2_FC1_RWIN_combined.csv'), usecols=['time', 'value']).set_index('time')
        df5 = pd.read_csv(os.path.join(base_dir, '463_KL1_SC1_RWIn_combined.csv'), usecols=['time', 'value']).set_index('time')
        
        if df3.empty or df4.empty or df5.empty:
            print("423_FM_division_result.csv 계산에 필요한 파일 중 하나가 비어 있습니다.")
            pd.DataFrame(columns=['time', 'calculated_value']).to_csv(os.path.join(base_dir, '423_FM_division_result.csv'), index=False)
        else:
            division_df = (df3 + df4) / df5
            division_df.reset_index().to_csv(os.path.join(base_dir, '423_FM_division_result.csv'), index=False)
            print("423_FM_division_result.csv 파일이 생성되었습니다.")
    
    except pd.errors.EmptyDataError as e:
        print("빈 파일로 인해 연산을 수행할 수 없습니다:", e)

# 1단계: 각 폴더의 파일을 병합
combine_folder_files()

# 2단계: 병합이 완료된 후 연산을 수행
perform_calculations()










# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# config.yaml 파일 로드 함수
def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config

# Main 실행 함수
def main():
    # config.yaml 파일 로드
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    config = load_config(config_path)

    # Config 파일에서 경로와 파라미터 불러오기
    base_dir = config['base_dir']
    isolation_dir = config['isolation_dir']
    model_type = config['model']  # 사용할 모델 타입 (ARIMA 또는 LOF)

    # 모델 타입 확인 및 프로세서 선택
    if model_type == "LOF":
        # LOF 모델의 파라미터 로드
        n_neighbors_list = config['lof'].get('n_neighbors_list', [20])
        min_clean_length = config['lof'].get('min_clean_length', 30)

        processor = LOFProcessor(
            base_dir=base_dir,
            isolation_dir=isolation_dir,
            n_neighbors_list=n_neighbors_list,
            min_clean_length=min_clean_length
        )
    elif model_type == "ARIMA":
        # ARIMA 모델의 파라미터 로드
        order = config['arima'].get('order', (5, 1, 0))
        min_clean_length = config['arima'].get('min_clean_length', 20)

        processor = ARIMAProcessor(
            base_dir=base_dir,
            isolation_dir=isolation_dir,
            order=order,
            min_clean_length=min_clean_length
        )
    else:
        logging.error("Invalid model type specified in config.yaml. Please use 'ARIMA' or 'LOF'.")
        return

    # data 디렉토리 내의 모든 CSV 파일을 처리
    processor.process_data(config.get("config_path", "config.yaml"))

if __name__ == "__main__":
    main()
