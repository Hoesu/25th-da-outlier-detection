import os
import yaml
import pandas as pd
import logging
from model import LOFProcessor, OutlierProcessor  # LOF와 ARIMA 프로세서 임포트

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# config.yaml 파일 로드 함수
def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as file:  # UTF-8로 파일 읽기
        config = yaml.safe_load(file)
    return config

# Main 실행 함수
def main():
    # 현재 파일의 디렉토리에서 config.yaml 파일 로드
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    config = load_config(config_path)

    # Config 파일에서 경로와 파라미터 불러오기
    base_dir = config['base_dir']
    isolation_dir = config['isolation_dir']
    combined_outliers_file = config['combined_outliers_file']
    interpolate_method = config['preheater']['interpolate_method']  # Preheater 옵션
    model_type = config['model']  # 사용할 모델 타입 (LOF 또는 ARIMA)

    # 날짜 범위 설정
    start_month = config['date_range']['start_month']
    end_month = config['date_range']['end_month']
    exclude_months = set(config['date_range']['exclude_months'])
    months = pd.period_range(start=start_month, end=end_month, freq='M').strftime('%Y%m')
    months = [month for month in months if month not in exclude_months]

    # 모델 타입에 따라 프로세서를 생성
    # 모델 타입에 따라 프로세서를 생성
    if model_type == "LOF":
        n_neighbors_list = config['lof']['n_neighbors_list']
        processor = LOFProcessor(
            base_dir=base_dir,
            isolation_dir=isolation_dir,
            n_neighbors_list=n_neighbors_list,
            combined_outliers_file=combined_outliers_file,  # LOFProcessor에서는 combined_outliers_file을 사용함
            interpolate_method=interpolate_method
        )
    elif model_type == "ARIMA":
        arima_order = tuple(config['arima']['order'])
        processor = OutlierProcessor(
            base_dir=base_dir,
            isolation_dir=isolation_dir,
            arima_order=arima_order,
            interpolate_method=interpolate_method  # ARIMA 모델에서는 combined_outliers_file 제거
        )
    else:
        logging.error("Invalid model type specified in config.yaml. Choose either 'LOF' or 'ARIMA'.")
        return
    

    # 폴더 내 모든 파일을 처리
    folder_list = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
    processor.process_data(folder_list, months)

if __name__ == "__main__":
    main()
