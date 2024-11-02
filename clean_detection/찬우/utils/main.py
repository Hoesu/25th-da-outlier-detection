import os
import yaml
import logging
from model.arima import OutlierProcessor as ARIMAProcessor
from model.LOF import LOFProcessor

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
        n_neighbors_list = config['lof']['n_neighbors_list']
        min_clean_length = config['lof']['min_clean_length']

        processor = LOFProcessor(
            base_dir=base_dir,
            isolation_dir=isolation_dir,
            n_neighbors_list=n_neighbors_list,
            min_clean_length=min_clean_length
        )
    elif model_type == "ARIMA":
        # ARIMA 모델의 파라미터 로드
        min_clean_length = config['arima']['min_clean_length']

        processor = ARIMAProcessor(
            base_dir=base_dir,
            isolation_dir=isolation_dir,
            min_clean_length=min_clean_length
        )
    else:
        logging.error("Invalid model type specified in config.yaml. Please use 'ARIMA' or 'LOF'.")
        return

    # data 디렉토리 내의 모든 CSV 파일을 처리
    processor.process_data()

if __name__ == "__main__":
    main()
