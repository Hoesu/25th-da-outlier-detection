import os
import pandas as pd
import re
import logging
from model.isolation_forest import OutlierProcessor as IsolationOutlierProcessor
from model.LOF import OutlierProcessor as LOFOutlierProcessor
from model.arima import OutlierProcessor as ARIMAOutlierProcessor
from model.knn import OutlierProcessor as KNNOutlierProcessor
from model.merge import pngMerger, csvMerger ,JSONMerger
# 병합을 위한 클래스 임포트
# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 실행 부분
if __name__ == '__main__':
    # 모델 선택
    model_choice = input("Choose a model (LOF, IsolationForest, ARIMA, KNN, all): ").strip().lower()

    # base_dir을 현재 디렉토리의 상위 디렉토리로 설정 후 data 폴더로 설정
    base_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))  # 상위 폴더로 이동
    base_dir = os.path.join(base_dir, 'data')  # data 폴더로 경로 설정

    # 결과 저장 폴더 설정
    output_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))  # 상위 폴더로 이동
    output_dir = os.path.join(output_dir, 'output')  # output 폴더로 경로 설정

    start_month = '202205'
    end_month = '202306'
    months = pd.period_range(start=start_month, end=end_month, freq='M').strftime('%Y%m')
    months = [month for month in months if month != '202304']

    folder_list = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]

    # 1. LOF 모델 실행
    if model_choice == 'lof' or model_choice == 'all':
        n_neighbors_list = [50, 30, 70]  # 하이퍼파라미터 리스트
        for n_neighbors in n_neighbors_list:
            isolation_dir = os.path.join(output_dir, f'lof_n_neighbors_{n_neighbors}')  # 하이퍼파라미터별 폴더 생성
            if not os.path.exists(isolation_dir):
                os.makedirs(isolation_dir)

            processor = LOFOutlierProcessor(base_dir=base_dir, isolation_dir=isolation_dir, n_neighbors_list=[n_neighbors])
            processor.process_data(folder_list, months)

            # LOF JSON 병합
            merger = JSONMerger(model=f'lof_n_neighbors_{n_neighbors}')
            merger.merge_json_files()

    # 2. IsolationForest 모델 실행
    if model_choice == 'isolationforest' or model_choice == 'all':
        contamination_list = [0.005, 0.01, 0.05]  # 하이퍼파라미터 리스트
        for contamination in contamination_list:
            isolation_dir = os.path.join(output_dir, f'isolation_contamination_{contamination}')  # 하이퍼파라미터별 폴더 생성
            if not os.path.exists(isolation_dir):
                os.makedirs(isolation_dir)

            processor = IsolationOutlierProcessor(base_dir=base_dir, isolation_dir=isolation_dir, n_estimators=200, max_samples=0.8, contamination_list=[contamination])
            processor.process_data(folder_list, months)

            # IsolationForest JSON 병합
            merger = JSONMerger(model=f'isolation_contamination_{contamination}')
            merger.merge_json_files()

    # 3. ARIMA 모델 실행
    if model_choice == 'arima' or model_choice == 'all':
        arima_orders = [(1, 1, 1), (2, 1, 2), (5, 1, 0)]  # ARIMA 하이퍼파라미터 조합 리스트
        for order in arima_orders:
            isolation_dir = os.path.join(output_dir, f'arima_order_{order}')  # 하이퍼파라미터 조합별 결과 저장 폴더 생성
            if not os.path.exists(isolation_dir):
                os.makedirs(isolation_dir)

            processor = ARIMAOutlierProcessor(base_dir=base_dir, isolation_dir=isolation_dir, arima_order=order)
            processor.process_data(folder_list, months)

            # ARIMA JSON 병합
            merger = JSONMerger(model=f'arima_order_{order}')
            merger.merge_json_files()

    # 4. KNN 모델 실행
    if model_choice == 'knn' or model_choice == 'all':
        n_neighbors_list = [30, 50, 70]  # 하이퍼파라미터 리스트
        for n_neighbors in n_neighbors_list:
            isolation_dir = os.path.join(output_dir, f'knn_n_neighbors_{n_neighbors}')  # 하이퍼파라미터별 폴더 생성
            if not os.path.exists(isolation_dir):
                os.makedirs(isolation_dir)

            processor = KNNOutlierProcessor(base_dir=base_dir, isolation_dir=isolation_dir, n_neighbors_list=[n_neighbors])
            processor.process_data(folder_list, months)

            # KNN JSON 병합
            merger = JSONMerger(model=f'knn_n_neighbors_{n_neighbors}')
            merger.merge_json_files()

    # 실행 완료 메시지
    print(f"{model_choice.upper()} 모델 실행이 완료되었습니다.")
    
 
