import os
import pandas as pd
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class csvMerger:
    def __init__(self):
        """
        초기화 메서드: base_dir과 output_dir을 자동으로 탐색하여 설정합니다.
        """
        self.base_dir = os.path.abspath(os.path.join(os.getcwd(), '..', 'output'))  # CSV 파일이 있는 상위 폴더
        self.output_dir = os.path.abspath(os.path.join(os.getcwd(), '..', 'merged_output_csv'))  # 병합된 파일을 저장할 폴더

        # 결과 저장 폴더가 존재하지 않으면 생성
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def merge_csv_files(self):
        """
        각 하위 폴더에서 CSV 파일들을 병합하여 새로운 파일을 생성하는 메서드.
        """
        for folder_name in os.listdir(self.base_dir):
            folder_path = os.path.join(self.base_dir, folder_name)
            
            # 상위 폴더 탐색
            if os.path.isdir(folder_path):
                # 병합 결과를 저장할 하위 폴더 생성
                merged_sub_folder_path = os.path.join(self.output_dir, folder_name)
                if not os.path.exists(merged_sub_folder_path):
                    os.makedirs(merged_sub_folder_path)
                    
                # 하위 폴더 처리
                for sub_folder_name in os.listdir(folder_path):  # 각 하위 폴더 처리
                    sub_folder_path = os.path.join(folder_path, sub_folder_name)
                    if os.path.isdir(sub_folder_path):
                        output_file_path = os.path.join(merged_sub_folder_path, f"{sub_folder_name}_merged.csv")
                        
                        # 이미 병합된 파일이 존재하면 패스
                        if os.path.exists(output_file_path):
                            logging.info(f"File already exists, skipping: {output_file_path}")
                            continue
                        
                        merged_data = pd.DataFrame()

                        # CSV 파일 병합
                        for file_name in os.listdir(sub_folder_path):
                            if file_name.endswith('.csv'):
                                file_path = os.path.join(sub_folder_path, file_name)
                                try:
                                    df = pd.read_csv(file_path)
                                    merged_data = pd.concat([merged_data, df], ignore_index=True)
                                    logging.info(f"File merged: {file_path}")
                                except Exception as e:
                                    logging.error(f"Error processing file {file_path}: {e}")
                        
                        # 병합된 CSV 파일을 해당 하위 폴더에 저장
                        if not merged_data.empty:
                            merged_data.to_csv(output_file_path, index=False)
                            logging.info(f"Merged CSV saved: {output_file_path}")
                        else:
                            logging.info(f"No CSV files to merge in folder: {sub_folder_name}")

# csvMerger 클래스를 사용하여 파일 병합 실행
if __name__ == "__main__":
    csv_merger = csvMerger()  # base_dir과 output_dir을 클래스에서 자동 설정
    csv_merger.merge_csv_files()
