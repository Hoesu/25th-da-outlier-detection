import os
import pandas as pd
from PIL import Image
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class pngMerger:
    def __init__(self):
        """
        초기화 메서드: base_dir과 output_dir을 자동으로 탐색하여 설정합니다.
        """
        self.base_dir = os.path.abspath(os.path.join(os.getcwd(), '..', 'output'))  # CSV와 PNG 파일이 있는 상위 폴더
        self.output_dir = os.path.abspath(os.path.join(os.getcwd(), '..', 'merged_output1'))  # 병합된 파일을 저장할 폴더

        # 결과 저장 폴더가 존재하지 않으면 생성
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def merge_png_files_based_on_csv(self, mode='horizontal'):
        """
        각 하위 폴더의 CSV 파일을 읽어 PNG 파일들을 병합하여 새로운 PNG 파일을 생성하는 메서드.
        :param mode: 'horizontal' (수평 결합) 또는 'vertical' (수직 결합)
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
                        csv_files = [f for f in os.listdir(sub_folder_path) if f.endswith('.csv')]
                        
                        if csv_files:
                            logging.info(f"Processing folder: {sub_folder_path}")
                            
                            for csv_file in csv_files:
                                csv_file_path = os.path.join(sub_folder_path, csv_file)
                                try:
                                    # CSV 파일 읽기
                                    df = pd.read_csv(csv_file_path)
                                    time_series = df['time'].sort_values().unique()  # time 컬럼 기준으로 정렬된 시계열 데이터
                                    png_files = [f for f in os.listdir(sub_folder_path) if f.endswith('.png')]

                                    images = []
                                    # PNG 파일 병합 (time 시계열에 맞춰 가로로 병합)
                                    for png_file in png_files:
                                        file_path = os.path.join(sub_folder_path, png_file)
                                        img = Image.open(file_path)
                                        images.append(img)
                                        logging.info(f"Image loaded: {file_path}")

                                    if images:
                                        merged_image = self.concat_images(images, mode=mode)

                                        # 병합된 PNG 저장
                                        merged_file_output_path = os.path.join(merged_sub_folder_path, f"{sub_folder_name}_merged.png")
                                        merged_image.save(merged_file_output_path)
                                        logging.info(f"Merged PNG saved: {merged_file_output_path}")
                                except Exception as e:
                                    logging.error(f"Error processing CSV or PNG files in {sub_folder_path}: {e}")

    def concat_images(self, images, mode='horizontal'):
        """
        이미지 리스트를 결합하여 하나의 이미지로 반환하는 메서드.
        :param images: 결합할 이미지 객체들의 리스트
        :param mode: 'horizontal' (수평 결합) 또는 'vertical' (수직 결합)
        :return: 결합된 이미지 객체
        """
        widths, heights = zip(*(i.size for i in images))
        
        if mode == 'horizontal':
            total_width = sum(widths)
            max_height = max(heights)
            new_image = Image.new('RGB', (total_width, max_height))
            x_offset = 0
            for img in images:
                new_image.paste(img, (x_offset, 0))
                x_offset += img.size[0]
        elif mode == 'vertical':
            max_width = max(widths)
            total_height = sum(heights)
            new_image = Image.new('RGB', (max_width, total_height))
            y_offset = 0
            for img in images:
                new_image.paste(img, (0, y_offset))
                y_offset += img.size[1]
        else:
            raise ValueError("Mode should be 'horizontal' or 'vertical'")
        
        return new_image

# pngMerger 클래스를 사용하여 파일 병합 실행
if __name__ == "__main__":
    png_merger = pngMerger()  # base_dir과 output_dir을 클래스에서 자동 설정
    png_merger.merge_png_files_based_on_csv(mode='horizontal')  # 수평으로 결합하려면 'horizontal', 수직으로 결합하려면 'vertical'
