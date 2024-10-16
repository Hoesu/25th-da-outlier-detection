import os
import json
import logging
import os
from PIL import Image
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class pngMerger:
    def __init__(self):
        """
        초기화 메서드: base_dir과 output_dir을 자동으로 탐색하여 설정합니다.
        """
        self.base_dir = os.path.abspath(os.path.join(os.getcwd(), '..', 'output'))  # PNG 파일이 있는 상위 폴더
        self.output_dir = os.path.abspath(os.path.join(os.getcwd(), '..', 'merged_output'))  # 병합된 파일을 저장할 폴더

        # 결과 저장 폴더가 존재하지 않으면 생성
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def merge_png_files(self, mode='horizontal'):
        """
        각 하위 폴더에서 PNG 파일들을 병합하여 새로운 PNG 파일을 생성하는 메서드.
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
                        output_file_path = os.path.join(merged_sub_folder_path, f"{sub_folder_name}_merged.png")
                        
                        # 이미 병합된 파일이 존재하면 패스
                        if os.path.exists(output_file_path):
                            logging.info(f"File already exists, skipping: {output_file_path}")
                            continue
                        
                        images = []

                        # PNG 파일 병합
                        for file_name in os.listdir(sub_folder_path):
                            if file_name.endswith('.png'):
                                file_path = os.path.join(sub_folder_path, file_name)
                                try:
                                    img = Image.open(file_path)
                                    images.append(img)
                                    logging.info(f"Image loaded: {file_path}")
                                except Exception as e:
                                    logging.error(f"Error processing file {file_path}: {e}")
                        
                        # 병합된 PNG 저장
                        if images:
                            merged_image = self.concat_images(images, mode=mode)
                            merged_image.save(output_file_path)
                            logging.info(f"Merged PNG saved: {output_file_path}")
                        else:
                            logging.info(f"No PNG files to merge in folder: {sub_folder_name}")

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
    png_merger.merge_png_files(mode='horizontal')  # 수평으로 결합하려면 'horizontal', 수직으로 결합하려면 'vertical'

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

class JSONMerger:
    def __init__(self, model='isolation', contamination_rates=None, n_neighbors_list=None):
        """
        JSON 병합기를 초기화합니다.
        
        :param model: 사용할 모델 이름 ('isolation', 'lof', 'arima' 등)
        :param contamination_rates: IsolationForest와 같은 모델에서 사용할 contamination rate 리스트
        :param n_neighbors_list: LOF 모델에서 사용할 n_neighbors 리스트
        """
        if contamination_rates is None:
            contamination_rates = ['0.01', '0.05', '0.005']
        
        if n_neighbors_list is None:
            n_neighbors_list = [30, 50, 70]

        self.model = model
        self.contamination_rates = contamination_rates
        self.n_neighbors_list = n_neighbors_list
        self.base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model)

    def merge_json_files(self):
        if self.model == 'isolation':
            self._merge_contamination_json_files()
        elif self.model == 'lof':
            self._merge_lof_json_files()
        elif self.model == 'arima':
            self._merge_arima_json_files()
        else:
            logging.error("지원하지 않는 모델입니다.")
    
    def _merge_contamination_json_files(self):
        """
        IsolationForest 모델의 contamination rate에 따른 JSON 병합.
        """
        for rate in self.contamination_rates:
            contamination_path = os.path.join(self.base_path, f'contamination_{rate}')
            self._merge_files(contamination_path, rate)

    def _merge_lof_json_files(self):
        """
        LOF 모델의 n_neighbors 값에 따른 JSON 병합.
        """
        for n_neighbors in self.n_neighbors_list:
            lof_path = os.path.join(self.base_path, f'n_neighbors_{n_neighbors}')
            self._merge_files(lof_path, n_neighbors)

    def _merge_arima_json_files(self):
        """
        ARIMA 모델의 JSON 병합.
        """
        arima_path = self.base_path
        self._merge_files(arima_path, 'arima')

    def _merge_files(self, folder_path, rate_or_neighbors):
        """
        공통된 JSON 병합 로직.
        
        :param folder_path: JSON 파일이 저장된 폴더 경로
        :param rate_or_neighbors: 해당 모델의 contamination rate 또는 n_neighbors 값
        """
        merged_data = {}
        for folder in os.listdir(folder_path):
            folder_path_inner = os.path.join(folder_path, folder)

            if os.path.isdir(folder_path_inner):
                for file_name in os.listdir(folder_path_inner):
                    if file_name.endswith('.json'):
                        file_path = os.path.join(folder_path_inner, file_name)

                        with open(file_path, 'r') as json_file:
                            data = json.load(json_file)

                            for key, value in data.items():
                                if key not in merged_data:
                                    merged_data[key] = value
                                else:
                                    merged_data[key].update(value)

        output_file = os.path.join(self.base_path, f'{self.model}_interval_{rate_or_neighbors}.json')
        with open(output_file, 'w') as out_file:
            json.dump(merged_data, out_file, indent=4)

        logging.info(f'{self.model}_interval_{rate_or_neighbors}.json 파일이 생성되었습니다.')

# csvMerger 클래스를 사용하여 파일 병합 실행
if __name__ == "__main__":
    csv_merger = csvMerger()  # base_dir과 output_dir을 클래스에서 자동 설정
    csv_merger.merge_csv_files()