import os
import json
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class OutlierDataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.data = self.prepare_data()

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]

    def load_csv(self) -> pd.DataFrame:
        """
        원본 csv파일 데이터프레임으로 불러오기.
        """
        data_path = self.config['data_path']
        data = pd.read_csv(data_path, usecols=['value'])
        return data
    
    def load_json(self) -> list[list[int]]:
        """
        비교적 이상치의 위험성이 적은 구간에 대한 정보 불러오기.
        """
        data_path = self.config['data_path']
        interval_path = self.config['interval_path']
        dirc_name = data_path.split('/')[1]
        file_name = data_path.split('/')[2]
        with open(interval_path, 'r') as file:
            interval = json.load(file)[dirc_name][file_name]
        return interval
    
    def split_by_interval(self,
                          data: pd.DataFrame,
                          intervals: list[list[int]]) -> list[pd.DataFrame]:
        """
        불러온 구간 정보로 원본 데이터프레임 분할, 분할한 데이터프레임을 리스트로 반환.
        """
        subsets = []
        for start, end in intervals:
            subset = data.iloc[start:end]
            subsets.append(subset)
        return subsets

    def slice_by_window(self, data: list[pd.DataFrame]) -> list[np.ndarray]:
        """
        분할된 데이터프레임별로 주어진 윈도우로 스텝 사이즈 만큼 이동하며 데이터 추출.
        만약에 주어진 구간 안에서 윈도우 설정이 불가능하면 해당 구간을 건너뛴다.
        """
        window_size = self.config['seq_size']
        step_size = self.config['step_size']
        windows = []
        for subset in data:
            values = subset['value'].to_numpy()
            for start in range(0, len(values) - window_size + 1, step_size):
                window = values[start:start + window_size]
                windows.append(window)
        return windows

    def standardize(self, data: list[np.ndarray]) -> torch.Tensor:
        """
        모든 값들을 정규화, [seq_size, batch_size, 1] 차원의 텐서 반환.
        """
        data_array = np.array(data)
        means = data_array.mean()
        stds = data_array.std()
        normalized_data = (data_array - means) / stds
        return torch.tensor(normalized_data.T, dtype=torch.float32).unsqueeze(-1)

    def add_noise(self, data: torch.Tensor) -> torch.Tensor:
        """
        모든 값에 평균을 0, 표준편차를 1로 하는 정규분포로부터 샘플링된 노이즈를 추가한다.
        """
        noise = torch.normal(0, 0.01, data.size())
        return data + noise

    def process_test(self, data: pd.DataFrame) -> list[np.ndarray]:
        """
        테스트용 데이터셋 생성시 원본 데이터프레임을 [-1, seq_size] 형태의 2차원 배열로 반환.
        """
        seq_size = self.config['seq_size']
        values = data['value'].to_numpy()
        return [values[i:i + seq_size] for i in range(len(values) - seq_size + 1)]

    def prepare_data(self) -> torch.Tensor:
        """
        훈련, 테스트 케이스에 따라서 데이터 전처리 진행.
        """
        train = self.config['train']
        batch_size = self.config['batch_size']
        data = self.load_csv()

        if train:
            intervals = self.load_json()
            data = self.split_by_interval(data, intervals)
            data = self.slice_by_window(data)
            data = self.standardize(data)
            data = self.add_noise(data)
        else:
            data = self.process_test(data)
            data = self.standardize(data)
        return data