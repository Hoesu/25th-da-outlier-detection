import os
import yaml
import torch
import logging
import platform
import argparse

import torch
import torch.utils.data
from torch import optim

from utils.model import VAE
from utils.dataset import Dataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config.yaml', help='choose config file')
    return parser.parse_args()

def setup_logging(path):
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(path),
            logging.StreamHandler()
        ]
    )

def load_yaml_config(config_path: str) -> dict:
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

## TODO 1 =============================================================
## 모든 메소드의 입력값, 출력값에 대한 타이핑 추가하기. (model.py 주석 참고.)
## def example_method(number: int,
##                    string: str
##                    data  : dict) -> float:

## 모든 메소드 입력값에 맞춰서 구현해놓기 (깃허브 참고)
## 필요하면 언제든지 수정할 수 있게 주석 상세하게 추가하기.
## 주어진 환경에서 돌아갈 수 있는 코드인지 스스로 확인.

def loss_function(recon_x, x, mu, logvar,lamb, mu_att, logvar_att):
    """
    깃허브 참고
    """
    pass

def get_batches(iterable, batch_size):
    """
    깃허브 참고
    """
    pass

def train(epoch,lambda_kl):
    """
    깃허브 참고
    """
    pass

def test(epoch,lambda_kl):
    """
    깃허브 참고
    """
    pass

def save(config, ...):
    """
    모델 훈련, 테스트 후에는 output 디렉토리 아래에 사용한 데이터셋의
    기기명, 데이터 수집 기간, 피더 번호, 태그명 별로 디렉토리를 생성합니다.
    모델 훈련 후에는 가중치 체크포인트 파일과 사용한 config 옵션을 저장합니다.
    모델 테스트 후에는 이상치 값들과 시각화 이미지를 저장합니다.
    """
    pass
## =================================================================


if __name__ == '__main__':
    ## Set up Argument Parser
    args = parse_args()

    ## Set up Logger
    ## TODO 2 ==========================================================
    ## log_file_path를 save 메소드에서 사용할 디렉토리로 올바르게 배정해줘야겠죠?
    ## =================================================================
    log_file_path = '...'
    setup_logging(log_file_path)

    ## Set Device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        logging.info(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif platform.system() == "Darwin" and torch.backends.mps.is_available():
        device = torch.device("mps")
        logging.info("Using MPS GPU on Apple Silicon.")
    else:
        device = torch.device("cpu")
        logging.info("No GPU available, using CPU instead.")

    ## Load config.yaml
    config = load_yaml_config(args.config)
    logging.info("Configuration file loaded successfully.")

    ## Set train/test mode
    train = config['train']
    if train:
        logging.info("Train mode.")
    else:
        logging.info("Test mode.")

    ## Assign hyperparameters from config
    optimizer_choice = config['optimizer_choice']
    learning_rate = config['learning_rate']
    logging.info(f"Optimizer choice: {optimizer_choice}, Learning rate: {learning_rate}")

    ## load dataset from config
    dataset = Dataset(config)
    logging.info("Dataset loaded successfully.")

    ## load model from config
    model = VAE(config)
    model.to(device)
    logging.info("Model loaded and moved to the appropriate device.")

    ## set optimizer from config
    if optimizer_choice == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        logging.info("Using AdamW optimizer.")
    elif optimizer_choice == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        logging.info("Using SGD optimizer.")
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        logging.info("Using Adam optimizer.")
    assert optimizer_choice in ['AdamW', 'SGD', 'Adam']

    ## TODO 3 ==========================================================
    ## 위에 메소드 작성 끝나면 그거 사용해서 알아서 채워넣기 ㅋ
    ## =================================================================