import os
import yaml
import logging
import platform
import argparse
import numpy as np
from tqdm import tqdm
from functools import partial
import matplotlib.pyplot as plt
plt.set_loglevel('WARNING')

import torch
import torch.utils.data
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from model import VAE
from dataset import OutlierDataset

def parse_args():
    """
    터미널 커맨드 라인으로부터 받아올 인자를 설정합니다.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config.yaml', help='choose config file')
    return parser.parse_args()


def setup_logging(path):
    """
    - 로깅 과정을 설정합니다.
    """
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(path),
            logging.StreamHandler()
        ]
    )


def collate_train(batch):
    """
    - [seq_num, seq_size, 1] 차원의 전체 데이터셋으로부터
    - [seq_size, batch_size, 1] 차원 단위의 입력을 받아오기 위해
    - 데이터로더의 collate_fn 인자로 넘겨줄 메소드입니다.
    """
    batch = torch.stack(batch).permute(1, 0, 2)
    return batch


def collate_inference(batch, additional_param=None):
    """
    - [seq_num, seq_size, 1] 차원의 전체 데이터셋으로부터
    - [seq_size, batch_size, 1] 차원 단위의 입력을 받아오기 위해
    - 데이터로더의 collate_fn 인자로 넘겨줄 메소드입니다.
    - 인퍼런스 단계에서 누락되는 정보가 발생하지 않도록 제로패딩을 추가합니다.
    """
    batch = torch.stack(batch).permute(1, 0, 2)
    max_batch_size = additional_param['batch_size']
    padded_batch = torch.zeros(batch.shape[0], max_batch_size, 1)
    padded_batch[:, :batch.shape[1], :] = batch
    return padded_batch


def loss_function(config: dict,
                  recon_x: torch.Tensor,
                  x: torch.Tensor,
                  mu: torch.Tensor,
                  logvar: torch.Tensor,
                  lamb: torch.Tensor,
                  mu_att: torch.Tensor,
                  logvar_att: torch.Tensor) -> torch.Tensor:
    """
    - 손실 함수를 계산합니다.
    - `recon_x`: 모델이 재구성한 입력 데이터
    - `x`: 원래 입력 데이터
    - `mu`: 잠재 변수의 평균값
    - `logvar`: 잠재 변수의 로그 분산값
    - `lamb`: KL divergence에 대한 가중치
    - `mu_att`, `logvar_att`: 어텐션 잠재 변수의 평균과 로그 분산
    - `config['eta']`: 어텐션 부분의 KL divergence 가중치
    """
    CE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = lamb*(-0.5) * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD_attention = lamb*config['eta']*(-0.5) * torch.sum(1 + logvar_att - mu_att.pow(2) - logvar_att.exp())
    return CE + KLD + KLD_attention


def train(lambda_kl: float,
          model: VAE,
          dataloader: DataLoader,
          device: torch.device,
          optimizer: optim.Optimizer) -> float:
    """
    - 주어진 테이터셋 한 사이클에 대해 모델을 학습시킵니다.
    - `lambda_kl`: KL divergence에 대한 가중치
    - `model`: VAE 모델
    - `dataloader`: 학습 데이터 로더
    - `device`: 모델과 데이터를 학습시킬 장치
    - `optimizer`: 최적화 알고리즘
    - 반환 값: 평균 학습 손실
    """
    model.train()
    train_loss = 0
    for trainbatch in dataloader:
        trainbatch = trainbatch.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar, _, _, _, _, _, c_t_mus, c_t_logvars = model(trainbatch)
        loss = loss_function(config, recon_batch, trainbatch, mu, logvar, lambda_kl, c_t_mus, c_t_logvars)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    return train_loss / len(dataloader)
    

def test(lambda_kl: float,
         model: VAE,
         dataloader: DataLoader,
         device: torch.device) -> float:
    """
    - 모델을 평가합니다.
    - `lambda_kl`: KL divergence에 대한 가중치
    - `model`: VAE 모델
    - `dataloader`: 테스트 데이터 로더
    - `device`: 모델과 데이터를 테스트할 장치 (CPU 또는 GPU)
    - 반환 값: 평균 테스트 손실
    """
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for testbatch in dataloader:
            testbatch = testbatch.to(device)
            recon_batch, mu, logvar, _, _, _, _, _, c_t_mus, c_t_logvars, _, _ = model(testbatch)
            loss = loss_function(config, recon_batch, testbatch, mu, logvar, lambda_kl, c_t_mus, c_t_logvars)
            test_loss += loss.item()
    return test_loss / len(dataloader)


def recon_probability(config: dict,
                      data_batch: torch.Tensor,
                      output_mu_all: torch.Tensor,
                      output_logvar_all: torch.Tensor) -> torch.Tensor:
    """
    - 배치별로 reconstruction probability를 계산합니다.
    """
    sample_reps = config['sample_reps']
    seq_size = config['seq_size']
    batch_size = config['batch_size']
    feature_num = config['input_size']
    probabilities = torch.empty(sample_reps, seq_size, batch_size, feature_num)

    for rep in range(sample_reps):
        mu = output_mu_all[rep, :, :, :]
        var = torch.exp(output_logvar_all[rep, :, :, :])

        prob = (1 / torch.sqrt(2 * torch.pi * var)) * torch.exp(-0.5 * ((data_batch - mu) ** 2) / var)
        probabilities[rep] = prob

    mean_probabilities = probabilities.mean(dim=0)
    return mean_probabilities


def get_anomaly_intervals(anomaly_indices):
    """
    """
    intervals = []
    if len(anomaly_indices) == 0:
        return intervals
    
    start = anomaly_indices[0]
    for i in range(1, len(anomaly_indices)):
        if anomaly_indices[i] != anomaly_indices[i - 1] + 1:
            intervals.append((start, anomaly_indices[i - 1]))
            start = anomaly_indices[i]
    intervals.append((start, anomaly_indices[-1]))
    return intervals


if __name__ == '__main__':

    ## 파서로부터 config 파일의 경로를 받아옵니다.
    args = parse_args()


    ## 사용자 설정을 사용학 위해 config.yaml파일을 딕셔너리로 불러옵니다.
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)


    ## 결과물 저장용 디렉토리를 생성하고, 로깅을 시작합니다.
    data_path = config['data_path']
    dirc_name = data_path.split('/')[1]
    file_name = data_path.split('/')[2][:-4]
    output_dirc = os.path.join('./output', dirc_name, file_name)
    if config['train']:
        output_dirc = os.path.join(output_dirc, 'training_results')
    else:
        checkpoint_path = os.path.join(output_dirc, 'training_results/model_best.pt')
        output_dirc = os.path.join(output_dirc, 'inference_results')
    if not os.path.exists(output_dirc):
        os.makedirs(output_dirc)
    log_file_path = os.path.join(output_dirc, 'logging.log')
    setup_logging(log_file_path)


    ## CUDA GPU, MPS, CPU 중에서 사용가능한 디바이스를 설정합니다.
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        logging.info(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif platform.system() == "Darwin" and torch.backends.mps.is_available():
        device = torch.device("mps")
        logging.info("Using MPS GPU on Apple Silicon.")
    else:
        device = torch.device("cpu")
        logging.info("No GPU available, using CPU instead.")


    ## 훈련/인퍼런스 모드에 맞춰 로그 내역을 기록합니다.
    if config['train']:
        logging.info("Train mode.")
    else:
        logging.info("Test mode.")


    ## 데이터셋을 불러오고 데이터로더 객체를 알맞게 생성해줍니다.
    ## 훈련 모드: split_ratio에 맞춰 전체 데이터셋을 분할하여 훈련/검증 데이터로더를 생성합니다.
    ## 인퍼런스 모드: 전체 데이터셋으로 인퍼런스 데이터로더를 생성합니다.
    dataset = OutlierDataset(config)
    if config['train']:
        dataset_size = len(dataset)
        train_size = int(dataset_size * config['split_ratio'])
        validation_size = dataset_size - train_size
        train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=config['batch_size'],
                                      shuffle=True,
                                      drop_last=True,
                                      num_workers=4,
                                      collate_fn=collate_train)
        val_dataloader = DataLoader(validation_dataset,
                                    batch_size=config['batch_size'],
                                    shuffle=False,
                                    drop_last=True,
                                    num_workers=4,
                                    collate_fn=collate_train)
        logging.info("Training and validation dataset loaded successfully.")

    else:
        inference_dataloader = DataLoader(dataset.data,
                                          batch_size=config['batch_size'],
                                          shuffle=False,
                                          drop_last=False,
                                          num_workers=4,
                                          collate_fn=partial(collate_inference, additional_param=config))
        logging.info("Inference dataset loaded successfully.")


    ## 모델 불러오고 사용할 디바이스에 올려줍니다.
    model = VAE(config)
    if config['train']==False:
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        model.load_state_dict(checkpoint)
    model.to(device)
    logging.info("모델을 디바이스로 불러왔습니다.")


    ## config 파일에서 optimizer, learning rate 세팅을 가져옵니다.
    if config['train']:
        optimizer_choice = config['optimizer_choice']
        learning_rate = config['learning_rate']
        if optimizer_choice == 'AdamW':
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        elif optimizer_choice == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        else:
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        logging.info(f"Optimizer choice: {optimizer_choice}, Learning rate: {learning_rate}")


    ## 훈련 모드: 모델을 훈련하고 체크포인트 저장 및 사용한 옵션을 기록합니다.
    ## 인퍼런스 모드: 모델을 불러오고 주어진 데이터셋에 대한 reconstruction probability, 시각화 결과를 저장합니다.
    if config['train']:
        epochs = config['epochs']
        lambda_kl = config['lambda_kl']
        val_loss_min = 1000

        ## 모델 훈련을 시작합니다.
        for epoch in range(1, epochs + 1):

            ## 훈련 사이클을 1회 돌립니다.
            train_loss = train(lambda_kl, model, train_dataloader, device, optimizer)
            logging.info(f"Epoch {epoch}, Train Loss: {train_loss:.4f}")
            
            ## 훈련 결과를 검증합니다.
            val_loss = test(lambda_kl, model, val_dataloader, device)
            logging.info(f"Epoch {epoch}, Validation Loss: {val_loss:.4f}")
            
            ## val_loss 최저점 기록시 모델 체크포인트를 저장하고, val_loss_min을 업데이트 합니다.
            if val_loss < val_loss_min:
                checkpoint_path = os.path.join(output_dirc, f"model_best.pt")
                torch.save(model.state_dict(), checkpoint_path)
                logging.info(f"Model checkpoint saved at epoch {epoch} to {checkpoint_path}")
                val_loss_min = val_loss

        ## 훈련이 종료되면 최종 모델 체크포인트를 따로 저장합니다.
        checkpoint_path = os.path.join(output_dirc, f"model_last.pt")
        torch.save(model.state_dict(), checkpoint_path)
        logging.info(f"Model checkpoint saved at epoch {epoch} to {checkpoint_path}")
        
        ## 훈련이 종료되면 마지막으로 사용한 설정값들도 함께 기록합니다.
        last_config_path = os.path.join(output_dirc, 'last_config.yaml')
        with open(last_config_path, 'w') as file:
            yaml.dump(config, file)
        logging.info(f"User configurations saved to {last_config_path}")
        logging.info("Training completed successfully.")

    else:
        all_original = []
        all_reconstructed = []
        all_probabilities = []

        ## 인퍼런스를 시작합니다.
        with torch.no_grad():
            for data_batch in tqdm(inference_dataloader, desc="Running Inference"):
                data_batch = data_batch.to(device)
                output_good, _, _, _, _, _, _, _, _, _, output_mu_all, output_logvar_all = model(data_batch)
                recon_prob = recon_probability(config, data_batch, output_mu_all, output_logvar_all)

                data_batch = data_batch.squeeze(2).transpose(0, 1).flatten()
                output_good = output_good.squeeze(2).transpose(0, 1).flatten()
                recon_prob = recon_prob.squeeze(2).transpose(0, 1).flatten()

                all_original.append(data_batch)
                all_reconstructed.append(output_good)
                all_probabilities.append(recon_prob)

        all_original = torch.cat(all_original)
        all_reconstructed = torch.cat(all_reconstructed)
        all_probabilities = torch.cat(all_probabilities)

        all_original_np = all_original.cpu().numpy()
        all_reconstructed_np = all_reconstructed.cpu().numpy()
        all_probabilities_np = all_probabilities.cpu().numpy()

        original_data = dataset.original
        original_length = len(original_data)
        all_original_np = all_original_np[:original_length]
        all_reconstructed_np = all_reconstructed_np[:original_length]
        all_probabilities_np = all_probabilities_np[:original_length]

        plot_path = os.path.join(output_dirc, 'original_vs_reconstructed.png')
        plt.figure(figsize=(16, 5))
        plt.plot(all_original_np, label='Original', alpha=1, color='blue')
        plt.plot(all_reconstructed_np, label='Reconstructed', alpha=0.5, color='orange')
        plt.plot(all_probabilities_np, label='prob', alpha=0.5, color='red')
        plt.title('Original VS Reconstructed Data')
        plt.legend()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logging.info("인퍼런스 시각화 결과를 저장했습니다: original_vs_reconstructed.png")

        recon_prob_threshold = config['recon_prob_threshold']
        anomaly_indices = np.where(all_probabilities_np <= recon_prob_threshold)[0]
        anomaly_intervals = get_anomaly_intervals(anomaly_indices)

        plot_path = os.path.join(output_dirc, 'anomalous_regions.png')
        plt.figure(figsize=(16,5))
        plt.plot(original_data['value'], label='Original', alpha=1, color='blue')
        for interval in anomaly_intervals:
            plt.axvspan(interval[0], interval[1], color='red', alpha=1)
        plt.title('Anomalous Regions')
        plt.legend()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logging.info("인퍼런스 시각화 결과를 저장했습니다: anomalous_regions.png")

        csv_path = os.path.join(output_dirc, f'anomaly_threshold_{recon_prob_threshold}.csv')
        label = np.zeros(original_length)
        label[anomaly_indices]=1
        original_data['label']=label
        original_data.to_csv(csv_path, index=False)
        logging.info(f"인퍼런스 결과를 저장했습니다: anomaly_threshold_{recon_prob_threshold}.csv'")