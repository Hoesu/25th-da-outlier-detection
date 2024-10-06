import os
import yaml
import logging
import platform
import argparse
from tqdm import tqdm

import torch
from torch import optim
import torch.utils.data
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from model import VAE
from dataset import OutlierDataset

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

def collate_fn(batch):
    batch = torch.stack(batch)
    return batch.permute(1, 0, 2)

def loss_function(config,
                  recon_x,
                  x,
                  mu,
                  logvar,
                  lamb,
                  mu_att,
                  logvar_att):
    CE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = lamb*(-0.5) * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD_attention = lamb*config['eta']*(-0.5) * torch.sum(1 + logvar_att - mu_att.pow(2) - logvar_att.exp())
    return CE + KLD + KLD_attention

def train(lambda_kl: float,
          model: VAE,
          dataloader: DataLoader,
          device: torch.device,
          optimizer: optim.Optimizer) -> float:
    
    model.train()
    train_loss = 0
    
    for trainbatch in dataloader:
        trainbatch = trainbatch.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar, _, _, _, _, _, c_t_mus, c_t_logvars, _ = model(trainbatch)
        loss = loss_function(config, recon_batch, trainbatch, mu, logvar, lambda_kl, c_t_mus, c_t_logvars)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    return train_loss
    
def test(lambda_kl: float,
         model: VAE,
         dataloader: DataLoader,
         device: torch.device) -> float:
    
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for testbatch in dataloader:
            testbatch = testbatch.to(device)
            recon_batch, mu, logvar, _, _, _, _, _, c_t_mus, c_t_logvars, _, _ = model(testbatch)
            loss = loss_function(config, recon_batch, testbatch, mu, logvar, lambda_kl, c_t_mus, c_t_logvars)
            test_loss += loss.item()
    return test_loss

def save(config):
    """
    모델 훈련, 테스트 후에는 output 디렉토리 아래에 사용한 데이터셋의
    기기명, 데이터 수집 기간, 피더 번호, 태그명 별로 디렉토리를 생성합니다.
    모델 훈련 후에는 가중치 체크포인트 파일과 사용한 config 옵션을 저장합니다.
    모델 테스트 후에는 이상치 값들과 시각화 이미지를 저장합니다.
    """
    pass

if __name__ == '__main__':
    ## Set up Argument Parser
    args = parse_args()

    ## Load config.yaml
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    ## Set up output directory and log
    data_path = config['data_path']
    dirc_name = data_path.split('/')[1]
    file_name = data_path.split('/')[2][:-4]
    output_dirc = os.path.join('./output', dirc_name, file_name)
    if config['train']:
        output_dirc = os.path.join(output_dirc, 'training_results')
    else:
        output_dirc = os.path.join(output_dirc, 'inference_results')
    if not os.path.exists(output_dirc):
        os.makedirs(output_dirc)
    log_file_path = os.path.join(output_dirc, 'logging.log')
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

    ## Set train/test mode
    if config['train']:
        logging.info("Train mode.")
    else:
        logging.info("Test mode.")

    ## Assign hyperparameters from config
    optimizer_choice = config['optimizer_choice']
    learning_rate = config['learning_rate']
    logging.info(f"Optimizer choice: {optimizer_choice}, Learning rate: {learning_rate}")

    ## load dataset from config
    dataset = OutlierDataset(config)
    if config['train']:
        dataset_size = len(dataset)
        train_size = int(dataset_size * 0.8)
        validation_size = dataset_size - train_size
        train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=config['batch_size'],
                                      shuffle=True,
                                      drop_last=True,
                                      collate_fn=collate_fn)
        val_dataloader = DataLoader(validation_dataset,
                                    batch_size=config['batch_size'],
                                    shuffle=False,
                                    drop_last=True,
                                    collate_fn=collate_fn)
        logging.info("Training and validation datasets loaded successfully.")
    else:
        test_dataloader = DataLoader(dataset.data,
                                     batch_size=config['batch_size'],
                                     shuffle=False,
                                     drop_last=True)
    logging.info("Test dataset loaded successfully.")

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

    ## Start Training
    if config['train']:
        epochs = config['epochs']
        lambda_kl = config['lambda_kl']

        for epoch in range(1, epochs + 1):
            train_loss = train(lambda_kl, model, train_dataloader, device, optimizer)
            logging.info(f"Epoch {epoch}, Train Loss: {train_loss:.4f}")
            
            # Validation step
            val_loss = test(lambda_kl, model, val_dataloader, device)
            logging.info(f"Epoch {epoch}, Validation Loss: {val_loss:.4f}")
            
            # Save model checkpoints after every epoch
            checkpoint_path = os.path.join(output_dirc, f"model_epoch_{epoch}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            logging.info(f"Model checkpoint saved at epoch {epoch} to {checkpoint_path}")
        
        logging.info("Training completed successfully.")

    else:
        # Testing phase
        lambda_kl = config['lambda_kl']
        test_loss = test(lambda_kl, model, test_dataloader, device)
        logging.info(f"Test Loss: {test_loss:.4f}")
        
        # Save test results or anomalies
        save(config)
        logging.info("Testing completed successfully and results saved.")
