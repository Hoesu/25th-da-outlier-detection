
# DA Outlier Detection

"Unsupervised Anomaly Detection in Energy Time Series Data Using Variational Recurrent Autoencoders with Attention" 논문 기반 이상치 탐지 프로젝트.

## Authors

- [YBIGTA 22기 홍세아](https://github.com/Joirv)
- [YBIGTA 23기 정회수](https://github.com/Hoesu)
- [YBIGTA 23기 김소민](https://github.com/min913)
- [YBIGTA 24기 김종진](https://github.com/ToBeWithYouPopcorn)
- [YBIGTA 25기 문찬우](https://github.com/urbanking)
- [YBIGTA 25기 한예지](https://github.com/hyez2)

## Environment
Create conda virtual environment
```bash
  conda create -n outlier python=3.11
  conda activate outlier
```
Github pull request
```bash
  git init
  git remote add origin https://github.com/YBIGTA/25th-da-outlier-detection.git
  git branch -m main
  git pull origin main
```
Install requirements
```bash
  pip install -r requirements.txt
```

## File Structure
```bash
WORKING DIRECTORY
├── data                    # Outlier dataset (confidential)
├── output                  # Model checkpoints & outlier visualizations
├── utils
│   ├── config.yaml         # Configurations
│   ├── model.py            # Model initialization
│   ├── dataset.py          # Dataset initialization
│   └── main.py             # Main method
└── requirements.txt
```

## Data Pipeline
![diagram](https://github.com/user-attachments/assets/3549393d-2edd-4822-8340-c54adf6f9e38)


## Configurations
```bash
# dataset.py
data_path: "PATH_TO_CSV"
interval_path: "PATH_TO_JSON"
step_size: 50
split_ratio: 0.8

# model.py
lstm_size: 128
latent_size: 20
input_size: 1
seq_size: 150
num_lyears: 1
batch_size: 16
attention_size: 2
sample_reps: 20
directions: 2

# main.py
train: True
recon_prob_threshold: 0.20
optimizer_choice: 'AdamW'
learning_rate: 0.02
epochs: 10
lambda_kl: 0
eta: 0.01
```

## Deployment

```bash
  python main.py -c 'CONFIG_PATH'
```

## Acknowledgements

 - ["Unsupervised Anomaly Detection in Energy Time Series Data Using Variational Recurrent Autoencoders with Attention"](https://www.joaopereira.ai/assets/pdf/accepted_version_ICMLA18.pdf)
 - [https://github.com/LauJohansson/AnomalyDetection_VAE_LSTM.git](https://github.com/LauJohansson/AnomalyDetection_VAE_LSTM.git)