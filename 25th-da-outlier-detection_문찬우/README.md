
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
아나콘다 가상환경 생성
```bash
  conda create -n outlier python=3.11
  conda activate outlier
```
깃허브 코드 가져오기
```bash
  git init
  git remote add origin https://github.com/YBIGTA/25th-da-outlier-detection.git
  git branch -m main
  git pull origin main
```
필수 라이브러리 설치
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

## Configurations
```bash
  TO BE ANNOUNCED.
```

## Deployment

```bash
  python main.py -c 'CONFIG_PATH'
```

## Acknowledgements

 - ["Unsupervised Anomaly Detection in Energy Time Series Data Using Variational Recurrent Autoencoders with Attention"](https://www.joaopereira.ai/assets/pdf/accepted_version_ICMLA18.pdf)
 - [https://github.com/LauJohansson/AnomalyDetection_VAE_LSTM.git](https://github.com/LauJohansson/AnomalyDetection_VAE_LSTM.git)