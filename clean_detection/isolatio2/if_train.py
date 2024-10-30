import pandas as pd
import os
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import argparse
import pickle  # pkl 파일을 위해 pickle 사용

# 명령줄 인자 처리
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--directory', type=str, required=True, help='target directory path')
parser.add_argument('-c', '--contamination', type=float, default=0, help='the amount of contamination; (0, 0.5])')
parser.add_argument('-f', '--file', type=str, required=True, help='CSV file name')
args = parser.parse_args()

####################################################################
## DATA ############################################################

def getDataframe(datapath):
    """CSV 파일을 불러와 정렬하고 필요한 열을 선택합니다."""
    data = pd.read_csv(datapath)
    data = data.sort_values('time').reset_index(drop=True)
    data = data[['time', 'value']].rename(columns={'time': 'ds', 'value': 'y'})
    return data

# 파일 경로 설정
DATA_PATH = os.path.join(args.directory, args.file)
SAVE_PATH = os.path.join(args.directory, 'isolation_forest')
os.makedirs(SAVE_PATH, exist_ok=True)

# 데이터 불러오기
data = getDataframe(DATA_PATH)

####################################################################
## MODEL ###########################################################

# 모델 파일 이름 설정 (CSV 이름 뒤에 _IF.pkl로 저장)
model_filename = os.path.join(SAVE_PATH, args.file.replace('.csv', '_IF.pkl'))

# Isolation Forest 모델 설정 및 학습
clf = IsolationForest(
    contamination=args.contamination if args.contamination > 0 else 'auto',
    max_features=1.0, bootstrap=False, n_jobs=-1, random_state=42
)

# 모델 학습 및 저장
clf.fit(data[['y']])  # 'y' 열을 학습에 사용

# 학습된 모델을 pkl 형식으로 저장
with open(model_filename, 'wb') as f:
    pickle.dump(clf, f)

# 저장된 모델을 불러오기
with open(model_filename, 'rb') as f:
    loaded_model = pickle.load(f)

# 모델을 사용해 이상치 예측
data['anomaly'] = loaded_model.predict(data[['y']])  # 정상=1, 이상치=-1
data['anomaly'] = data['anomaly'].map({1: False, -1: True})  # True: 이상치

# 학습 결과를 파일 이름에 맞게 CSV로 저장
save_csv_path = os.path.join(SAVE_PATH, args.file.replace('.csv', '_anomaly.csv'))

