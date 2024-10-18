import pandas as pd
import os
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--directory', type=str, required=True, help='target directory path')
parser.add_argument('-c', '--contamination', type=float, default=0, help='the amount of contamination; (0, 0.5])')
args = parser.parse_args()


####################################################################
## DATA ############################################################

def getDataframe(datapath):
  csv_files = [file for file in os.listdir(datapath) if file.endswith('.csv')]
  csv_files.sort()

  data = None
  for file in csv_files:
      csv_path = os.path.join(datapath, file)
      if data is None:
          data = pd.read_csv(csv_path)
      else :
          temp = pd.read_csv(csv_path)
          data = pd.concat([data, temp])

  data = data.sort_values('time')
  data = data.reset_index().drop(['index'], axis=1)

  #data['timestamp'] = data['time'].apply(lambda x: x//1000)
  #data['datetime'] = data['timestamp'].apply(datetime.fromtimestamp)

  #data = data[['datetime', 'value']]
  #data = data.rename(columns={'datetime': 'ds', 'value': 'y'})
  data = data[['time', 'value']]
  data = data.rename(columns={'time': 'ds', 'value': 'y'})

  y_values = data['y'].values.reshape(-1, 1)

  return data, y_values


DATA_PATH = args.directory
SUBDIR_NAME = os.path.basename(DATA_PATH)
BASE_PATH = DATA_PATH.split("/sorted")[0]
SAVE_PATH = os.path.join(BASE_PATH, 'isolation_forest')

os.makedirs(SAVE_PATH, exist_ok=True)

data, y_values = getDataframe(DATA_PATH)

####################################################################
## MODEL ###########################################################

train_x, test_x = train_test_split(data['y'], train_size=0.8, test_size=0.2, random_state=1)

if args.contamination == 0:
    clf = IsolationForest(  max_features=1.0, bootstrap=False, n_jobs=-1,
                            random_state=None, verbose=0)
else :
    clf = IsolationForest(  contamination=float(args.contamination),
						    max_features=1.0, bootstrap=False, n_jobs=-1,
						    random_state=None, verbose=0)

clf.fit(data)

pred = clf.predict(data)
data['anomaly'] = pred
# normal = 1 / outlier = -1

outliers= data.loc[data['anomaly']==-1]
outlier_index=list(outliers.index)

####################################################################
## VISUALIZATION ###################################################

detected = data
detected['anomaly'] = detected['anomaly'].map({1: False, -1: True})
detected.to_csv(os.path.join(SAVE_PATH, SUBDIR_NAME + '_if_anomaly_' + str(args.contamination) + '.csv'))

plt.figure(figsize=(12, 6))

# anomaly가 False인 데이터
plt.scatter(detected[detected['anomaly'] == 1]['ds'], detected[detected['anomaly'] == 1]['y'], color='black', label='Normal', s=1)

# anomaly가 True인 데이터
plt.scatter(detected[detected['anomaly'] == -1]['ds'], detected[detected['anomaly'] == -1]['y'], color='red', label='Anomaly', s=10)

# 라인 차트 그리기
plt.plot(detected['ds'], detected['y'], color='gray', alpha=0.5)

# 레이블 및 타이틀 설정
plt.xlabel('Date Time')
plt.ylabel('y Value')
if args.contamination == 0:
    plt.title(SUBDIR_NAME + ' cont=auto')
else:
    plt.title(SUBDIR_NAME + ' cont=' + str(args.contamination))
plt.legend()
plt.xticks(rotation=45)
plt.grid()

# 그래프 표시
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH, SUBDIR_NAME + '_if_anomaly_' + str(args.contamination) + '.png'))
