import pandas as pd
from extractIntervals import extractIntervals
import os
from datetime import datetime
import argparse
import json
from tqdm import tqdm 
from getDataframe import getDataframe

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--directory', type=str, default='/Users/sea/YBIGTAlab/DA_outlier/sorted', help='sorted directory path')
parser.add_argument('-a', '--anomaly', type=str, default='/Users/sea/YBIGTAlab/DA_outlier/isolation_forest_anomaly', help='directory path contains csv files of anomaly')
parser.add_argument('-s', '--savepath', type=str, default=None, help='save path')
# parser.add_argument('-m', '--mode', type=str, default='original', help='original or obtained')
args = parser.parse_args()

####################################################################
## SETTING #########################################################

SORTED_PATH = args.directory
ANOMALY_PATH = args.anomaly

SAVE_PATH = args.savepath
if SAVE_PATH is None:
    SAVE_PATH = os.path.dirname(SORTED_PATH)

SAVE_NAME = os.path.join(SAVE_PATH, os.path.basename(ANOMALY_PATH) + '.json')

EXECUTED = {'463_KL1_IZ1', '463_PHT_TCB_RXEX', '483_DQ1_FC1_RWIN','fm12.csv'}

####################################################################
## ORIGINAL MODE ###################################################

labels = [item for item in os.listdir(SORTED_PATH) if os.path.isdir(os.path.join(SORTED_PATH, item))]

labels = list(set(labels) - EXECUTED)
labels.sort()

maindict = dict()

for label in tqdm(labels):
    matching_files = [f for f in os.listdir(ANOMALY_PATH) if f.startswith(label) and f.endswith('.csv')]
    if len(matching_files)==0:
        continue
    matched = matching_files[0]
    anomaly = pd.read_csv(os.path.join(ANOMALY_PATH, matched)).rename(columns = {'ds':'time'})

    files = os.listdir(os.path.join(SORTED_PATH, label))
    files.sort()

    subdict = dict()
    for file in tqdm(files):
        original = pd.read_csv(os.path.join(SORTED_PATH, label, file))

        merged = pd.merge(original[['time', 'value']], anomaly[['time', 'anomaly']], on = 'time', how = 'inner')
        merged['time'] = merged['time'].apply(lambda x: x//1000).apply(datetime.fromtimestamp)
        merged = merged.sort_values('time')

        intervals = extractIntervals(merged)

        subdict[file] = intervals
    
    maindict[label] = subdict
    
####################################################################
## OBTINED MODE ####################################################

csv_files = [item for item in os.listdir(SORTED_PATH) if item.endswith('.csv')]
csv_files = list(set(csv_files) - EXECUTED)

for csv in tqdm(csv_files):
    label = csv[:-4]

    matching_files = [f for f in os.listdir(ANOMALY_PATH) if f.startswith(label) and f.endswith('.csv')]
    matched = matching_files[0]
    
    anomaly = pd.read_csv(os.path.join(ANOMALY_PATH, matched)).rename(columns = {'ds':'time'})    
    original = pd.read_csv(os.path.join(SORTED_PATH, csv)).rename(columns={'ds':'time'})

    merged = pd.merge(original[['time', 'y']], anomaly[['time', 'anomaly']], on = 'time', how = 'inner')
    merged['time'] = merged['time'].apply(lambda x: x//1000).apply(datetime.fromtimestamp)
    merged = merged.sort_values('time')

    intervals = extractIntervals(merged)
    maindict[csv] = intervals

####################################################################
## SAVING ##########################################################

with open(SAVE_NAME, 'w') as json_file:
    json.dump(maindict, json_file)  
