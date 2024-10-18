import pandas as pd
import os
from datetime import datetime
from typing import Literal

def getDataframe (datapath, mode : Literal['datetime', 'timestamp'] = 'datetime'):
    '''
    * isolation forest : timestamp
    '''
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
    
    if mode == 'datetime':
        data['timestamp'] = data['time'].apply(lambda x: x//1000)
        data['datetime'] = data['timestamp'].apply(datetime.fromtimestamp)

        data = data[['datetime', 'value']]
        data = data.rename(columns={'datetime': 'ds', 'value': 'y'})
    else:
        data = data[['time', 'value']]
        data = data.rename(columns={'time': 'ds', 'value': 'y'})

    y_values = data['y'].values.reshape(-1, 1)

    return data, y_values