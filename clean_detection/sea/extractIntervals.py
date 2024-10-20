import pandas as pd

def extractIntervals(df):
    intervals = []
    starttime = None
    
    for i, row in df.iterrows():
        if row['anomaly'] == False:
            if starttime is None:
                starttime = i
        else:
            if starttime is not None:
                endtime = i
                #intervals.append([starttime.strftime('%Y-%m-%d %H:%M:%S'), endtime.strftime('%Y-%m-%d %H:%M:%S')])
                intervals.append([starttime,endtime])
                starttime = None

    if starttime is not None:
        endtime = df.index[-1]
        # intervals.append([starttime.strftime('%Y-%m-%d %H:%M:%S'), endtime.strftime('%Y-%m-%d %H:%M:%S')])
        intervals.append([starttime,endtime])

    return intervals