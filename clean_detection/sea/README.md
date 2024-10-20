# Files
```bash
├── README.md
├── csv_to_json.py
├── displayPlot.py
├── extractIntervals.py
├── getDataframe.py
├── if_train.py
├── isolation_forest_anomaly
├── isolation_forest_anomaly.json
├── isolation_forest_result
└── run_if_train.py
```

# Isolation Forest
## if_train.py
```bash
python if_train.py -d <DIRECTORY_PATH> -c <CONTAMINATION_VALUE>
```
- 'Directory Path' : path to target direcotry
    - 'sorted' 디렉토리 내 Label 로 시작하는 폴더
    - (e.g.) /Users/sea/YBIGTAlab/DA_outlier/sorted/423_BE1_IZ1
- 'Contamination Value'
    - 입력 안하면 auto

## run_if_train.py
모든 label에 대해 하나하나 돌리는 게 귀찮아서, sorted 디렉토리 한꺼번에 돌리는 코드
```bash
pytho run_if_train.py -d <PATH TO SORTED> -c <CONTAMINATION VALUE>
```
- 'Path to sorted' : 'sorted' 디렉토리 경로
    - 폴더 명이 'sorted'여야 함 (아니면 에러 남)

# Results
- ./isolation_forest_result/