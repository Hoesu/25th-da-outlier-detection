import os
import subprocess
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--directory', type=str, default='/Users/sea/YBIGTAlab/DA_outlier/sorted' , help='sorted directory path')
parser.add_argument('-c', '--contamination', type=float, default=0, help='the amount of contamination; (0, 0.5])')
args = parser.parse_args()

# sorted 디렉토리 경로
sorted_dir = args.directory

# sorted 내의 모든 하위 디렉토리 순회
subdirs = os.listdir(sorted_dir)
subdirs.sort()
for subdir in tqdm(subdirs):
    subdir_path = os.path.join(sorted_dir, subdir)
    
    if subdir in ['463_KL1_IZ1', '463_PHT_TCB_RXEX', '483_DQ1_FC1_RWIN']:
        continue
    if os.path.isdir(subdir_path):
        if args.contamination == 0:
            command = ["python", "if_train.py", "-d", subdir_path]
        else:
            command = ["python", "if_train.py", "-d", subdir_path, "-c", str(args.contamination)]
        subprocess.run(command)