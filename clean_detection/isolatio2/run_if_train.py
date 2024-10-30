import os
import subprocess
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--directory', type=str, help='sorted directory path')
parser.add_argument('-c', '--contamination', type=float, default=0, help='the amount of contamination; (0, 0.5])')
args = parser.parse_args()

# 디렉터리 내 모든 CSV 파일 처리
csv_files = [f for f in os.listdir(args.directory) if f.endswith('.csv')]
csv_files.sort()

for csv_file in tqdm(csv_files):
    csv_path = os.path.join(args.directory, csv_file)
    print(f"Processing {csv_file}")

    # 커맨드 생성 및 실행
    if args.contamination == 0:
        command = ["python", "if_train.py", "-d", args.directory, "-f", csv_file]
    else:
        command = ["python", "if_train.py", "-d", args.directory, "-f", csv_file, "-c", str(args.contamination)]

    subprocess.run(command)
