import argparse
import os
from run_autoencoder_outlier import process_data_with_autoencoder
from AutoEncoder_extract_outliers_by_window_size import extract_outliers

# argparse 설정
parser = argparse.ArgumentParser(description="AutoEncoder Outlier Detection Script")
parser.add_argument('-d', '--data_directory', type=str, default='../../data', help='Path to the data directory (combined folder)')
parser.add_argument('-s', '--save_directory', type=str, default='./autoencoder_results', help='Path to save the output files and visualizations')
parser.add_argument('-w', '--window_sizes', type=int, nargs='+', required=True, help='List of window sizes for feature extraction')
parser.add_argument('-o', '--output_directory', type=str, default='./final_output_for_autoencoder', help='Path to save the final output CSV files')
parser.add_argument('-ae','--autoencoder_type', type=str, default='autoencoder', help='choose encoder type')
args = parser.parse_args()

# 경로 설정
data_directory = os.path.abspath(args.data_directory)
save_directory = os.path.join(os.getcwd(), args.save_directory)
output_directory = os.path.join(os.getcwd(), args.output_directory)
encoder_type = args.autoencoder_type

# 1. 데이터 처리 및 이상치 탐지
print("[INFO] 데이터 처리 및 AutoEncoder 이상치 탐지 시작...")
os.makedirs(save_directory, exist_ok=True)
process_data_with_autoencoder(data_directory, save_directory, args.window_sizes, encoder_type)
print("[INFO] 데이터 처리 및 AutoEncoder 이상치 탐지 완료")

# 2. 이상치 추출 및 최종 결과 저장
print("[INFO] 이상치 추출 및 최종 결과 저장 시작...")
os.makedirs(output_directory, exist_ok=True)
extract_outliers(save_directory, output_directory)
print("[INFO] 이상치 추출 및 최종 결과 저장 완료")
