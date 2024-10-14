import argparse
import os
from run_deep_autoencoder_XGB_outlier import process_data_with_deep_autoencoder
from DeepAE_XGB_extract_outliers_by_window_size import extract_outliers

# argparse 설정
parser = argparse.ArgumentParser(description="Deep AutoEncoder Outlier Detection Script")
parser.add_argument('-d', '--data_directory', type=str, default='../../data', help='Path to the data directory (combined folder)')
parser.add_argument('-s', '--save_directory', type=str, default='./deepAE_XGB_results', help='Path to save the output files and visualizations')
parser.add_argument('-w', '--window_sizes', type=int, nargs='+', required=True, help='List of window sizes for feature extraction')
parser.add_argument('-o', '--output_directory', type=str, default='./final_output_for_deepAE_XGB', help='Path to save the final output CSV files')
args = parser.parse_args()

# 경로 설정
data_directory = os.path.abspath(args.data_directory)
save_directory = os.path.join(os.getcwd(), args.save_directory)
output_directory = os.path.join(os.getcwd(), args.output_directory)

# 1. 데이터 처리 및 이상치 탐지
os.makedirs(save_directory, exist_ok=True)
process_data_with_deep_autoencoder(data_directory, save_directory, args.window_sizes)

# 2. 이상치 추출 및 최종 결과 저장
os.makedirs(output_directory, exist_ok=True)
extract_outliers(save_directory, output_directory)

print(f"Deep AutoEncoder 이상치 탐지 및 결과 저장 완료!")
