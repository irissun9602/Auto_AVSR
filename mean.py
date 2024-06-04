import csv
import numpy as np

# CSV 파일 경로
input_csv_file = 'output_transcripts_with_metrics.csv'

# 데이터 읽기
data = []
with open(input_csv_file, mode='r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        data.append(row)

# 메트릭 초기화
metrics = {
    'wer_audio': [],
    'cer_audio': [],
    'bleu_audio': [],
    'wer_video': [],
    'cer_video': [],
    'bleu_video': [],
    'wer_audiovisual': [],
    'cer_audiovisual': [],
    'bleu_audiovisual': []
}

# 데이터 수집
for row in data:
    for metric in metrics:
        if row[metric] != '':
            metrics[metric].append(float(row[metric]))

# 평균 계산 및 출력
for metric, values in metrics.items():
    if values:
        average = np.mean(values)
        print(f"Average {metric}: {average}")
    else:
        print(f"Average {metric}: No data")
