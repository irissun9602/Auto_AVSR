import csv

# 파일 경로
transcripts_file = 'transcripts.csv'
transcripts_av_file = 'transcripts_av.csv'

# 데이터를 읽어오기 위한 딕셔너리
data = {}
av_data = {}

# transcripts.csv 파일을 읽어와서 데이터 저장
with open(transcripts_file, mode='r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        file_path = row['file_path']
        data[file_path] = row

# transcripts_av.csv 파일을 읽어와서 데이터 저장
with open(transcripts_av_file, mode='r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        file_path = row['file_path']
        av_data[file_path] = row['audiovisual']

# 데이터 병합
for file_path in data:
    if file_path in av_data:
        data[file_path]['audiovisual'] = av_data[file_path]
    else:
        data[file_path]['audiovisual'] = ''  # AV 데이터가 없는 경우 빈 문자열 추가

# CSV 파일에 쓰기 (순서 조정)
fieldnames = list(data[next(iter(data))].keys())
fieldnames.remove('audiovisual')
fieldnames.remove('gt')
fieldnames.append('audiovisual')
fieldnames.append('gt')

with open(transcripts_file, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    for row in data.values():
        writer.writerow(row)

print(f"Results have been written to {transcripts_file}")

