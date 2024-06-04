import csv

# 파일 경로
transcript_a_file = 'transcripts_a.csv'
transcript_v_file = 'transcripts_v.csv'
transcript_av_file = 'transcripts_av.csv'
output_file = 'transcripts.csv'

# 데이터를 읽어오기 위한 딕셔너리
audio_data = {}

# transcript_a.csv 파일을 읽어와서 audio 데이터 저장
with open(transcript_a_file, mode='r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        file_path = row['file_path']
        audio_data[file_path] = row['audio']

# transcript_v.csv 파일을 읽어와서 audio 데이터를 추가한 후 새로운 파일로 저장
with open(transcript_v_file, mode='r') as file:
    reader = csv.DictReader(file)
    fieldnames = reader.fieldnames
    rows = list(reader)

with open(output_file, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()

    for row in rows:
        file_path = row['file_path']
        row['audio'] = audio_data.get(file_path, '')  # audio 데이터 추가
        writer.writerow(row)

print(f"Results have been written to {output_file}")
