import csv
import jiwer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import argparse
import re

nltk.download('punkt')

# CER 계산 함수 (character error rate, 문자)
def calculate_cer(reference, hypothesis):
    return jiwer.cer(reference, hypothesis)

# 텍스트 전처리 함수
def preprocess_text(text):
    text = text.lower()  # 소문자로 변환
    text = re.sub(r'[^\w\s]', '', text)  # 구두점 제거
    return text

# 메인 함수
def main(modalities):
    input_csv_file = 'transcripts.csv'  # 주어진 CSV 파일 이름
    output_csv_file = 'output_transcripts_with_metrics.csv'

    data = []
    with open(input_csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)
    print(f"Modalities to process: {modalities}")

    # 메트릭 계산 및 컬럼 추가
    for row in data:
        gt = preprocess_text(row['gt'])
        
        metrics = {
            'wer_audio': '',
            'cer_audio': '',
            'bleu_audio': '',
            'wer_video': '',
            'cer_video': '',
            'bleu_video': '',
            'wer_audiovisual': '',
            'cer_audiovisual': '',
            'bleu_audiovisual': ''
        }
        
        for modality in modalities:
            if row.get(modality):
                hypothesis = preprocess_text(row[modality])

                # WER
                metrics[f'wer_{modality}'] = jiwer.wer(gt, hypothesis)
                
                # CER
                metrics[f'cer_{modality}'] = calculate_cer(gt, hypothesis)
                
                # BLEU
                reference_tokens = nltk.word_tokenize(gt)
                hypothesis_tokens = nltk.word_tokenize(hypothesis)
                smoothing_function = SmoothingFunction().method4
                bleu_score = sentence_bleu([reference_tokens], hypothesis_tokens, smoothing_function=smoothing_function)
                
                if bleu_score == 0:
                    print(f"BLEU Score is 0 for file: {row['file_path']}")
                    print(f"Reference: {gt}")
                    print(f"Hypothesis: {hypothesis}")
                
                metrics[f'bleu_{modality}'] = bleu_score
        
        row.update(metrics)

        # 디버깅: 메트릭 추가 후 데이터 확인
        print(f"Updated row data: {row}")

    # CSV 파일에 쓰기
    fieldnames = list(data[0].keys())
    with open(output_csv_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

    print(f"Results have been written to {output_csv_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate WER, CER, and BLEU metrics for given modalities.")
    parser.add_argument('--modalities', nargs='+', default=['audio', 'video', 'audiovisual'], 
                        help="List of modalities to process (default: ['audio', 'video']). Options: 'audio', 'video', 'audiovisual'.")
    args = parser.parse_args()

    main(args.modalities)
