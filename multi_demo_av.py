import os
import hydra
import glob
import csv
import json
from tqdm import tqdm

import torch
import torchaudio
import torchvision
from datamodule.transforms import AudioTransform, VideoTransform
from datamodule.av_dataset import cut_or_pad


class InferencePipeline(torch.nn.Module):
    def __init__(self, cfg, detector="retinaface"):
        super(InferencePipeline, self).__init__()
        self.modality = cfg.data.modality
        if self.modality in ["audio", "audiovisual"]:
            self.audio_transform = AudioTransform(subset="test")
        if self.modality in ["video", "audiovisual"]:
            if detector == "mediapipe":
                from preparation.detectors.mediapipe.detector import LandmarksDetector
                from preparation.detectors.mediapipe.video_process import VideoProcess
                self.landmarks_detector = LandmarksDetector()
                self.video_process = VideoProcess(convert_gray=False)
            elif detector == "retinaface":
                from preparation.detectors.retinaface.detector import LandmarksDetector
                from preparation.detectors.retinaface.video_process import VideoProcess
                self.landmarks_detector = LandmarksDetector(device="cuda:0")
                self.video_process = VideoProcess(convert_gray=False)
            self.video_transform = VideoTransform(subset="test")

        if cfg.data.modality in ["audio", "video"]:
            from lightning import ModelModule
        elif cfg.data.modality == "audiovisual":
            from lightning_av import ModelModule
        self.modelmodule = ModelModule(cfg)
        self.modelmodule.model.load_state_dict(torch.load(cfg.pretrained_model_path, map_location=lambda storage, loc: storage))
        self.modelmodule.eval()


    def forward(self, data_filename):
        data_filename = os.path.abspath(data_filename)
        assert os.path.isfile(data_filename), f"data_filename: {data_filename} does not exist."

        if self.modality in ["audio", "audiovisual"]:
            audio, sample_rate = self.load_audio(data_filename)
            audio = self.audio_process(audio, sample_rate)
            audio = audio.transpose(1, 0)
            audio = self.audio_transform(audio)

        if self.modality in ["video", "audiovisual"]:
            video = self.load_video(data_filename)
            landmarks = self.landmarks_detector(video)
            video = self.video_process(video, landmarks)
            video = torch.tensor(video)
            video = video.permute((0, 3, 1, 2))
            video = self.video_transform(video)

        if self.modality == "video":
            with torch.no_grad():
                transcript = self.modelmodule(video)
        elif self.modality == "audio":
            with torch.no_grad():
                transcript = self.modelmodule(audio)

        elif self.modality == "audiovisual":
            print(len(audio), len(video))
            assert 530 < len(audio) // len(video) < 670, "The video frame rate should be between 24 and 30 fps."

            rate_ratio = len(audio) // len(video)
            if rate_ratio == 640:
                pass
            else:
                print(f"The ideal video frame rate is set to 25 fps, but the current frame rate ratio, calculated as {len(video)*16000/len(audio):.1f}, which may affect the performance.")
                audio = cut_or_pad(audio, len(video) * 640)
            with torch.no_grad():
                transcript = self.modelmodule(video, audio)

        return transcript

    def load_audio(self, data_filename):
        waveform, sample_rate = torchaudio.load(data_filename, normalize=True)
        return waveform, sample_rate

    def load_video(self, data_filename):
        return torchvision.io.read_video(data_filename, pts_unit="sec")[0].numpy()

    def audio_process(self, waveform, sample_rate, target_sample_rate=16000):
        if sample_rate != target_sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, target_sample_rate
            )
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform

@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg):
    pipeline = InferencePipeline(cfg)
    
    root_path = os.path.abspath(cfg.file_path)
    sub_dirs = sorted(os.listdir(root_path))  # 상위 디렉터리 이름을 기준으로 정렬

    file_paths = []
    for sub_dir in sub_dirs:
        dir_path = os.path.join(root_path, sub_dir)
        sub_file_paths = glob.glob(os.path.join(dir_path, "*/*/real.mp4"))
        file_paths.extend(sub_file_paths)
        if len(file_paths) >= 100:
            break  # 최대 100개의 파일까지만 선택

    file_paths = file_paths[:100]  # 파일 목록을 최대 100개로 제한

    csv_file = 'transcripts_av.csv'

    # Check if the CSV file already exists
    if os.path.exists(csv_file):
        # Load existing data
        with open(csv_file, mode='r') as file:
            reader = csv.DictReader(file)
            existing_data = list(reader)
    else:
        existing_data = []

    # Create or append to the CSV file
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['file_path', 'audiovisual'])

        if not existing_data:
            writer.writeheader()
        else:
            # Write existing data first
            writer.writerows(existing_data)

        with tqdm(total=len(file_paths)) as pbar:
            for file_path in file_paths:
                try:
                    transcript = pipeline(file_path)

                    # Check if the file_path already exists in the CSV
                    existing_entry = next((row for row in existing_data if row['file_path'] == file_path), None)
                    
                    if existing_entry:
                        # Update the existing entry with the new transcript
                        existing_entry['audiovisual'] = transcript
                    else:
                        # Create a new entry
                        new_entry = {'file_path': file_path, 'audiovisual': transcript}
                        existing_data.append(new_entry)
                    
                    # Print the result of the current file
                    print(f"File: {file_path}, Audiovisual Transcript: {transcript}")
                    
                    # Write the updated or new entry
                    writer.writerow(existing_entry if existing_entry else new_entry)
                    pbar.update(1)
                except Exception as e:
                    print(f"Failed to process {file_path}: {e}")

if __name__ == "__main__":
    main()