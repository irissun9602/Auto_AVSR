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

def set_cuda_visible_device(mig_uuid):
    # os.environ['CUDA_VISIBLE_DEVICES'] = mig_uuid
    print(f"CUDA_VISIBLE_DEVICES set to {os.environ['CUDA_VISIBLE_DEVICES']}")

class InferencePipeline(torch.nn.Module):
    def __init__(self, cfg, detector="retinaface", device="cuda"):
        super(InferencePipeline, self).__init__()
        self.modality = cfg.data.modality
        print('0')
        if self.modality in ["audio", "audiovisual"]:
            self.audio_transform = AudioTransform(subset="test")
        if self.modality in ["video", "audiovisual"]:
            if detector == "mediapipe":
                from preparation.detectors.mediapipe.detector import LandmarksDetector
                from preparation.detectors.mediapipe.video_process_1m import VideoProcess
                self.landmarks_detector = LandmarksDetector()
                self.video_process = VideoProcess(convert_gray=False)
            elif detector == "retinaface":
                from preparation.detectors.retinaface.detector import LandmarksDetector
                from preparation.detectors.retinaface.video_process_1m import VideoProcess
                print('1')
                self.landmarks_detector = LandmarksDetector(device=device) 
                print('2')
                self.video_process = VideoProcess(convert_gray=False)
                print('3')
            self.video_transform = VideoTransform(subset="test")
        print('4')
        if cfg.data.modality in ["audio", "video"]:
            from lightning import ModelModule
        elif cfg.data.modality == "audiovisual":
            from lightning_av import ModelModule
        print('5')
        self.modelmodule = ModelModule(cfg) ###############################################################
        print('6')
        self.modelmodule.model.load_state_dict(torch.load(cfg.pretrained_model_path, map_location=lambda storage, loc: storage)) ######
        print('7')
        self.modelmodule.eval()
        self.modelmodule=self.modelmodule.to(device)
        # self.modelmodule=torch.compile(self.modelmodule, mode="reduce-overhead")
    def forward(self, data_filename):
        data_filename = os.path.abspath(data_filename)
        assert os.path.isfile(data_filename), f"data_filename: {data_filename} does not exist."

        if self.modality in ["video", "audiovisual"]:
            # print('8')
            video = self.load_video(data_filename)
            # print('9')
            landmarks = self.landmarks_detector(video)
            # print('10')
            video = self.video_process(video, landmarks)
            # print('11')
            video = torch.tensor(video)
            video = video.permute((0, 3, 1, 2))
            video = self.video_transform(video)
        # print('12')
        if self.modality == "video":
            with torch.no_grad():
                transcript = self.modelmodule(video) 
            # print('13')
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
        res= torchvision.io.read_video(data_filename, pts_unit="sec")[0].numpy()
        # print(res)
        # raise
        return res

    def audio_process(self, waveform, sample_rate, target_sample_rate=16000):
        if sample_rate != target_sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, target_sample_rate
            )
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform

@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg):
    # Set the CUDA_VISIBLE_DEVICES environment variable to the MIG instance UUID
    mig_uuid = cfg.mig_uuid
    set_cuda_visible_device(mig_uuid)
    
    # Check if CUDA is available and if the device is set correctly
    print("CUDA Available:", torch.cuda.is_available())
    print("CUDA Device Count:", torch.cuda.device_count())
    
    pipeline = InferencePipeline(cfg)
    
    file_path = cfg.file_path
    file_paths = []
    with open(file_path, 'r') as f:
        for line in f:
            file_paths.append(line.strip())
    
    print(len(file_paths))

    csv_file = cfg.csv_file

    # Create or append to the CSV file
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['file_path', 'video'])

        with tqdm(total=len(file_paths)) as pbar:
            for file_path in file_paths:
                # try:
                transcript = pipeline(file_path)
                new_entry = {'file_path': file_path, 'video': ''}
                new_entry['video'] = transcript

                if cfg.data.modality == "video":
                    print(f"File: {file_path}, Video Transcript: {transcript}")
                writer.writerow(new_entry)
                pbar.update(1)
                # except Exception as e:
                    # print(f"Failed to process {file_path}: {e}")

if __name__ == "__main__":
    main()
