#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2021 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import warnings
import sys
import os
from preparation.tools.face_alignment.ibug.face_alignment import FANPredictor
from preparation.tools.face_detection.ibug.face_detection import RetinaFacePredictor

warnings.filterwarnings("ignore")


class LandmarksDetector:
    def __init__(self, device="cuda", model_name="resnet50"):
        self.face_detector = RetinaFacePredictor(
            device=device,
            threshold=0.8,
            model=RetinaFacePredictor.get_model(model_name),
        )
        self.landmark_detector = FANPredictor(device=device, model=None)

    def __call__(self, video_frames):
        landmarks = []
        print(f'{len(video_frames)}')
        for frame in video_frames:
            # 데이터 gpu로 보내기
            # print('?')
            detected_faces = self.face_detector(frame, rgb=False)
            # print('!')
            face_points, _ = self.landmark_detector(frame, detected_faces, rgb=True)
            # print('??')
            if len(detected_faces) == 0:
                landmarks.append(None)
            else:
                # print(f'{len(detected_faces)}')
                max_id, max_size = 0, 0
                for idx, bbox in enumerate(detected_faces):
                    bbox_size = (bbox[2] - bbox[0]) + (bbox[3] - bbox[1])
                    if bbox_size > max_size:
                        max_id, max_size = idx, bbox_size
                landmarks.append(face_points[max_id])
        return landmarks
