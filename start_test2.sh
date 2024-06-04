#!/bin/bash

# pretrained_model_path와 MIG UUID 리스트를 정의합니다.
pretrained_model_path='/home/minsun/auto_avsr/pretrained/vsr_trlrwlrs2lrs3vox2avsp_base.pth'
mig_uuids=(
  "MIG-85ddd508-3117-5068-8480-bbad0801d23d" "MIG-14776a4b-3699-5390-9d01-de2e4ed3f4b2"
  "MIG-a4b6f30e-d7c0-5ac1-b6cc-76c6ed8a793e" "MIG-1dbada2b-0404-5d67-96b1-7a8eb4d23792"
  "MIG-874e5435-13da-55a2-a0c7-5221160114fd" "MIG-477d33a7-8c26-5389-88c5-cb1a87019153"
  "MIG-ef2c512e-0902-5220-9503-73a8b0c7ca9f" "MIG-a77e9a8d-a83a-59f1-b985-40c233fad3cb"
  "MIG-8b51d6f8-50c8-5072-ac94-89276e8f04d4" "MIG-77a63ec0-6ea4-5416-844c-86f5867f0de3"
  "MIG-69ff26c6-e624-5249-bb9c-d14a8d892e8c" "MIG-daa8ea16-1fb7-58d8-8bcb-abfdf650d79a"
  "MIG-b394e376-f4b3-5a27-8ecf-9670c8ea4560" "MIG-044c7ee3-0ec6-5451-b501-9c619bef22ae"
  "MIG-77efc30a-bf20-560b-931b-2ff2097e57bb" "MIG-6184623f-3477-5d5f-94ca-bf1cb090a8fc"
  "MIG-cccc894c-0d51-5b7e-ba6c-150c913fa561" "MIG-62143160-9d06-5f5c-a6f6-f42e9c6e4d0c"
  "MIG-ef239cdd-a7b7-50c7-9500-2e8106076a20" "MIG-7fb04b1c-cbee-5b15-85ad-7926d15f53a9"
  "MIG-3f430cb4-229b-547f-8dcd-40cfb87dd618" "MIG-76a6bbd8-eb33-547d-82d4-5c6c7b449239"
  "MIG-aa0b9585-5524-5c4a-b267-42f1ea937007" "MIG-a2a0b0e9-7a10-53d7-8e06-817e6c30f3af"
  "MIG-8938c481-5cd2-51eb-bddf-5587732e0adf" "MIG-136a3531-5419-5b3d-a04c-3aab64e6d11c"
  "MIG-0fa18cbb-91fa-5611-9fdd-1462529a2df4" "MIG-42e14448-0186-5307-8276-c821ec46e9b2"
)

# 각 MIG UUID에 대해 스크립트를 실행합니다.
python multi_demo_test_mig.py data.modality='video' \
  pretrained_model_path="${pretrained_model_path}" \
  file_path="filelist/filelist014.txt" \
  csv_file="test014.csv" \
  mig_uuid="${mig_uuids[14]}" &

python multi_demo_test_mig.py data.modality='video' \
  pretrained_model_path="${pretrained_model_path}" \
  file_path="filelist/filelist015.txt" \
  csv_file="test015.csv" \
  mig_uuid="${mig_uuids[15]}" &

python multi_demo_test_mig.py data.modality='video' \
  pretrained_model_path="${pretrained_model_path}" \
  file_path="filelist/filelist016.txt" \
  csv_file="test016.csv" \
  mig_uuid="${mig_uuids[16]}" &

python multi_demo_test_mig.py data.modality='video' \
  pretrained_model_path="${pretrained_model_path}" \
  file_path="filelist/filelist017.txt" \
  csv_file="test017.csv" \
  mig_uuid="${mig_uuids[17]}" &
python multi_demo_test_mig.py data.modality='video' \
  pretrained_model_path="${pretrained_model_path}" \
  file_path="filelist/filelist018.txt" \
  csv_file="test018.csv" \
  mig_uuid="${mig_uuids[18]}" &

python multi_demo_test_mig.py data.modality='video' \
  pretrained_model_path="${pretrained_model_path}" \
  file_path="filelist/filelist019.txt" \
  csv_file="test019.csv" \
  mig_uuid="${mig_uuids[19]}" &

python multi_demo_test_mig.py data.modality='video' \
  pretrained_model_path="${pretrained_model_path}" \
  file_path="filelist/filelist020.txt" \
  csv_file="test020.csv" \
  mig_uuid="${mig_uuids[20]}" &


# 필요한 만큼 이어서 작성합니다.

# wait 명령어를 통해 모든 백그라운드 작업이 완료될 때까지 기다림
wait