# #!/bin/bash
# # python demo.py data.modality='video' \
# #                pretrained_model_path='/media/NAS/USERS/moonbo/avsr/vsr_trlrwlrs2lrs3vox2avsp_base.pth' \
# #                file_path='/media/NAS/DATASET/1mDFDC/liptest/19_real.mp4'

CUDA_VISIBLE_DEVICES=MIG-a4b6f30e-d7c0-5ac1-b6cc-76c6ed8a793e python multi_demo_test_mig.py data.modality='video' \
                pretrained_model_path='/home/minsun/auto_avsr/pretrained/vsr_trlrwlrs2lrs3vox2avsp_base.pth' \
                file_path='filelist/filelist002.txt' \
                csv_file='test002.csv' \
                mig_uuid='MIG-14776a4b-3699-5390-9d01-de2e4ed3f4b2' 

# # python multi_demo.py data.modality='audio' \
# #                 pretrained_model_path='/media/NAS/USERS/moonbo/avsr/vasr_trlrwlrs2lrs3vox2avsp_base.pth' \
# #                 file_path='/media/NAS/DATASET/1mDFDC/train/'
