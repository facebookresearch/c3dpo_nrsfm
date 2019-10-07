TODO list:
overall
- [x] try clean install & run
- [x] remove all slurm calls
- [ ] make check_all.sh script
- [x] change hosting to amazon aws
    - https://our.internmc.facebook.com/intern/wiki/FAIR/Platforms/FAIRClusters/S3Storage/#public-file-hosting-dl-f

video_writer.py
- [x] fix the default ffmpeg

demo.py
- [x] make the actual file

dataset_configs.py
- [x] change the dataset root folder

eval_zoo.py
- [x] rename the eval vars to MPJPE + Stress

keypoints_dataset.py
- [ ] remove bbox_kp_visibility and move to dataset_preproc

show_rotating_models.py
- [x] dont use at all
- [ ] finish get_exp_dirs()
- [ ] move get_visdom_visuals to vis_utils

experiment.py
- [x] print results to a .txt file every epoch
- [x] remove slurm calls

fs3cmd sync -p ./data/datasets/c3dpo/datasets/ s3://dl.fbaipublicfiles.com/c3dpo_nrsfm/