# Unofficial GeoNet Implementation based on Tensorflow 2.0

GeoNet: Unsupervised Learning of Dense Depth, Optical Flow and Camera Pose (CVPR 2018)

Zhichao Yin and Jianping Shi

arxiv preprint: ([https://arxiv.org/abs/1803.02276](https://arxiv.org/abs/1803.02276))

Official GeoNet project page :
https://github.com/yzcjtr/GeoNet

## Data Loader
TODO : More compatible with Tensorflow 2.0

## Training

    python main.py

## Testing

> Camera Pose

    python main.py --mode=test_pose --dataset_dir=./KITTI_odometry/dataset/ --init_ckpt_file=./checkpoint/iter-30 --batch_size=1 --seq_length=3 --pose_test_seq=9 --output_dir=./predictions/pose/posenet_seq09/

> Monocular Depth

    python main.py --mode=test_depth --dataset_dir=./raw/ --init_ckpt_file=./checkpoint/iter-30 --batch_size=1 --output_dir=./predictions/depth/depthnet_seq09/

## Optical Flow
TODO

