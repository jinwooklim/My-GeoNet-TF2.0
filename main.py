from __future__ import absolute_import, division, print_function, unicode_literals
import os
import time
import pprint
import argparse
from data_loader import DataLoader
from geonet_model import *
# from geonet_train import *
from geonet_test_pose import *
from geonet_test_depth import *
import matplotlib.pyplot as plt
import cv2
import sys
import random
sys.path.insert(0, './kitti_eval/flow_tool/')
import flowlib as fl


parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='train_rigid', help='Mode of program')\
# parser.add_argument('--mode', type=str, default='train_flow', help='Mode of program')
parser.add_argument('--dataset_dir', type=str, default='E:\\all_dataset\\KITTI_dump', help='Path of dataset')
parser.add_argument('--shuffle_buffer_size', type=int, default=2048, help='shuffle_buffer_size') # 128
parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
parser.add_argument('--img_height', type=int, default=128, help='img_height')
parser.add_argument('--img_width', type=int, default=416, help='img_width')
parser.add_argument('--seq_length', type=int, default=3, help='seq_length')
parser.add_argument('--num_source', type=int, default=2, help='num_source')
parser.add_argument('--num_scales', type=int, default=4, help='num_scales')

parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint/', help='Checkpoint dir')
# parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint/flow/', help='Checkpoint dir')
parser.add_argument('--init_ckpt_file', type=str, default='./checkpoint/iter-0', help='Ckpt name')
parser.add_argument('--save_ckpt_freq', type=int, default=100000)
parser.add_argument('--max_to_keep', type=int, default=3)
parser.add_argument('--summary_dir', type=str, default='./summary/', help='summary_dir')
parser.add_argument('--save_summary_freq', type=int, default=500)

parser.add_argument('--learning_rate', type=float, default=0.0002)
parser.add_argument('--max_steps', type=int, default=600100)
parser.add_argument('--alpha_recon_image', type=float, default=0.85)
parser.add_argument('--rigid_warp_weight', type=float, default=1.0)
parser.add_argument('--disp_smooth_weight', type=float, default=0.5)

parser.add_argument('--flownet_type', type=str, default='residual')
parser.add_argument('--flow_warp_weight', type=float, default=1.0)
parser.add_argument('--flow_smooth_weight', type=float, default=0.2)
parser.add_argument('--flow_consistency_weight', type=float, default=0.2)
parser.add_argument('--flow_consistency_alpha', type=float, default=3.0)
parser.add_argument('--flow_consistency_beta', type=float, default=0.05)

parser.add_argument('--output_dir', type=str, default='./predictions/')
parser.add_argument('--pose_test_seq', type=int, default=9)

FLAGS = parser.parse_args()
FLAGS = vars(FLAGS) # Convert Namespace object to vars object


if __name__ == "__main__":
    pp = pprint.PrettyPrinter()
    pp.pprint(FLAGS)

    if FLAGS['mode'] == 'train_flow':
        # train_flow(FLAGS)
        pass
    elif FLAGS['mode'] == "test_pose":
        test_pose(FLAGS)
    elif FLAGS['mode'] == "test_depth":
        test_depth(FLAGS)

