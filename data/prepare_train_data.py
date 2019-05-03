#-*- coding: utf-8 -*-
# Mostly based on the code written by Tinghui Zhou: 
# https://github.com/tinghuiz/SfMLearner/blob/master/data/prepare_train_data.py
from __future__ import division
import argparse
import scipy.misc
import numpy as np
from glob import glob
from joblib import Parallel, delayed
import os
import platform
import KITTIto6
# import pose2obd
import prepare_train_data_obd
import make_relative_pose
from obd2flow import obd2flow, warp_image, flow_to_image #, flow_warp
import writeFlowFile
import readFlowFile
import utils
import utils_road
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir",   type=str, required=True, help="where the dataset is stored")
parser.add_argument("--dataset_name",  type=str, required=True, choices=["kitti_raw_eigen", "kitti_raw_stereo", "kitti_odom", "cityscapes"])
parser.add_argument("--dump_root",     type=str, required=True, help="where to dump the data")
parser.add_argument("--seq_length",    type=int, required=True, help="length of each training sequence")
parser.add_argument("--img_height",    type=int, default=128,   help="image height")
parser.add_argument("--img_width",     type=int, default=416,   help="image width")
parser.add_argument("--num_threads",   type=int, default=4,     help="number of threads to use")
parser.add_argument("--remove_static", help="remove static frames from kitti raw data", action='store_true')

##### Configuration for OBD #####
parser.add_argument("--odompose_dir", type=str, help="where the odometry pose is stored")
parser.add_argument('--test_seq', type=int, help="sequence for test")
args = parser.parse_args()


def concat_image_seq(seq):
    for i, im in enumerate(seq):
        if i == 0:
            res = im
        else:
            res = np.hstack((res, im))
    return res


def dump_example(n):
    if n % 2000 == 0:
        print('Progress %d/%d....' % (n, data_loader.num_train))
    example = data_loader.get_train_example_with_idx(n)
    if example == False:
        return
    image_seq = concat_image_seq(example['image_seq'])
    intrinsics = example['intrinsics']
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    dump_dir = os.path.join(args.dump_root, example['folder_name'])

    try: 
        os.makedirs(dump_dir)
    except OSError:
        if not os.path.isdir(dump_dir):
            raise
    dump_img_file = dump_dir + '/%s.jpg' % example['file_name']
    scipy.misc.imsave(dump_img_file, image_seq.astype(np.uint8))
    dump_cam_file = dump_dir + '/%s_cam.txt' % example['file_name']
    with open(dump_cam_file, 'w') as f:
        f.write('%f,0.,%f,0.,%f,%f,0.,0.,1.' % (fx, cx, fy, cy))
    
    if args.dataset_name == 'kitti_odom':
        ## Input path
        poses_dir = os.path.join(args.dataset_dir, 'poses', example['folder_name'])
        ## Output path
        dump_pose_file = os.path.join(dump_dir + '/%s_pose.txt' % example['file_name'])
        with open(dump_pose_file, 'w') as fp:
            ##### KITTI_img_file -> POSE #####
            poseSix = KITTIto6.convert(poses_dir, example['file_name'])
            fp.write('%f,%f,%f,%f,%f,%f'%(poseSix[0], poseSix[1], poseSix[2], poseSix[3], poseSix[4], poseSix[5]))


def dump_test_example(n):
    if n % 2000 == 0:
        print('Progress %d/%d....' % (n, data_loader.num_test))
    example = data_loader.get_test_example_with_idx(n)
    if example == False:
        return
    image_seq = concat_image_seq(example['image_seq'])
    intrinsics = example['intrinsics']
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    dump_dir = os.path.join(args.dump_root, example['folder_name'])

    try:
        os.makedirs(dump_dir)
    except OSError:
        if not os.path.isdir(dump_dir):
            raise
    dump_img_file = dump_dir + '/%s.jpg' % example['file_name']
    scipy.misc.imsave(dump_img_file, image_seq.astype(np.uint8))
    dump_cam_file = dump_dir + '/%s_cam.txt' % example['file_name']
    with open(dump_cam_file, 'w') as f:
        f.write('%f,0.,%f,0.,%f,%f,0.,0.,1.' % (fx, cx, fy, cy))

    if args.dataset_name == 'kitti_odom':
        ## Input path
        poses_dir = os.path.join(args.dataset_dir, 'poses', example['folder_name'])
        ## Output path
        dump_pose_file = os.path.join(dump_dir + '/%s_pose.txt' % example['file_name'])
        with open(dump_pose_file, 'w') as fp:
            ##### KITTI_img_file -> POSE #####
            poseSix = KITTIto6.convert(poses_dir, example['file_name'])
            fp.write('%f,%f,%f,%f,%f,%f' % (poseSix[0], poseSix[1], poseSix[2], poseSix[3], poseSix[4], poseSix[5]))


def dump_relative_pose(n):
    ##### FOR obd #####
    if n % 2000 == 0:
        print('dump relative pose %d/%d....' % (n, data_loader.num_train))
    example = data_loader.get_train_example_with_idx(n)
    if example == False:
        return
    dump_dir = os.path.join(args.dump_root, example['folder_name'])

    try:
        os.makedirs(dump_dir)
    except OSError:
        if not os.path.isdir(dump_dir):
            raise

    dump_relpose_file = os.path.join(dump_dir + '/%s_relpose.txt' % example['file_name'])
    with open(dump_relpose_file, 'w') as fo:
        ##### POSE -> OBD #####
        relpose = make_relative_pose.convert(data_loader.num_train, dump_dir, example['file_name'])
        fo.write('%f,%f,%f,%f,%f,%f'%(relpose[0], relpose[1], relpose[2], relpose[3], relpose[4], relpose[5]))

# def dump_pose2obd(n):
#     ##### FOR obd #####
#     if n % 2000 == 0:
#         print('pose2obd %d/%d....' % (n, data_loader.num_train))
#     example = data_loader.get_train_example_with_idx(n)
#     if example == False:
#         return
#     dump_dir = os.path.join(args.dump_root, example['folder_name'])
#
#     try:
#         os.makedirs(dump_dir)
#     except OSError:
#         if not os.path.isdir(dump_dir):
#             raise
#
#     dump_obd_file = os.path.join(dump_dir + '/%s_obd.txt' % example['file_name'])
#     with open(dump_obd_file, 'w') as fo:
#         ##### POSE -> OBD #####
#         obd, offset = pose2obd.convert(args.seq_length, dump_dir, example['file_name'])
#         for i in range(offset*2):
#             #print(i, obd[i])
#             if(i < (offset*2)- 1):
#                 fo.write('%f,%f,'%(obd[i][0], obd[i][1]))
#             else:
#                 fo.write('%f,%f'%(obd[i][0], obd[i][1]))


def dump_obd2flow(n):
    if(n%100 == 0):
        print('Dump obd2flow %d/%d....' % (n, data_loader.num_train))
    example = data_loader.get_train_example_with_idx(n)
    if example == False:
        return
    dump_dir = os.path.join(args.dump_root, example['folder_name'])

    try:
        os.makedirs(dump_dir)
    except OSError:
        if not os.path.isdir(dump_dir):
            raise
    image_seq = np.array(example['image_seq'])

    dumped_intrinsic_file = open(os.path.join(dump_dir + '/%s_cam.txt' % example['file_name']))
    dumped_intrinsic = dumped_intrinsic_file.readline()
    dumped_intrinsic = dumped_intrinsic.strip("\n")
    dumped_intrinsic = dumped_intrinsic.split(",")
    dumped_intrinsic = [int(float(n)) for n in dumped_intrinsic]
    dumped_intrinsic = np.reshape(dumped_intrinsic, (3, 3))
    dumped_intrinsic = tf.expand_dims(tf.convert_to_tensor(dumped_intrinsic, tf.float32), axis=0)

    # print(dumped_intrinsic) # ok

    dumped_obd_file = open(os.path.join(dump_dir + '/%s_obd.txt' % example['file_name']))
    dumped_obd = dumped_obd_file.readline()
    dumped_obd = dumped_obd.strip("\n")
    dumped_obd = dumped_obd.split(",")
    dumped_obd = [int(float(n)) for n in dumped_obd]
    dumped_obd = np.reshape(dumped_obd, (args.seq_length - 1, 2))

    NORMALIZE_VALUE = 1 # 1000 # 300

    for i in range(args.seq_length - 1):
        # print(i, dumped_obd[i])

        # # Version # 1
        # obd_flow_path = os.path.join(dump_dir + '/%s_obd_flow_%02d.flo' % (example['file_name'], i))
        # if(i == 0):
        #     src1_img = tf.expand_dims(tf.convert_to_tensor(image_seq[0, :, :], tf.uint8), axis=0)
        #     obd_flow = obd2flow(dumped_obd[i], image_seq[0, :, :]) # flow src1 -> tgt
        #     obd_flow = obd_flow / NORMALIZE_VALUE
        #     obd_warped_img = utils.flow_warp(src1_img, -obd_flow)
        #     res = obd_warped_img
        # elif(i == 1):
        #     src2_img = tf.expand_dims(tf.convert_to_tensor(image_seq[2, :, :], tf.uint8), axis=0)
        #     obd_flow = obd2flow(-dumped_obd[i], image_seq[2, :, :]) # flow src2 -> tgt
        #     obd_flow = obd_flow / NORMALIZE_VALUE
        #     obd_warped_img = utils.flow_warp(src2_img, obd_flow)
        #     res = tf.concat([res, obd_warped_img], axis=2)
        # writeFlowFile.write(obd_flow, obd_flow_path)
        #
        # # Visualize the obd_rigid_flow
        # direct_flow2img_path = os.path.join(dump_dir + '/%s_obd_flow_img_%02d.png' % (example['file_name'], i))
        # obd_rigid_flow_img = flow_to_image(obd_flow)
        # scipy.misc.imsave(direct_flow2img_path, obd_rigid_flow_img.astype(np.uint8))

        # Version # 2
        obd_flow_path = os.path.join(dump_dir + '/%s_obd_flow_%02d.flo' % (example['file_name'], i))
        if(i==0): # bwd
            src1_img = tf.expand_dims(tf.convert_to_tensor(image_seq[0, :, :], tf.uint8), axis=0)
            tgt_img = tf.expand_dims(tf.convert_to_tensor(image_seq[1, :, :], tf.uint8), axis=0)
            temp_pose = utils_road.get_pose_by_obd(dumped_obd[i])
            # obd_rigid_flow = utils.compute_rigid_flow(tgt_img, -temp_pose, dumped_intrinsic, reverse_pose=True)  # False # 아래가 긴 평행사변형
            obd_rigid_flow = utils_road.compute_rigid_flow(tgt_img, temp_pose, dumped_intrinsic, reverse_pose=True)  # False
            obd_warped_img = utils_road.flow_warp(src1_img, obd_rigid_flow)
            res = obd_warped_img
        elif(i==1): # fwd
            src2_img = tf.expand_dims(tf.convert_to_tensor(image_seq[2, :, :], tf.uint8), axis=0)
            tgt_img = tf.expand_dims(tf.convert_to_tensor(image_seq[1, :, :], tf.uint8), axis=0)
            temp_pose = utils_road.get_pose_by_obd(dumped_obd[i])
            obd_rigid_flow = utils_road.compute_rigid_flow(tgt_img, temp_pose, dumped_intrinsic, reverse_pose=False)
            obd_warped_img = utils_road.flow_warp(src2_img, obd_rigid_flow)
            res = tf.concat([res, obd_warped_img], axis=2)

        obd_rigid_flow = tf.squeeze(obd_rigid_flow, axis=0)
        # print(obd2flow)
        # writeFlowFile.write(obd_rigid_flow, obd_flow_path) # Write obd file

        # Visualize the obd_rigid_flow
        # direct_flow2img_path = os.path.join(dump_dir + '/%s_obd_flow_img_%02d.png' % (example['file_name'], i))
        # obd_rigid_flow_img = flow_to_image(obd_rigid_flow)
        # scipy.misc.imsave(direct_flow2img_path, obd_rigid_flow_img.astype(np.uint8)) # Write each obd_flow_img


    # Attach
    warpedImg_path = os.path.join(dump_dir + '/%s_warped_img.png' % (example['file_name']))
    res = tf.squeeze(res, axis=0)
    # scipy.misc.imsave(warpedImg_path, res.astype(np.uint8))
    scipy.misc.imsave(warpedImg_path, res)


def create_test_flow_text(dump_root, seq_number):
    # Write "test_flow.txt" on dump_root
    if platform.system() == 'Linux':
        test_seq = seq_number
    elif platform.system() == 'Windows':
        test_seq = seq_number
    with open(os.path.join(dump_root, 'test_flow_%s.txt'%seq_number), 'w') as txf:
        print(str(os.path.join(dump_root, seq_number)))
        data_list = [f for f in os.listdir(os.path.join(dump_root, seq_number)) if os.path.splitext(f)[1] == '.jpg']
        data_list.sort()
        for i in range(len(data_list)):
            txf.write("%s %s\n"%(seq_number, os.path.splitext(data_list[i])[0]))


def main():
    if not os.path.exists(args.dump_root):
        os.makedirs(args.dump_root)
    if not os.path.exists(args.dump_root):
        os.makedirs(args.dump_root)

    global data_loader
    if args.dataset_name == 'kitti_odom':
        from kitti.kitti_odom_loader import kitti_odom_loader
        data_loader = kitti_odom_loader(args.dataset_dir,
                                        img_height=args.img_height,
                                        img_width=args.img_width,
                                        seq_length=args.seq_length)

    if args.dataset_name == 'kitti_raw_eigen':
        from kitti.kitti_raw_loader import kitti_raw_loader
        data_loader = kitti_raw_loader(args.dataset_dir,
                                       split='eigen',
                                       img_height=args.img_height,
                                       img_width=args.img_width,
                                       seq_length=args.seq_length,
                                       remove_static=args.remove_static)

    if args.dataset_name == 'kitti_raw_stereo':
        from kitti.kitti_raw_loader import kitti_raw_loader
        data_loader = kitti_raw_loader(args.dataset_dir,
                                       split='stereo',
                                       img_height=args.img_height,
                                       img_width=args.img_width,
                                       seq_length=args.seq_length,
                                       remove_static=args.remove_static)

    if args.dataset_name == 'cityscapes':
        from cityscapes.cityscapes_loader import cityscapes_loader
        data_loader = cityscapes_loader(args.dataset_dir,
                                        img_height=args.img_height,
                                        img_width=args.img_width,
                                        seq_length=args.seq_length)

    # Parallel(n_jobs=args.num_threads)(delayed(dump_example)(n) for n in range(data_loader.num_train))
    # Parallel(n_jobs=args.num_threads)(delayed(dump_test_example)(n) for n in range(data_loader.num_test))
    
    if args.dataset_name == 'kitti_odom':
        ## Dump relative pose
        # Parallel(n_jobs=args.num_threads)(delayed(dump_relative_pose)(n) for n in range(data_loader.num_train))

        ## make obd
        prepare_train_data_obd.create_obd(args.dataset_dir, args.dump_root)

        ## obd2flow
        # for n in range(data_loader.num_train):
        #     dump_obd2flow(n)
        pass

    # Split into train
    # Write train.txt on dump_root
    np.random.seed(8964)
    subfolders = os.listdir(args.dump_root)
    with open(os.path.join(args.dump_root, 'train.txt'), 'w') as txf:
        print(data_loader.train_seqs)
        for s in data_loader.train_seqs: #subfolders:
            if not os.path.isdir(args.dump_root + '/%02d' % s):
                continue
            imfiles = glob(os.path.join(args.dump_root, '%02d'%s, '*.jpg'))
            frame_ids = [os.path.basename(fi).split('.')[0] for fi in imfiles]
            for frame in frame_ids:
                    txf.write('%02d %s\n' % (s, frame))

    # Write "test_flow.txt" on dump_test_root
    # if platform.system() == 'Linux':
    #     test_seq = "%02d"%args.test_seq
    # elif platform.system() == 'Windows':
    #     test_seq = "%02d"%args.test_seq #args.test_seq
    # with open(os.path.join(args.dump_root, 'test_flow.txt'), 'w') as txf:
    #     data_list = [f for f in os.listdir(os.path.join(args.dump_root, test_seq)) if os.path.splitext(f)[1] == '.jpg']
    #     data_list.sort()
    #     for i in range(len(data_list)):
    #         txf.write("%s %s\n"%(test_seq, os.path.splitext(data_list[i])[0]))

    create_test_flow_text(args.dump_root, "09")
    create_test_flow_text(args.dump_root, "10")




main()



