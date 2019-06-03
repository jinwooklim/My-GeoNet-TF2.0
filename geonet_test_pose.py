from __future__ import division
import os
import math
import scipy.misc
import tensorflow as tf
import numpy as np
from glob import glob
from geonet_model import *
from kitti_eval.pose_evaluation_utils import dump_pose_seq_TUM


def test_pose(FLAGS):

    if not os.path.isdir(FLAGS['output_dir']):
        os.makedirs(FLAGS['output_dir'])

    geonet = GeoNet(FLAGS['num_scales'], FLAGS['num_source'], FLAGS['alpha_recon_image'])

    ##### load test frames #####
    seq_dir = os.path.join(FLAGS['dataset_dir'], 'sequences', '%.2d' % FLAGS['pose_test_seq'])
    img_dir = os.path.join(seq_dir, 'image_2')
    N = len(glob(img_dir + '/*.png'))
    test_frames = ['%.2d %.6d' % (FLAGS.pose_test_seq, n) for n in range(N)]

    ##### load time file #####
    with open(FLAGS['dataset_dir'] + '/sequences/%.2d/times.txt' % FLAGS['pose_test_seq'], 'r') as f:
        times = f.readlines()
    times = np.array([float(s[:-1]) for s in times])

    ##### Go! #####
    max_src_offset = (FLAGS['seq_length'] - 1) // 2
    checkpoint_path = os.path.join(FLAGS['init_ckpt_file'])
    geonet.load_weights(checkpoint_path)

    for tgt_idx in range(max_src_offset, N-max_src_offset, FLAGS['batch_size']):
        if (tgt_idx-max_src_offset) % 100 == 0:
            print('Progress: %d/%d' % (tgt_idx-max_src_offset, N))

        inputs = np.zeros((FLAGS['batch_size'], FLAGS['img_height'],
                 FLAGS['img_width'], 3*FLAGS['seq_length']), dtype=np.float32)

        for b in range(FLAGS['batch_size']):
            idx = tgt_idx + b
            if idx >= N-max_src_offset:
                break
            image_seq = load_image_sequence(FLAGS['dataset_dir'],
                                            test_frames,
                                            idx,
                                            FLAGS['seq_length'],
                                            FLAGS['img_height'],
                                            FLAGS['img_width'])
            inputs[b] = image_seq

        pred_poses = geonet.pose_net(inputs, training=False)
        pred_poses = pred_poses.numpy()
        # Insert the target pose [0, 0, 0, 0, 0, 0]
        pred_poses = np.insert(pred_poses, max_src_offset, np.zeros((1, 6)), axis=1)

        for b in range(FLAGS['batch_size']):
            idx = tgt_idx + b
            if idx >= N - max_src_offset:
                break
            pred_pose = pred_poses[b]
            curr_times = times[idx - max_src_offset:idx + max_src_offset + 1]
            out_file = FLAGS['output_dir'] + '%.6d.txt' % (idx - max_src_offset)
            dump_pose_seq_TUM(out_file, pred_pose, curr_times)


def load_image_sequence(dataset_dir,
                        frames,
                        tgt_idx,
                        seq_length,
                        img_height,
                        img_width):
    half_offset = int((seq_length - 1)/2)
    for o in range(-half_offset, half_offset+1):
        curr_idx = tgt_idx + o
        curr_drive, curr_frame_id = frames[curr_idx].split(' ')
        img_file = os.path.join(
            dataset_dir, 'sequences', '%s/image_2/%s.png' % (curr_drive, curr_frame_id))
        curr_img = scipy.misc.imread(img_file)
        curr_img = scipy.misc.imresize(curr_img, (img_height, img_width))
        if o == -half_offset:
            image_seq = curr_img
        elif o == 0:
            image_seq = np.dstack((curr_img, image_seq))
        else:
            image_seq = np.dstack((image_seq, curr_img))
    return image_seq
