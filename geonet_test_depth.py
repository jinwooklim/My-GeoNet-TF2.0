from __future__ import division
import tensorflow as tf
import numpy as np
import os
import PIL.Image as pil
from geonet_model import *


def test_depth(FLAGS):
    ##### load testing list #####
    with open('data/kitti/test_files_%s.txt'%'eigen', 'r') as f:
        test_files = f.readlines()
        test_files = [FLAGS['dataset_dir'] + t[:-1] for t in test_files]
    if not os.path.exists(FLAGS['output_dir']):
        os.makedirs(FLAGS['output_dir'])

    geonet = GeoNet(FLAGS['num_scales'], FLAGS['num_source'], FLAGS['alpha_recon_image'])

    checkpoint_path = os.path.join(FLAGS['init_ckpt_file'])
    geonet.load_weights(checkpoint_path)

    ##### Go #####
    pred_all = []
    for t in range(0, len(test_files), FLAGS['batch_size']):
        if t % 100 == 0:
            print('processing: %d/%d' % (t, len(test_files)))
        inputs = np.zeros(
            (FLAGS['batch_size'], FLAGS['img_height'], FLAGS['img_width'], 3),
            dtype=np.float32)

        for b in range(FLAGS['batch_size']):
            idx = t + b
            if idx >= len(test_files):
                break
            mixed_path = os.path.normpath(test_files[idx])
            fh = open(mixed_path, 'rb')
            raw_im = pil.open(fh)
            scaled_im = raw_im.resize((FLAGS['img_width'], FLAGS['img_height']), pil.ANTIALIAS)
            inputs[b] = np.array(scaled_im)

        pred = geonet.disp_net(inputs, training=False)
        pred_np = pred[0].numpy()
        pred_np_b = pred_np[b, :, :, 0]
        for b in range(FLAGS['batch_size']):
            idx = t + b
            if idx >= len(test_files):
                break
            pred_all.append(pred_np_b)

    np.save(FLAGS['output_dir'] + '/' + os.path.basename(FLAGS['init_ckpt_file']), pred_all)
