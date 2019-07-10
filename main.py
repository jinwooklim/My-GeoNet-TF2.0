from __future__ import absolute_import, division, print_function, unicode_literals
import os
import time
import pprint
import argparse
from data_loader import DataLoader
from geonet_model import *
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


@tf.function
def batch_data_augmentation(FLAGS, src_image_stack, tgt_image, intrinsics):
    @tf.function
    def data_augmentation(im, intrinsics, out_h, out_w):
        # Random scaling
        @tf.function
        def random_scaling(im, intrinsics):
            batch_size, in_h, in_w, _ = im.get_shape().as_list()
            scaling = tf.random.uniform([2], 1, 1.15)
            x_scaling = scaling[0]
            y_scaling = scaling[1]
            out_h = tf.cast(in_h * y_scaling, dtype=tf.int32)
            out_w = tf.cast(in_w * x_scaling, dtype=tf.int32)
            im = tf.image.resize(im, [out_h, out_w])
            fx = intrinsics[:, 0, 0] * x_scaling
            fy = intrinsics[:, 1, 1] * y_scaling
            cx = intrinsics[:, 0, 2] * x_scaling
            cy = intrinsics[:, 1, 2] * y_scaling
            intrinsics = make_intrinsics_matrix(fx, fy, cx, cy)
            return im, intrinsics

        # Random cropping
        @tf.function
        def random_cropping(im, intrinsics, out_h, out_w):
            # batch_size, in_h, in_w, _ = im.get_shape().as_list()
            batch_size, in_h, in_w, _ = tf.unstack(tf.shape(im))
            offset_y = tf.random.uniform([1], 0, in_h - out_h + 1, dtype=tf.int32)[0]
            offset_x = tf.random.uniform([1], 0, in_w - out_w + 1, dtype=tf.int32)[0]
            im = tf.image.crop_to_bounding_box(
                im, offset_y, offset_x, out_h, out_w)
            fx = intrinsics[:, 0, 0]
            fy = intrinsics[:, 1, 1]
            cx = intrinsics[:, 0, 2] - tf.cast(offset_x, dtype=tf.float32)
            cy = intrinsics[:, 1, 2] - tf.cast(offset_y, dtype=tf.float32)
            intrinsics = make_intrinsics_matrix(fx, fy, cx, cy)
            return im, intrinsics

        # Random coloring
        @tf.function
        def random_coloring(im):
            batch_size, in_h, in_w, in_c = im.get_shape().as_list()
            im_f = tf.image.convert_image_dtype(im, tf.float32)

            # randomly shift gamma
            random_gamma = tf.random.uniform([], 0.8, 1.2)
            im_aug = im_f ** random_gamma

            # randomly shift brightness
            random_brightness = tf.random.uniform([], 0.5, 2.0)
            im_aug = im_aug * random_brightness

            # randomly shift color
            random_colors = tf.random.uniform([in_c], 0.8, 1.2)
            white = tf.ones([batch_size, in_h, in_w])
            color_image = tf.stack([white * random_colors[i] for i in range(in_c)], axis=3)
            im_aug *= color_image

            # saturate
            im_aug = tf.clip_by_value(im_aug, 0, 1)

            im_aug = tf.image.convert_image_dtype(im_aug, tf.uint8)

            return im_aug

        im, intrinsics = random_scaling(im, intrinsics)
        im, intrinsics = random_cropping(im, intrinsics, out_h, out_w)
        im = tf.cast(im, dtype=tf.uint8)
        do_augment = tf.random.uniform([], 0, 1)
        im = tf.cond(do_augment > 0.5, lambda: random_coloring(im), lambda: im)
        return im, intrinsics

    @tf.function
    def make_intrinsics_matrix(fx, fy, cx, cy):
        # Assumes batch input
        batch_size = fx.get_shape().as_list()[0]
        zeros = tf.zeros_like(fx)
        r1 = tf.stack([fx, zeros, cx], axis=1)
        r2 = tf.stack([zeros, fy, cy], axis=1)
        r3 = tf.constant([0., 0., 1.], shape=[1, 3])
        r3 = tf.tile(r3, [batch_size, 1])
        intrinsics = tf.stack([r1, r2, r3], axis=1)
        return intrinsics

    @tf.function
    def get_multi_scale_intrinsics(intrinsics, num_scales):
        intrinsics_mscale = []
        # Scale the intrinsics accordingly for each scale
        for s in range(num_scales):
            fx = intrinsics[:, 0, 0] / (2 ** s)
            fy = intrinsics[:, 1, 1] / (2 ** s)
            cx = intrinsics[:, 0, 2] / (2 ** s)
            cy = intrinsics[:, 1, 2] / (2 ** s)
            intrinsics_mscale.append(
                make_intrinsics_matrix(fx, fy, cx, cy))
        intrinsics_mscale = tf.stack(intrinsics_mscale, axis=1)
        return intrinsics_mscale

    image_all = tf.concat([tgt_image, src_image_stack], axis=3)
    image_all, intrinsics = data_augmentation(image_all, intrinsics, FLAGS['img_height'], FLAGS['img_width'])
    tgt_image = image_all[:, :, :, :3]
    src_image_stack = image_all[:, :, :, 3:]
    intrinsics = get_multi_scale_intrinsics(intrinsics, FLAGS['num_scales'])
    return src_image_stack, tgt_image, intrinsics


@tf.function
def deprocess_image(image):
    # Assuming input image is float32
    image = (image + 1.)/2.
    return tf.image.convert_image_dtype(image, dtype=tf.uint8)


def gray2rgb(im, cmap='plasma'):
    cmap = plt.get_cmap(cmap)
    result_img = cmap(im.astype(np.float32))
    if result_img.shape[2] > 3:
        result_img = np.delete(result_img, 3, 2)
    return result_img


def normalize_depth_for_display(depth, pc=95, crop_percent=0, normalizer=None,
                                cmap='plasma'):
    """Converts a depth map to an RGB image."""
    # Convert to disparity.
    disp = 1.0 / (depth + 1e-6)
    if normalizer is not None:
        disp /= normalizer
    else:
        disp /= (np.percentile(disp, pc) + 1e-6)
        disp = np.clip(disp, 0, 1)
        disp = gray2rgb(disp, cmap=cmap)
        keep_h = int(disp.shape[0] * (1 - crop_percent))
        disp = disp[:keep_h]

    disp = tf.squeeze(disp)
    return disp


def train():
    if not (os.path.exists(FLAGS['checkpoint_dir']) and FLAGS['mode'] == 'train_rigid'):
        os.makedirs(FLAGS['checkpoint_dir'])
    else:
        if len(os.listdir(FLAGS['checkpoint_dir'])) > 0:
            print("Notice : Please remove exist checkpoints")
            exit()

    seed = 8964
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    data_loader = DataLoader(FLAGS)
    adm_optimizer = tf.optimizers.Adam(FLAGS['learning_rate'], 0.9)
    geonet = GeoNet(FLAGS)
    if FLAGS['mode'] == 'train_flow':
        checkpoint_path = os.path.join(FLAGS['init_ckpt_file'])
        geonet.load_weights(checkpoint_path)

    summary_writer = tf.summary.create_file_writer(FLAGS['summary_dir'])

    with summary_writer.as_default():
        start_time = time.time()
        for step in range(FLAGS['max_steps']):
            src_image_stack, tgt_image, intrinsics = next(data_loader.iter)

            # Data augmentation
            src_image_stack, tgt_image, intrinsics = batch_data_augmentation(FLAGS, src_image_stack, tgt_image, intrinsics)

            with tf.GradientTape() as tape:
                if FLAGS['mode'] == 'train_rigid':
                    tgt_image_pyramid, src_image_concat_pyramid, \
                    pred_disp, pred_depth, pred_poses, \
                    fwd_rigid_error_pyramid, bwd_rigid_error_pyramid, \
                    fwd_rigid_warp_pyramid, bwd_rigid_warp_pyramid, \
                    fwd_rigid_flow_pyramid, bwd_rigid_flow_pyramid = geonet(
                        [tgt_image, src_image_stack, intrinsics], training=True)

                    loss = rigid_losses(FLAGS['num_scales'], FLAGS['num_source'],
                                        FLAGS['rigid_warp_weight'], FLAGS['disp_smooth_weight'],
                                        tgt_image_pyramid, src_image_concat_pyramid,
                                        fwd_rigid_error_pyramid, bwd_rigid_error_pyramid,
                                        pred_disp)

                elif FLAGS['mode'] == 'train_flow':
                    tgt_image_pyramid, src_image_concat_pyramid,\
                    fwd_full_flow_pyramid, bwd_full_flow_pyramid, \
                    fwd_full_error_pyramid, bwd_full_error_pyramid, \
                    noc_masks_tgt, noc_masks_src, \
                    tgt_image_tile_pyramid, src_image_concat_pyramid, \
                    fwd_flow_diff_pyramid, bwd_flow_diff_pyramid = geonet([tgt_image, src_image_stack, intrinsics],
                                                                          training=True)

                    loss = flow_losses(FLAGS['num_scales'], FLAGS['num_source'],
                                       FLAGS['flow_warp_weight'], FLAGS['flow_consistency_weight'],
                                       FLAGS['flow_smooth_weight'],
                                       fwd_full_flow_pyramid, bwd_full_flow_pyramid,
                                       fwd_full_error_pyramid, bwd_full_error_pyramid,
                                       noc_masks_tgt, noc_masks_src,
                                       tgt_image_tile_pyramid, src_image_concat_pyramid,
                                       fwd_flow_diff_pyramid, bwd_flow_diff_pyramid)

                gradients = tape.gradient(loss, geonet.trainable_variables)
                adm_optimizer.apply_gradients(zip(gradients, geonet.trainable_variables))

            if (step % 100 == 0):
                time_per_iter = (time.time() - start_time) / 100
                start_time = time.time()
                print('Iteration: [%7d] | Time: %4.4fs/iter | Loss: %.3f'% (step, time_per_iter, loss))

            if (step % FLAGS['save_summary_freq'] == 0):
                # Summary
                # loss
                tf.summary.scalar('loss', loss, step=step)

                # input images
                tf.summary.image('tgt_image', deprocess_image(tgt_image_pyramid[0])/255, step=step) # [4, 128, 416, 3]
                tf.summary.image('src_image', deprocess_image(src_image_concat_pyramid[0])/255, step=step) # [8, 128, 416, 3]

                if FLAGS['mode'] == 'train_rigid':
                    # pred_depth # [12, 128, 416, 1] x num_scales=3
                    norm_pred_depth = normalize_depth_for_display(pred_depth[0].numpy())
                    tf.summary.image('norm_pred_depth', norm_pred_depth, step=step)

                    # rigid_warp_img
                    tf.summary.image('fwd_rigid_warp_img', deprocess_image(fwd_rigid_warp_pyramid[0])/255, step=step)  # [8, 128, 416, 3]
                    tf.summary.image('bwd_rigid_warp_img', deprocess_image(bwd_rigid_warp_pyramid[0])/255, step=step) # [8, 128, 416, 3]

                    # pred rigid flow
                    color_fwd_flow_list = []
                    color_bwd_flow_list = []
                    for i in range(FLAGS['batch_size']):
                        color_fwd_flow = fl.flow_to_image(fwd_rigid_flow_pyramid[0][i,:,:,:].numpy()) # [8, 128, 416, 2]
                        color_fwd_flow = cv2.cvtColor(color_fwd_flow, cv2.COLOR_RGB2BGR)
                        # color_fwd_flow = tf.expand_dims(color_fwd_flow, axis=0)
                        color_fwd_flow_list.append(color_fwd_flow)
                        color_bwd_flow = fl.flow_to_image(bwd_rigid_flow_pyramid[0][i,:,:,:].numpy()) # [8, 128, 416, 2]
                        color_bwd_flow = cv2.cvtColor(color_bwd_flow, cv2.COLOR_RGB2BGR)
                        # color_bwd_flow = tf.expand_dims(color_bwd_flow, axis=0)
                        color_bwd_flow_list.append(color_bwd_flow)
                    color_fwd_flow = np.array(color_fwd_flow_list)
                    # color_fwd_flow = tf.squeeze(color_fwd_flow)
                    color_bwd_flow = np.array(color_bwd_flow_list)
                    # color_bwd_flow = tf.squeeze(color_bwd_flow)
                    tf.summary.image('color_fwd_flow', color_fwd_flow, step=step)
                    tf.summary.image('color_bwd_flow', color_bwd_flow, step=step)

                elif FLAGS['mode'] == 'train_flow':
                    # pred full flow
                    color_fwd_flow_list = []
                    color_bwd_flow_list = []
                    for i in range(FLAGS['batch_size']):
                        color_fwd_flow = fl.flow_to_image(fwd_full_flow_pyramid[0][i,:,:,:].numpy()) # [8, 128, 416, 2]
                        color_fwd_flow = cv2.cvtColor(color_fwd_flow, cv2.COLOR_RGB2BGR)
                        # color_fwd_flow = tf.expand_dims(color_fwd_flow, axis=0)
                        color_fwd_flow_list.append(color_fwd_flow)
                        color_bwd_flow = fl.flow_to_image(bwd_full_flow_pyramid[0][i,:,:,:].numpy()) # [8, 128, 416, 2]
                        color_bwd_flow = cv2.cvtColor(color_bwd_flow, cv2.COLOR_RGB2BGR)
                        # color_bwd_flow = tf.expand_dims(color_bwd_flow, axis=0)
                        color_bwd_flow_list.append(color_bwd_flow)
                    color_fwd_flow = np.array(color_fwd_flow_list)
                    # color_fwd_flow = tf.squeeze(color_fwd_flow)
                    color_bwd_flow = np.array(color_bwd_flow_list)
                    # color_bwd_flow = tf.squeeze(color_bwd_flow)
                    tf.summary.image('color_fwd_flow', color_fwd_flow, step=step)
                    tf.summary.image('color_bwd_flow', color_bwd_flow, step=step)

            if(step % FLAGS['save_ckpt_freq'] == 0):
                save_path = FLAGS['checkpoint_dir'] + "iter-" + str(step)
                geonet.save_weights(save_path)
                print("Saved checkpoint for step {}: {}".format(step, FLAGS['checkpoint_dir']))


if __name__ == "__main__":
    pp = pprint.PrettyPrinter()
    pp.pprint(FLAGS)

    if FLAGS['mode'] == "train_rigid":
        train()
    elif FLAGS['mode'] == 'train_flow':
        train()
    elif FLAGS['mode'] == "test_pose":
        test_pose(FLAGS)
    elif FLAGS['mode'] == "test_depth":
        test_depth(FLAGS)

