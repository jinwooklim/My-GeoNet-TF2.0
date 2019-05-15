from __future__ import absolute_import, division, print_function, unicode_literals
import os
import time
import pprint
import argparse
from data_loader import DataLoader
from geonet_model import *
from geonet_test_pose import *
import matplotlib.pyplot as plt
import cv2
import sys
sys.path.insert(0, './kitti_eval/flow_tool/')
import flowlib as fl


parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='train_rigid', help='Mode of program')
parser.add_argument('--dataset_dir', type=str, default='E:\\all_dataset\\KITTI_dump', help='Path of dataset')
parser.add_argument('--shuffle_buffer_size', type=int, default=128, help='shuffle_buffer_size size')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
parser.add_argument('--img_height', type=int, default=128, help='img_height')
parser.add_argument('--img_width', type=int, default=416, help='img_width')
parser.add_argument('--seq_length', type=int, default=3, help='seq_length')
parser.add_argument('--num_source', type=int, default=2, help='num_source')
parser.add_argument('--num_scales', type=int, default=4, help='num_scales')

parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint/', help='Checkpoint dir')
parser.add_argument('--init_ckpt_file', type=str, default='ckpt-1', help='Ckpt name')
parser.add_argument('--save_ckpt_freq', type=int, default=10000)
parser.add_argument('--max_to_keep', type=int, default=3)
parser.add_argument('--summary_dir', type=str, default='./summary/', help='summary_dir')
parser.add_argument('--save_summary_freq', type=int, default=500)

parser.add_argument('--learning_rate', type=float, default=0.0002)
parser.add_argument('--max_steps', type=int, default=600000)
parser.add_argument('--alpha_recon_image', type=float, default=0.85)
parser.add_argument('--rigid_warp_weight', type=float, default=1.0)
parser.add_argument('--disp_smooth_weight', type=float, default=0.5)

parser.add_argument('--output_dir', type=str, default='./predictions/')
parser.add_argument('--pose_test_seq', type=int, default=9)

# FLAGS, unknown_args = parser.parse_known_args()
FLAGS = parser.parse_args()


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
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    else:
        if len(os.listdir(FLAGS.checkpoint_dir)) > 0:
            print("Notice : Please remove exist checkpoints")
            exit()

    data_loader = DataLoader(FLAGS)
    adm_optimizer = tf.optimizers.Adam(FLAGS.learning_rate, 0.9)
    geonet = GeoNet(FLAGS.num_scales, FLAGS.num_source, FLAGS.alpha_recon_image)

    ckpt = tf.train.Checkpoint(optimizer=adm_optimizer, net=geonet)
    ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.checkpoint_dir, max_to_keep=FLAGS.max_to_keep)
    summary_writer = tf.summary.create_file_writer(FLAGS.summary_dir)

    with summary_writer.as_default():
        start_time = time.time()
        for step in range(FLAGS.max_steps):
            with tf.GradientTape() as tape:
                src_image_stack, tgt_image, intrinsics = next(data_loader.iter)

                tgt_image_pyramid, src_image_concat_pyramid, pred_disp, pred_depth, pred_poses, fwd_rigid_error_pyramid, bwd_rigid_error_pyramid, fwd_rigid_warp_pyramid, bwd_rigid_warp_pyramid, fwd_rigid_flow_pyramid, bwd_rigid_flow_pyramid = geonet(
                    [tgt_image, src_image_stack, intrinsics], training=True)
                loss = losses(FLAGS.mode, FLAGS.num_scales, FLAGS.num_source, FLAGS.rigid_warp_weight,
                              FLAGS.disp_smooth_weight,
                              tgt_image_pyramid, src_image_concat_pyramid, fwd_rigid_error_pyramid, bwd_rigid_error_pyramid,
                              pred_disp)
                gradients = tape.gradient(loss, geonet.trainable_variables)
                adm_optimizer.apply_gradients(zip(gradients, geonet.trainable_variables))

                if (step % 100 == 0):
                    time_per_iter = (time.time() - start_time) / 100
                    start_time = time.time()
                    print('Iteration: [%7d] | Time: %4.4fs/iter | Loss: %.3f'% (step, time_per_iter, loss))

                if (step % FLAGS.save_summary_freq == 0):
                    # Summary
                    # loss
                    tf.summary.scalar('loss', loss, step=step)

                    # input images
                    tf.summary.image('tgt_image', tgt_image_pyramid[0]/255.0, step=step) # [4, 128, 416, 3]
                    tf.summary.image('src_image', src_image_concat_pyramid[0]/255.0, step=step) # [8, 128, 416, 3]

                    # pred_depth # [12, 128, 416, 1] x num_scales=3
                    norm_pred_depth = normalize_depth_for_display(pred_depth[0].numpy())
                    tf.summary.image('norm_pred_depth', norm_pred_depth, step=step)

                    # rigid_ward
                    tf.summary.image('fwd_rigid_warp_img', fwd_rigid_warp_pyramid[0]/255.0, step=step)  # [8, 128, 416, 3]
                    tf.summary.image('bwd_rigid_warp_img', bwd_rigid_warp_pyramid[0]/255.0, step=step) # [8, 128, 416, 3]

                    # pred_flow
                    color_fwd_flow_list = []
                    color_bwd_flow_list = []
                    for i in range(FLAGS.batch_size):
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

                # if (step % opt['save_ckpt_freq'] == 0) and step > 0:
                if(step % FLAGS.save_ckpt_freq == 0):
                    save_path = ckpt_manager.save()
                    print("Saved checkpoint for step {}: {}".format(step, save_path))


if __name__ == "__main__":
    pp = pprint.PrettyPrinter()
    pp.pprint(FLAGS)

    if FLAGS.mode == "train_rigid":
        train()
    elif FLAGS.mode == "test_pose":
        test_pose(FLAGS)

