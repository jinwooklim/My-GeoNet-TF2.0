from __future__ import absolute_import, division, print_function, unicode_literals
import os
import time
import pprint
from data_loader import DataLoader
from geonet_model import *
import cv2
import sys
sys.path.insert(0, './kitti_eval/flow_tool/')
import flowlib as fl

opt = {'mode': "train_rigid",
       'dataset_dir': "E:\\all_dataset\\KITTI_dump",
       'init_ckpt_file': "",
       'batch_size': 4,
       'img_height': 128,
       'img_width': 416,
       'seq_length': 3,
       'num_source': 2,
       'num_scales': 4,
       'num_map_threads': tf.data.experimental.AUTOTUNE,
       'shuffle_buffer_size': 2048,  # Lager is good
       'prefetch_buffer_size': tf.data.experimental.AUTOTUNE,

       'checkpoint_dir': "./checkpoint/",
       'summary_dir': "./summary/",
       'learning_rate': 0.0002,
       'max_to_keep': 10,
       'max_steps': 600000,
       'save_ckpt_freq': 5000,
       'alpha_recon_image': 0.85,

       'scale_normalize': False,
       'rigid_warp_weight': 1.0,
       'disp_smooth_weight': 0.5,

       'useobd': False,
       }


def train(opt):
    pp = pprint.PrettyPrinter()
    pp.pprint(opt)

    if not os.path.exists(opt['checkpoint_dir']):
        os.makedirs(opt['checkpoint_dir'])

    data_loader = DataLoader(opt)
    adm_optimizer = tf.optimizers.Adam(opt['learning_rate'], 0.9)
    geonet = GeoNet(opt)

    ckpt = tf.train.Checkpoint(optimizer=adm_optimizer, net=geonet)
    ckpt_manager = tf.train.CheckpointManager(ckpt, opt['checkpoint_dir'], max_to_keep=opt['max_to_keep'])
    summary_writer = tf.summary.create_file_writer(opt['summary_dir'])

    with summary_writer.as_default():
        start_time = time.time()
        for step in range(opt['max_steps']):
            with tf.GradientTape() as tape:
                src_image_stack, tgt_image, intrinsics = data_loader.load_train_batch()
                tgt_image_pyramid, src_image_concat_pyramid, fwd_rigid_error_pyramid, bwd_rigid_error_pyramid, pred_poses, pred_disp, pred_depth, fwd_rigid_warp_pyramid, bwd_rigid_warp_pyramid, fwd_rigid_flow_pyramid, bwd_rigid_flow_pyramid = geonet(
                    [tgt_image, src_image_stack, intrinsics], training=True)
                loss = losses(opt['mode'], opt['num_scales'], opt['num_source'], opt['rigid_warp_weight'],
                              opt['disp_smooth_weight'],
                              tgt_image_pyramid, src_image_concat_pyramid, fwd_rigid_error_pyramid, bwd_rigid_error_pyramid,
                              pred_disp)
                gradients = tape.gradient(loss, geonet.trainable_variables)
                adm_optimizer.apply_gradients(zip(gradients, geonet.trainable_variables))
                if (step % 100 == 0):
                    time_per_iter = (time.time() - start_time) / 100
                    start_time = time.time()
                    print('Iteration: [%7d] | Time: %4.4fs/iter | Loss: %.3f'% (step, time_per_iter, loss))

                if (step % 100 == 0):
                    # Summary
                    # loss
                    tf.summary.scalar('loss', loss, step=step)

                    # input images
                    tf.summary.image('tgt_image', tgt_image_pyramid[0]/255.0, step=step)
                    tf.summary.image('src_image', src_image_concat_pyramid[0]/255.0, step=step)

                    # pred_disp
                    tf.summary.image('pred_depth', pred_depth[0], step=step)

                    # rigid_ward
                    tf.summary.image('fwd_rigid_warp_img', fwd_rigid_warp_pyramid[0]/255.0, step=step)
                    tf.summary.image('bwd_rigid_warp_img', bwd_rigid_warp_pyramid[0]/255.0, step=step)

                    # pred_flow
                    color_fwd_flow = fl.flow_to_image(fwd_rigid_flow_pyramid[0][0,:,:,:].numpy())
                    color_fwd_flow = cv2.cvtColor(color_fwd_flow, cv2.COLOR_RGB2BGR)
                    color_fwd_flow = tf.expand_dims(color_fwd_flow, axis=0)
                    tf.summary.image('color_fwd_flow', color_fwd_flow, step=step)
                    color_bwd_flow = fl.flow_to_image(bwd_rigid_flow_pyramid[0][0,:,:,:].numpy())
                    color_bwd_flow = cv2.cvtColor(color_bwd_flow, cv2.COLOR_RGB2BGR)
                    color_bwd_flow = tf.expand_dims(color_bwd_flow, axis=0)
                    tf.summary.image('color_bwd_flow', color_bwd_flow, step=step)

                # if (step % opt['save_ckpt_freq'] == 0) and step > 0:
                if(step % opt['save_ckpt_freq'] == 0):
                    save_path = ckpt_manager.save()
                    print("Saved checkpoint for step {}: {}".format(step, save_path))


if __name__ == "__main__":
    if opt['mode'] == 'train_rigid':
        train(opt)
