from __future__ import absolute_import, division, print_function, unicode_literals
import os
import pprint
from data_loader import DataLoader
from geonet_model import *


def train(opt):
    pp = pprint.PrettyPrinter()
    pp.pprint(opt)

    if not os.path.exists(opt['checkpoint_dir']):
        os.makedirs(opt['checkpoint_dir'])

    data_loader = DataLoader(opt)
    model = GeoNet(opt)
    optimizer = tf.optimizers.Adam(opt['learning_rate'], 0.9)

    # ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
    # manager = tf.train.CheckpointManager(ckpt, opt['checkpoint_dir'],  max_to_keep=opt['max_to_keep'])

    for step in range(opt['max_steps']):
        src_image_stack, tgt_image, intrinsics = data_loader.load_train_batch()

        tgt_image_pyramid, src_image_concat_pyramid, fwd_rigid_error_pyramid, bwd_rigid_error_pyramid, pred_poses, pred_disp = model(
                [tgt_image, src_image_stack, intrinsics])

        with tf.GradientTape() as tape:
            loss = losses(opt['mode'], opt['num_scales'], opt['num_source'], opt['rigid_warp_weight'], opt['disp_smooth_weight'],
                      tgt_image_pyramid, src_image_concat_pyramid, fwd_rigid_error_pyramid, bwd_rigid_error_pyramid, pred_disp)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        if (step % 100):
            print("loss : ", loss)


if __name__ == "__main__":
    opt = {'mode': "train_rigid",
           'dataset_dir': "E:\\all_dataset\\KITTI_dump",
           'init_ckpt_file': "",
           'batch_size': 4,
           'img_height': 128,
           'img_width': 416,
           'seq_length': 3,
           'num_source': 2,
           'num_scales': 4,
           'buffer_size': 256,  # 2048

           'checkpoint_dir': "./checkpoint/",
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
    train(opt)
