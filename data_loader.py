import numpy as np
import tensorflow as tf
import os
import random
import utils2


class DataLoader(object):
    def __init__(self, opt=None):
        self.opt = opt

        def _parse_function_with_obd(img_filename, cam_filename, obd_filename):
            if opt['useobd']:
                image_seq = _img_parse_function(img_filename)
                tgt_image, src_image_stack = self.unpack_image_sequence(image_seq, opt['img_height'], opt['img_width'], opt['num_source'])
                intrinsics = _cam_parse_function(cam_filename)
                obd = _obd_parse_function(obd_filename)

                # Make Warped Image using OBD
                obd_warped_image_stack = self.get_obd_warped_image(image_seq, opt['img_height'], opt['img_width'], opt['num_source'], intrinsics, obd)

                return src_image_stack, tgt_image, intrinsics, obd_warped_image_stack

        def _parse_function(img_filename, cam_filename):
                tgt_image, src_image_stack = self.unpack_image_sequence(_img_parse_function(img_filename), opt['img_height'], opt['img_width'], opt['num_source'])
                intrinsics = _cam_parse_function(cam_filename)
                return src_image_stack, tgt_image, intrinsics

        def _img_parse_function(filename):
            image_string = tf.io.read_file(filename)
            image_decoded = tf.image.decode_image(image_string, dtype=tf.uint8)
            return image_decoded

        def _cam_parse_function(filename):
            text_string = tf.io.read_file(filename)
            rec_def = [1.] * 9
            text_decoded = tf.io.decode_csv(text_string, rec_def)
            text_decoded = tf.reshape(text_decoded, [3, 3])
            text_decoded = tf.cast(text_decoded, dtype=tf.float32)
            return text_decoded

        def _obd_parse_function(filename):
            text_string = tf.io.read_file(filename)
            rec_def = [1.] * (2*(opt['seq_length']-1))
            text_decoded = tf.io.decode_csv(text_string, rec_def)
            text_decoded = tf.reshape(text_decoded, [(opt['seq_length']-1), 2])
            text_decoded = tf.cast(text_decoded, dtype=tf.float32)
            return text_decoded

        # Load the list of training files into queues
        file_list = self.format_file_list(opt['dataset_dir'], 'train')
        img_filenames = tf.constant(file_list['image_file_list'])
        cam_filenames = tf.constant(file_list['cam_file_list'])
        if opt['useobd']:
            obd_filenames = tf.constant(file_list['obd_file_list'])

        if opt['useobd']:
            self.dataset = tf.data.Dataset.from_tensor_slices((img_filenames, cam_filenames, obd_filenames))
            self.dataset = self.dataset.map(_parse_function_with_obd)
        else:
            self.dataset = tf.data.Dataset.from_tensor_slices((img_filenames, cam_filenames))
            self.dataset = self.dataset.map(_parse_function, num_parallel_calls=opt['num_map_threads'])
        self.dataset = self.dataset.repeat(opt['max_epoch'])
        self.dataset = self.dataset.batch(opt['batch_size'])
        self.dataset = self.dataset.shuffle(buffer_size=opt['shuffle_buffer_size'])
        self.iter = iter(self.dataset)

    def load_train_batch(self):
        """Load a batch of training instances.
        """

        # next_image_batch = next(iter(self.dataset))
        # return next_image_batch

        opt = self.opt
        if opt['useobd']:
            # src_image_stack, tgt_image, intrinsics, obd_warped_image_stack = next(iter(self.dataset))
            src_image_stack, tgt_image, intrinsics, obd_warped_image_stack = next(self.iter)

            # Data Augmentation
            image_all = tf.concat([tgt_image, src_image_stack], axis=3)
            image_all, intrinsics = self.data_augmentation(image_all, intrinsics, opt['img_height'], opt['img_width'])
            tgt_image = image_all[:, :, :, :3]
            src_image_stack = image_all[:, :, :, 3:]
            intrinsics = self.get_multi_scale_intrinsics(intrinsics, opt['num_scales'])

            return src_image_stack, tgt_image, intrinsics, obd_warped_image_stack
        else:
            # src_image_stack, tgt_image, intrinsics = next(iter(self.dataset))
            src_image_stack, tgt_image, intrinsics = next(self.iter)

            # # Data Augmentation
            image_all = tf.concat([tgt_image, src_image_stack], axis=3)
            image_all, intrinsics = self.data_augmentation(image_all, intrinsics, opt['img_height'], opt['img_width'])
            tgt_image = image_all[:, :, :, :3]
            src_image_stack = image_all[:, :, :, 3:]
            intrinsics = self.get_multi_scale_intrinsics(intrinsics, opt['num_scales'])

            src_image_stack = tf.cast(src_image_stack, dtype=tf.float32)
            tgt_image = tf.cast(tgt_image, dtype=tf.float32)
            intrinsics = tf.cast(intrinsics, dtype=tf.float32)

            return src_image_stack, tgt_image, intrinsics

    def make_intrinsics_matrix(self, fx, fy, cx, cy):
        # Assumes batch input
        batch_size = fx.get_shape().as_list()[0]
        zeros = tf.zeros_like(fx)
        r1 = tf.stack([fx, zeros, cx], axis=1)
        r2 = tf.stack([zeros, fy, cy], axis=1)
        r3 = tf.constant([0.,0.,1.], shape=[1, 3])
        r3 = tf.tile(r3, [batch_size, 1])
        intrinsics = tf.stack([r1, r2, r3], axis=1)
        return intrinsics

    def data_augmentation(self, im, intrinsics, out_h, out_w):
        # Random scaling
        def random_scaling(im, intrinsics):
            batch_size, in_h, in_w, _ = im.get_shape().as_list()
            scaling = tf.random.uniform([2], 1, 1.15)
            x_scaling = scaling[0]
            y_scaling = scaling[1]
            out_h = tf.cast(in_h * y_scaling, dtype=tf.int32)
            out_w = tf.cast(in_w * x_scaling, dtype=tf.int32)
            ###
            # im = tf.image.resize_area(im, [out_h, out_w])
            im = tf.image.resize(im, [out_h, out_w])
            ###
            fx = intrinsics[:,0,0] * x_scaling
            fy = intrinsics[:,1,1] * y_scaling
            cx = intrinsics[:,0,2] * x_scaling
            cy = intrinsics[:,1,2] * y_scaling
            intrinsics = self.make_intrinsics_matrix(fx, fy, cx, cy)
            return im, intrinsics

        # Random cropping
        def random_cropping(im, intrinsics, out_h, out_w):
            # batch_size, in_h, in_w, _ = im.get_shape().as_list()
            batch_size, in_h, in_w, _ = tf.unstack(tf.shape(im))
            offset_y = tf.random.uniform([1], 0, in_h - out_h + 1, dtype=tf.int32)[0]
            offset_x = tf.random.uniform([1], 0, in_w - out_w + 1, dtype=tf.int32)[0]
            im = tf.image.crop_to_bounding_box(
                im, offset_y, offset_x, out_h, out_w)
            fx = intrinsics[:,0,0]
            fy = intrinsics[:,1,1]
            cx = intrinsics[:,0,2] - tf.cast(offset_x, dtype=tf.float32)
            cy = intrinsics[:,1,2] - tf.cast(offset_y, dtype=tf.float32)
            intrinsics = self.make_intrinsics_matrix(fx, fy, cx, cy)
            return im, intrinsics

        # Random coloring
        def random_coloring(im):
            batch_size, in_h, in_w, in_c = im.get_shape().as_list()
            im_f = tf.image.convert_image_dtype(im, tf.float32)

            # randomly shift gamma
            random_gamma = tf.random.uniform([], 0.8, 1.2)
            im_aug  = im_f  ** random_gamma

            # randomly shift brightness
            random_brightness = tf.random.uniform([], 0.5, 2.0)
            im_aug  =  im_aug * random_brightness

            # randomly shift color
            random_colors = tf.random.uniform([in_c], 0.8, 1.2)
            white = tf.ones([batch_size, in_h, in_w])
            color_image = tf.stack([white * random_colors[i] for i in range(in_c)], axis=3)
            im_aug  *= color_image

            # saturate
            im_aug  = tf.clip_by_value(im_aug,  0, 1)

            im_aug = tf.image.convert_image_dtype(im_aug, tf.uint8)

            return im_aug
        im, intrinsics = random_scaling(im, intrinsics)
        im, intrinsics = random_cropping(im, intrinsics, out_h, out_w)
        im = tf.cast(im, dtype=tf.uint8)
        do_augment  = tf.random.uniform([], 0, 1)
        im = tf.cond(do_augment > 0.5, lambda: random_coloring(im), lambda: im)
        return im, intrinsics

    def format_file_list(self, data_root, split):
        with open(data_root + '/%s.txt' % split, 'r') as f:
            frames = f.readlines()
        subfolders = [x.split(' ')[0] for x in frames]
        frame_ids = [x.split(' ')[1][:-1] for x in frames]
        image_file_list = [os.path.join(data_root, subfolders[i],
            frame_ids[i] + '.jpg') for i in range(len(frames))]
        cam_file_list = [os.path.join(data_root, subfolders[i],
            frame_ids[i] + '_cam.txt') for i in range(len(frames))]
        obd_file_list = [os.path.join(data_root, subfolders[i],
            frame_ids[i] + '_obd.txt') for i in range(len(frames))]
        all_list = {}
        all_list['image_file_list'] = image_file_list
        all_list['cam_file_list'] = cam_file_list
        all_list['obd_file_list'] = obd_file_list
        return all_list

    def unpack_image_sequence(self, image_seq, img_height, img_width, num_source):
        # Assuming the center image is the target frame
        tgt_start_idx = int(img_width * (num_source//2))
        tgt_image = tf.slice(image_seq,
                             [0, tgt_start_idx, 0],
                             [-1, img_width, -1])
        # Source frames before the target frame
        src_image_1 = tf.slice(image_seq,
                               [0, 0, 0],
                               [-1, int(img_width * (num_source//2)), -1])
        # Source frames after the target frame
        src_image_2 = tf.slice(image_seq,
                               [0, int(tgt_start_idx + img_width), 0],
                               [-1, int(img_width * (num_source//2)), -1])
        src_image_seq = tf.concat([src_image_1, src_image_2], axis=1)
        # Stack source frames along the color channels (i.e. [H, W, N*3])
        src_image_stack = tf.concat([tf.slice(src_image_seq,
                                    [0, i*img_width, 0],
                                    [-1, img_width, -1])
                                    for i in range(num_source)], axis=2)
        src_image_stack.set_shape([img_height,
                                   img_width,
                                   num_source * 3])
        tgt_image.set_shape([img_height, img_width, 3])
        return tgt_image, src_image_stack

    def get_multi_scale_intrinsics(self, intrinsics, num_scales):
        intrinsics_mscale = []
        # Scale the intrinsics accordingly for each scale
        for s in range(num_scales):
            fx = intrinsics[:,0,0]/(2 ** s)
            fy = intrinsics[:,1,1]/(2 ** s)
            cx = intrinsics[:,0,2]/(2 ** s)
            cy = intrinsics[:,1,2]/(2 ** s)
            intrinsics_mscale.append(
                self.make_intrinsics_matrix(fx, fy, cx, cy))
        intrinsics_mscale = tf.stack(intrinsics_mscale, axis=1)
        return intrinsics_mscale

    def get_obd_warped_image(self, image_seq, img_height, img_width, num_source, intrinsic, obd):# Assuming the center image is the target frame
        tgt_start_idx = int(img_width * (num_source//2))
        tgt_image = tf.slice(image_seq,
                             [0, tgt_start_idx, 0],
                             [-1, img_width, -1])
        tgt_image.set_shape([img_height, img_width, 3])
        # Source frames before the target frame
        src_image_1 = tf.slice(image_seq,
                               [0, 0, 0],
                               [-1, int(img_width * (num_source//2)), -1])
        src_image_1.set_shape([img_height, img_width, 3])
        # Source frames after the target frame
        src_image_2 = tf.slice(image_seq,
                               [0, int(tgt_start_idx + img_width), 0],
                               [-1, int(img_width * (num_source//2)), -1])
        src_image_2.set_shape([img_height, img_width, 3])

        # Convert OBD to Pose
        # expand_dims for batch (compatibility of previous code)
        tgt_image = tf.expand_dims(tgt_image, axis=0)
        src_image_1 = tf.expand_dims(src_image_1, axis=0)
        src_image_2 = tf.expand_dims(src_image_2, axis=0)
        intrinsic = tf.expand_dims(intrinsic, axis=0)

        # Src 1 , bwd
        temp_pose_01 = utils2.get_pose_by_obd_tf(obd[0])
        obd_rigid_flow_01 = utils2.compute_rigid_flow(tgt_image, -temp_pose_01, intrinsic, reverse_pose=True)
        obd_warped_img_01 = utils2.flow_warp(src_image_1, obd_rigid_flow_01)

        # Src 2 , fwd
        temp_pose_02 = utils2.get_pose_by_obd_tf(obd[1])
        obd_rigid_flow_02 = utils2.compute_rigid_flow(tgt_image, temp_pose_02, intrinsic, reverse_pose=False)
        obd_warped_img_02 = utils2.flow_warp(src_image_2, obd_rigid_flow_02)

        res = tf.concat([obd_warped_img_01, obd_warped_img_02], axis=3)
        res = tf.cast(res, tf.uint8) # return to float -> int
        res = tf.squeeze(res)
        return res