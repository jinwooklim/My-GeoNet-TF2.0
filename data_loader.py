import tensorflow as tf
import os


class DataLoader(object):
    def __init__(self, FLAGS=None):
        self.FLAGS = FLAGS

        # Load the list of training files into queues
        file_list = self.format_file_list(self.FLAGS['dataset_dir'], 'train')
        img_filenames = tf.constant(file_list['image_file_list'])
        cam_filenames = tf.constant(file_list['cam_file_list'])

        self.dataset = tf.data.Dataset.from_tensor_slices((img_filenames, cam_filenames))
        self.dataset = self.dataset.map(self._parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.dataset = self.dataset.shuffle(buffer_size=self.FLAGS['shuffle_buffer_size'])
        self.dataset = self.dataset.repeat()
        self.dataset = self.dataset.batch(self.FLAGS['batch_size'])
        self.dataset = self.dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        self.iter = iter(self.dataset)

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

    @tf.function
    def get_next(self):
        src_image_stack, tgt_image, intrinsics = next(self.iter)
        return self.batch_data_augmentation(src_image_stack, tgt_image, intrinsics)

    @tf.function
    def _parse_function(self, img_filename, cam_filename):
        tgt_image, src_image_stack = self.unpack_image_sequence(self._img_parse_function(img_filename), self.FLAGS['img_height'], self.FLAGS['img_width'], self.FLAGS['num_source'])
        intrinsics = self._cam_parse_function(cam_filename)
        return src_image_stack, tgt_image, intrinsics

    @tf.function
    def _img_parse_function(self, filename):
        image_string = tf.io.read_file(filename)
        image_decoded = tf.image.decode_image(image_string, dtype=tf.uint8, channels=3)
        image_decoded.set_shape([None, None, 3])
        return image_decoded

    @tf.function
    def _cam_parse_function(self, filename):
        text_string = tf.io.read_file(filename)
        rec_def = [1.] * 9
        text_decoded = tf.io.decode_csv(text_string, rec_def)
        text_decoded = tf.reshape(text_decoded, [3, 3])
        text_decoded = tf.cast(text_decoded, dtype=tf.float32)
        return text_decoded

    @tf.function
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

    @tf.function
    def make_intrinsics_matrix(self, fx, fy, cx, cy):
        # Assumes batch input
        batch_size = self.FLAGS['batch_size'] # fx.get_shape().as_list()[0]
        zeros = tf.zeros_like(fx)
        r1 = tf.stack([fx, zeros, cx], axis=1)
        r2 = tf.stack([zeros, fy, cy], axis=1)
        r3 = tf.constant([0., 0., 1.], shape=[1, 3])
        r3 = tf.tile(r3, [batch_size, 1])
        intrinsics = tf.stack([r1, r2, r3], axis=1)
        return intrinsics

    @tf.function
    def get_multi_scale_intrinsics(self, intrinsics, num_scales):
        intrinsics_mscale = []
        # Scale the intrinsics accordingly for each scale
        for s in range(num_scales):
            fx = intrinsics[:, 0, 0] / (2 ** s)
            fy = intrinsics[:, 1, 1] / (2 ** s)
            cx = intrinsics[:, 0, 2] / (2 ** s)
            cy = intrinsics[:, 1, 2] / (2 ** s)
            intrinsics_mscale.append(
                self.make_intrinsics_matrix(fx, fy, cx, cy))
        intrinsics_mscale = tf.stack(intrinsics_mscale, axis=1)
        return intrinsics_mscale

    # Random scaling
    @tf.function
    def random_scaling(self, im, intrinsics):
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
        intrinsics = self.make_intrinsics_matrix(fx, fy, cx, cy)
        return im, intrinsics

    # Random cropping
    @tf.function
    def random_cropping(self, im, intrinsics, out_h, out_w):
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
        intrinsics = self.make_intrinsics_matrix(fx, fy, cx, cy)
        return im, intrinsics

    # Random coloring
    @tf.function
    def random_coloring(self, im):
        batch_size = self.FLAGS['batch_size']
        _, in_h, in_w, in_c = im.get_shape().as_list()
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

    @tf.function
    def data_augmentation(self, im, intrinsics, out_h, out_w):
        im, intrinsics = self.random_scaling(im, intrinsics)
        im, intrinsics = self.random_cropping(im, intrinsics, out_h, out_w)
        im = tf.cast(im, dtype=tf.uint8)
        do_augment = tf.random.uniform([], 0, 1)
        im = tf.cond(do_augment > 0.5, lambda: self.random_coloring(im), lambda: im)
        return im, intrinsics

    @tf.function
    def batch_data_augmentation(self, src_image_stack, tgt_image, intrinsics):
        image_all = tf.concat([tgt_image, src_image_stack], axis=3)
        image_all, intrinsics = self.data_augmentation(image_all, intrinsics, self.FLAGS['img_height'], self.FLAGS['img_width'])
        tgt_image = image_all[:, :, :, :3]
        src_image_stack = image_all[:, :, :, 3:]
        intrinsics = self.get_multi_scale_intrinsics(intrinsics, self.FLAGS['num_scales'])
        return src_image_stack, tgt_image, intrinsics