import tensorflow as tf
import os


class DataLoader(object):
    def __init__(self, FLAGS=None):
        self.FLAGS = FLAGS

        @tf.function
        def _parse_function(img_filename, cam_filename):
            tgt_image, src_image_stack = self.unpack_image_sequence(_img_parse_function(img_filename), self.FLAGS['img_height'], self.FLAGS['img_width'], self.FLAGS['num_source'])
            intrinsics = _cam_parse_function(cam_filename)
            return src_image_stack, tgt_image, intrinsics

        @tf.function
        def _img_parse_function(filename):
            image_string = tf.io.read_file(filename)
            image_decoded = tf.image.decode_image(image_string, dtype=tf.uint8, channels=3)
            image_decoded.set_shape([None, None, 3])
            return image_decoded

        @tf.function
        def _cam_parse_function(filename):
            text_string = tf.io.read_file(filename)
            rec_def = [1.] * 9
            text_decoded = tf.io.decode_csv(text_string, rec_def)
            text_decoded = tf.reshape(text_decoded, [3, 3])
            text_decoded = tf.cast(text_decoded, dtype=tf.float32)
            return text_decoded

        # Load the list of training files into queues
        file_list = self.format_file_list(self.FLAGS['dataset_dir'], 'train')
        img_filenames = tf.constant(file_list['image_file_list'])
        cam_filenames = tf.constant(file_list['cam_file_list'])

        self.dataset = tf.data.Dataset.from_tensor_slices((img_filenames, cam_filenames))
        self.dataset = self.dataset.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
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