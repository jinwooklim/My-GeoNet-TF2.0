from geonet_nets import *
from utils import *
import tensorflow as tf


def scale_pyramid(img, num_scales):
    if img == None:
        return None
    else:
        scaled_imgs = [img]
        _, h, w, _ = img.get_shape().as_list()
        for i in range(num_scales - 1):
            ratio = 2 ** (i + 1)
            nh = int(h / ratio)
            nw = int(w / ratio)
            scaled_imgs.append(tf.image.resize(img, [nh, nw]))
    return scaled_imgs


def spatial_normalize(disp):
    _, curr_h, curr_w, curr_c = disp.get_shape().as_list()
    disp_mean = tf.reduce_mean(disp, axis=[1, 2, 3], keep_dims=True)
    disp_mean = tf.tile(disp_mean, [1, curr_h, curr_w, curr_c])
    return disp/disp_mean


def SSIM(x, y):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = tf.nn.avg_pool2d(x, 3, 1, 'SAME')
    mu_y = tf.nn.avg_pool2d(y, 3, 1, 'SAME')

    sigma_x = tf.nn.avg_pool2d(x**2, 3, 1, 'SAME') - mu_x ** 2
    sigma_y = tf.nn.avg_pool2d(y**2, 3, 1, 'SAME') - mu_y ** 2
    sigma_xy = tf.nn.avg_pool2d(x*y, 3, 1, 'SAME') - mu_x * mu_y

    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

    SSIM = SSIM_n / SSIM_d

    return tf.clip_by_value((1 - SSIM) / 2, 0, 1)


def image_similarity(alpha_recon_image, x, y):
    return (alpha_recon_image * SSIM(x, y)) + ((1.0 - alpha_recon_image) * tf.abs(x-y))


def L2_norm(x, axis=3, keepdims=True):
    curr_offset = 1e-10
    l2_norm = tf.norm(tf.abs(x) + curr_offset, axis=axis, keepdims=keepdims)
    return l2_norm


'''
TODO : Convert tf.function -> tf.keras.Model
'''
def build_rigid_flow_warping(num_scales, num_source, alpha_recon_image,
                             src_image_concat_pyramid, tgt_image_tile_pyramid,
                             pred_depth, intrinsics, pred_poses):
    bs = tf.shape(intrinsics)[0] # bs : 4
    # build rigid flow (fwd: tgt->src, bwd: src->tgt)
    fwd_rigid_flow_pyramid = []
    bwd_rigid_flow_pyramid = []
    for s in range(num_scales):
        for i in range(num_source):
            fwd_rigid_flow = compute_rigid_flow(tf.squeeze(pred_depth[s][:bs], axis=3),
                             pred_poses[:, i, :], intrinsics[:, s, :, :], False)
            bwd_rigid_flow = compute_rigid_flow(tf.squeeze(pred_depth[s][bs*(i+1):bs*(i+2)], axis=3),
                             pred_poses[:, i, :], intrinsics[:, s, :, :], True)
            if not i:
                fwd_rigid_flow_concat = fwd_rigid_flow
                bwd_rigid_flow_concat = bwd_rigid_flow
            else:
                fwd_rigid_flow_concat = tf.concat([fwd_rigid_flow_concat, fwd_rigid_flow], axis=0)
                bwd_rigid_flow_concat = tf.concat([bwd_rigid_flow_concat, bwd_rigid_flow], axis=0)
        fwd_rigid_flow_pyramid.append(fwd_rigid_flow_concat)
        bwd_rigid_flow_pyramid.append(bwd_rigid_flow_concat)

    # warping by rigid flow
    fwd_rigid_warp_pyramid = [flow_warp(src_image_concat_pyramid[s], fwd_rigid_flow_pyramid[s]) \
                                  for s in range(num_scales)]
    bwd_rigid_warp_pyramid = [flow_warp(tgt_image_tile_pyramid[s], bwd_rigid_flow_pyramid[s]) \
                                  for s in range(num_scales)]

    # compute reconstruction error
    fwd_rigid_error_pyramid = [image_similarity(alpha_recon_image, fwd_rigid_warp_pyramid[s], tgt_image_tile_pyramid[s]) for s in range(num_scales)]
    bwd_rigid_error_pyramid = [image_similarity(alpha_recon_image, bwd_rigid_warp_pyramid[s], src_image_concat_pyramid[s]) for s in range(num_scales)]

    return fwd_rigid_error_pyramid, bwd_rigid_error_pyramid, fwd_rigid_warp_pyramid, bwd_rigid_warp_pyramid, fwd_rigid_flow_pyramid, bwd_rigid_flow_pyramid


def gradient_x(img):
    gx = img[:, :, :-1, :] - img[:, :, 1:, :]
    return gx


def gradient_y(img):
    gy = img[:, :-1, :, :] - img[:, 1:, :, :]
    return gy


def compute_smooth_loss(disp, img):
    disp_gradients_x = gradient_x(disp)
    disp_gradients_y = gradient_y(disp)

    image_gradients_x = gradient_x(img)
    image_gradients_y = gradient_y(img)

    weights_x = tf.exp(-tf.reduce_mean(tf.abs(image_gradients_x), 3, keepdims=True))
    weights_y = tf.exp(-tf.reduce_mean(tf.abs(image_gradients_y), 3, keepdims=True))

    smoothness_x = disp_gradients_x * weights_x
    smoothness_y = disp_gradients_y * weights_y

    return tf.reduce_mean(tf.abs(smoothness_x)) + tf.reduce_mean(tf.abs(smoothness_y))


def build_full_flow_warping(num_scales, alpha_recon_image,
                            src_image_concat_pyramid, tgt_image_tile_pyramid,
                            fwd_full_flow_pyramid, bwd_full_flow_pyramid):
    # warping by full flow
    fwd_full_warp_pyramid = [flow_warp(src_image_concat_pyramid[s], fwd_full_flow_pyramid[s])
                             for s in range(num_scales)]

    bwd_full_warp_pyramid = [flow_warp(tgt_image_tile_pyramid[s], bwd_full_flow_pyramid[s])
                             for s in range(num_scales)]

    # compute reconstruction error
    fwd_full_error_pyramid = [image_similarity(alpha_recon_image, fwd_full_warp_pyramid[s], tgt_image_tile_pyramid[s])
                              for s in range(num_scales)]
    bwd_full_error_pyramid = [image_similarity(alpha_recon_image, bwd_full_warp_pyramid[s], src_image_concat_pyramid[s])
                              for s in range(num_scales)]

    return fwd_full_warp_pyramid, bwd_full_warp_pyramid, fwd_full_error_pyramid, bwd_full_error_pyramid


def build_flow_consistency(num_scales, flow_consistency_alpha, flow_consistency_beta, fwd_full_flow_pyramid, bwd_full_flow_pyramid):
    # warp pyramid full flow
    bwd2fwd_flow_pyramid = [flow_warp(bwd_full_flow_pyramid[s], fwd_full_flow_pyramid[s]) \
                                 for s in range(num_scales)]
    fwd2bwd_flow_pyramid = [flow_warp(fwd_full_flow_pyramid[s], bwd_full_flow_pyramid[s]) \
                                 for s in range(num_scales)]

    # calculate flow consistency
    fwd_flow_diff_pyramid = [tf.abs(bwd2fwd_flow_pyramid[s] + fwd_full_flow_pyramid[s]) for s in
                                  range(num_scales)]
    bwd_flow_diff_pyramid = [tf.abs(fwd2bwd_flow_pyramid[s] + bwd_full_flow_pyramid[s]) for s in
                                  range(num_scales)]

    # build flow consistency condition
    fwd_consist_bound = [flow_consistency_beta * L2_norm(fwd_full_flow_pyramid[s]) * 2 ** s for s in
                              range(num_scales)]
    bwd_consist_bound = [flow_consistency_beta * L2_norm(bwd_full_flow_pyramid[s]) * 2 ** s for s in
                              range(num_scales)]
    fwd_consist_bound = [tf.stop_gradient(tf.maximum(v, flow_consistency_alpha)) for v in
                              fwd_consist_bound]
    bwd_consist_bound = [tf.stop_gradient(tf.maximum(v, flow_consistency_alpha)) for v in
                              bwd_consist_bound]

    # build flow consistency mask
    noc_masks_src = [tf.cast(tf.less(L2_norm(bwd_flow_diff_pyramid[s]) * 2 ** s,
                                          bwd_consist_bound[s]), tf.float32) for s in range(num_scales)]
    noc_masks_tgt = [tf.cast(tf.less(L2_norm(fwd_flow_diff_pyramid[s]) * 2 ** s,
                                          fwd_consist_bound[s]), tf.float32) for s in range(num_scales)]

    return noc_masks_tgt, noc_masks_src, fwd_flow_diff_pyramid, bwd_flow_diff_pyramid


def rigid_losses(num_scales, num_source, rigid_warp_weight, disp_smooth_weight,
                 tgt_image_pyramid, src_image_concat_pyramid,
                 fwd_rigid_error_pyramid, bwd_rigid_error_pyramid,
                 pred_disp):

    rigid_warp_loss = 0
    disp_smooth_loss = 0

    for s in range(num_scales):
        # rigid_warp_loss
        if rigid_warp_weight > 0:
            rigid_warp_loss += rigid_warp_weight * num_source/2 * \
                            (tf.reduce_mean(fwd_rigid_error_pyramid[s]) + \
                             tf.reduce_mean(bwd_rigid_error_pyramid[s]))

        # disp_smooth_loss
        if disp_smooth_weight > 0:
            disp_smooth_loss += disp_smooth_weight/(2**s) * compute_smooth_loss(pred_disp[s],
                            tf.concat([tgt_image_pyramid[s], src_image_concat_pyramid[s]], axis=0))

    total_loss = 0
    total_loss += (rigid_warp_loss + disp_smooth_loss)

    return total_loss


def flow_losses(num_scales, num_source, flow_warp_weight, flow_consistency_weight, flow_smooth_weight,
                fwd_full_flow_pyramid, bwd_full_flow_pyramid,
                fwd_full_error_pyramid, bwd_full_error_pyramid,
                noc_masks_tgt, noc_masks_src,
                tgt_image_tile_pyramid, src_image_concat_pyramid,
                fwd_flow_diff_pyramid, bwd_flow_diff_pyramid):
    flow_warp_loss = 0
    flow_smooth_loss = 0
    flow_consistency_loss = 0

    for s in range(num_scales):
        # flow_warp_loss
        if flow_warp_weight > 0:
            if flow_consistency_weight == 0:
                flow_warp_loss += flow_warp_weight * num_source / 2 * \
                                  (tf.reduce_mean(fwd_full_error_pyramid[s]) + tf.reduce_mean(
                                      bwd_full_error_pyramid[s]))
            else:
                flow_warp_loss += flow_warp_weight * num_source / 2 * \
                                  (tf.reduce_sum(
                                      tf.reduce_mean(fwd_full_error_pyramid[s], axis=3, keepdims=True) * \
                                      noc_masks_tgt[s]) / tf.reduce_sum(noc_masks_tgt[s]) + \
                                   tf.reduce_sum(
                                       tf.reduce_mean(bwd_full_error_pyramid[s], axis=3, keepdims=True) * \
                                       noc_masks_src[s]) / tf.reduce_sum(noc_masks_src[s]))

        # flow_smooth_loss
        if flow_warp_weight > 0:
            flow_smooth_loss += flow_smooth_weight / (2 ** (s + 1)) * \
                                (compute_flow_smooth_loss(fwd_full_flow_pyramid[s],
                                                               tgt_image_tile_pyramid[s]) +
                                 compute_flow_smooth_loss(bwd_full_flow_pyramid[s],
                                                               src_image_concat_pyramid[s]))

        # flow_consistency_loss
        if flow_consistency_weight > 0:
            flow_consistency_loss += flow_consistency_weight / 2 * \
                                     (tf.reduce_sum(
                                         tf.reduce_mean(fwd_flow_diff_pyramid[s], axis=3, keepdims=True) * \
                                         noc_masks_tgt[s]) / tf.reduce_sum(noc_masks_tgt[s]) + \
                                      tf.reduce_sum(
                                          tf.reduce_mean(bwd_flow_diff_pyramid[s], axis=3, keepdims=True) * \
                                          noc_masks_src[s]) / tf.reduce_sum(noc_masks_src[s]))

        total_loss = 0
        total_loss += (flow_warp_loss + flow_smooth_loss + flow_consistency_loss)

        return total_loss


def preprocess_image(image):
    # Assuming input image is uint8
    if image == None:
        return None
    else:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        return image * 2. - 1.


def compute_flow_smooth_loss(flow, img):
    smoothness = 0
    for i in range(2):
        smoothness += compute_smooth_loss(tf.expand_dims(flow[:, :, :, i], -1), img)
    return smoothness/2


class GeoNet(Model):
    def __init__(self, FLAGS):
        super(GeoNet, self).__init__()
        self.FLAGS = FLAGS

        self.pose_net = PoseNet(num_source=self.FLAGS['num_source'])
        self.disp_net = DispNet()
        self.flow_net = FlowNet()

    def call(self, inputs, training=None, mask=None):
        '''
        # Input data preprocessing part
        '''
        tgt_image = preprocess_image(inputs[0])
        src_image_stack = preprocess_image(inputs[1])
        intrinsics = inputs[2]

        tgt_image_pyramid = scale_pyramid(tgt_image, self.FLAGS['num_scales'])
        tgt_image_tile_pyramid = [tf.tile(img, [self.FLAGS['num_source'], 1, 1, 1]) for img in tgt_image_pyramid]

        # src images concated along batch dimension
        if src_image_stack != None:
            src_image_concat = tf.concat([src_image_stack[:, :, :, 3*i:3*(i+1)] \
                                    for i in range(self.FLAGS['num_source'])], axis=0)
            src_image_concat_pyramid = scale_pyramid(src_image_concat, self.FLAGS['num_scales'])

        if self.FLAGS['mode'] == 'train_rigid':
            '''
            # DepthNet part
            '''
            # build dispnet_inputs
            if training == False: #self.opt['mode'] == 'test_depth':
                # for test_depth mode we only predict the depth of the target image
                dispnet_inputs = tgt_image
            else:
                # multiple depth predictions; tgt: disp[:bs,:,:,:] src.i: disp[bs*(i+1):bs*(i+2),:,:,:]
                dispnet_inputs = tgt_image
                for i in range(self.FLAGS['num_source']):
                    dispnet_inputs = tf.concat([dispnet_inputs, src_image_stack[:, :, :, 3 * i:3 * (i + 1)]], axis=0)

            # DepthNet Forward
            pred_disp = self.disp_net(dispnet_inputs, training=training)

            # DepthNet result
            pred_depth = [1. / d for d in pred_disp]

            '''
            # PoseNet part
            '''
            # build posenet_inputs
            posenet_inputs = tf.concat([tgt_image, src_image_stack], axis=3)

            # build posenet
            pred_poses = self.pose_net(posenet_inputs, training=training)

            '''
            # Computing Rigid Flow part
            '''
            fwd_rigid_error_pyramid, bwd_rigid_error_pyramid, \
            fwd_rigid_warp_pyramid, bwd_rigid_warp_pyramid, \
            fwd_rigid_flow_pyramid, bwd_rigid_flow_pyramid = build_rigid_flow_warping(
                self.FLAGS['num_scales'], self.FLAGS['num_source'], self.FLAGS['alpha_recon_image'],
                src_image_concat_pyramid, tgt_image_tile_pyramid, pred_depth, intrinsics, pred_poses)

            return tgt_image_pyramid, src_image_concat_pyramid, \
                   pred_disp, pred_depth, pred_poses, \
                   fwd_rigid_error_pyramid, bwd_rigid_error_pyramid, \
                   fwd_rigid_warp_pyramid, bwd_rigid_warp_pyramid, \
                   fwd_rigid_flow_pyramid, bwd_rigid_flow_pyramid

        elif self.FLAGS['mode'] == 'train_flow':
            '''
            # DepthNet part
            '''
            # build dispnet_inputs
            if training == False: #self.opt['mode'] == 'test_depth':
                # for test_depth mode we only predict the depth of the target image
                dispnet_inputs = tgt_image
            else:
                # multiple depth predictions; tgt: disp[:bs,:,:,:] src.i: disp[bs*(i+1):bs*(i+2),:,:,:]
                dispnet_inputs = tgt_image
                for i in range(self.FLAGS['num_source']):
                    dispnet_inputs = tf.concat([dispnet_inputs, src_image_stack[:, :, :, 3 * i:3 * (i + 1)]], axis=0)

            # DepthNet Forward
            pred_disp = self.disp_net(dispnet_inputs, training=False)

            # DepthNet result
            pred_depth = [1. / d for d in pred_disp]

            '''
            # PoseNet part
            '''
            # build posenet_inputs
            posenet_inputs = tf.concat([tgt_image, src_image_stack], axis=3)

            # build posenet
            pred_poses = self.pose_net(posenet_inputs, training=False)

            '''
            # Computing Rigid Flow part
            '''
            fwd_rigid_error_pyramid, bwd_rigid_error_pyramid, fwd_rigid_warp_pyramid, bwd_rigid_warp_pyramid, fwd_rigid_flow_pyramid, bwd_rigid_flow_pyramid = build_rigid_flow_warping(
                                                                                        self.FLAGS['num_scales'], self.FLAGS['num_source'], self.FLAGS['alpha_recon_image'],
                                                                                        src_image_concat_pyramid,
                                                                                        tgt_image_tile_pyramid,
                                                                                        pred_depth,
                                                                                        intrinsics,
                                                                                        pred_poses)

            '''
            # FlowNet part
            '''
            # build flownet_inputs
            fwd_flownet_inputs = tf.concat([tgt_image_tile_pyramid[0], src_image_concat_pyramid[0]], axis=3)
            bwd_flownet_inputs = tf.concat([src_image_concat_pyramid[0], tgt_image_tile_pyramid[0]], axis=3)
            if self.FLAGS['flownet_type'] == 'residual':
                fwd_flownet_inputs = tf.concat([fwd_flownet_inputs,
                                                     fwd_rigid_warp_pyramid[0],
                                                     fwd_rigid_flow_pyramid[0],
                                                     L2_norm(fwd_rigid_error_pyramid[0])], axis=3)
                bwd_flownet_inputs = tf.concat([bwd_flownet_inputs,
                                                     bwd_rigid_warp_pyramid[0],
                                                     bwd_rigid_flow_pyramid[0],
                                                     L2_norm(bwd_rigid_error_pyramid[0])], axis=3)
            flownet_inputs = tf.concat([fwd_flownet_inputs, bwd_flownet_inputs], axis=0)

            # build flownet
            pred_flow = self.flow_net(flownet_inputs)

            # unnormalize pyramid flow back into pixel metric
            for s in range(self.FLAGS['num_scales']):
                curr_bs, curr_h, curr_w, _ = pred_flow[s].get_shape().as_list()
                scale_factor = tf.cast(tf.constant([curr_w, curr_h], shape=[1, 1, 1, 2]), 'float32')
                scale_factor = tf.tile(scale_factor, [curr_bs, curr_h, curr_w, 1])
                pred_flow[s] = pred_flow[s] * scale_factor

            # split forward/backward flows
            fwd_full_flow_pyramid = [pred_flow[s][:self.FLAGS['batch_size'] * self.FLAGS['num_source']] for s in
                                          range(self.FLAGS['num_scales'])]
            bwd_full_flow_pyramid = [pred_flow[s][self.FLAGS['batch_size'] * self.FLAGS['num_source']:] for s in
                                          range(self.FLAGS['num_scales'])]

            # residual flow postprocessing
            if self.FLAGS['flownet_type'] == 'residual':
                fwd_full_flow_pyramid = [fwd_full_flow_pyramid[s] + fwd_rigid_flow_pyramid[s] for s in
                                              range(self.FLAGS['num_scales'])]
                bwd_full_flow_pyramid = [bwd_full_flow_pyramid[s] + bwd_rigid_flow_pyramid[s] for s in
                                              range(self.FLAGS['num_scales'])]

            '''
            # Computing Residual Flow part
            '''
            fwd_full_warp_pyramid, bwd_full_warp_pyramid, \
            fwd_full_error_pyramid, bwd_full_error_pyramid = build_full_flow_warping(self.FLAGS['num_scales'],
                                                                                     self.FLAGS['alpha_recon_image'],
                                                                                     src_image_concat_pyramid,
                                                                                     tgt_image_tile_pyramid,
                                                                                     fwd_full_flow_pyramid,
                                                                                     bwd_full_flow_pyramid)

            if self.FLAGS['flow_consistency_weight'] > 0:
                noc_masks_tgt, noc_masks_src, \
                fwd_flow_diff_pyramid, bwd_flow_diff_pyramid = build_flow_consistency(
                    self.FLAGS['num_scales'],
                    self.FLAGS['flow_consistency_alpha'], self.FLAGS['flow_consistency_beta'],
                    fwd_full_flow_pyramid, bwd_full_flow_pyramid)

                return [tgt_image_pyramid, src_image_concat_pyramid,
                        fwd_full_flow_pyramid, bwd_full_flow_pyramid,
                        fwd_full_error_pyramid, bwd_full_error_pyramid,
                        noc_masks_tgt, noc_masks_src,
                        tgt_image_tile_pyramid, src_image_concat_pyramid,
                        fwd_flow_diff_pyramid, bwd_flow_diff_pyramid]

            else:
                return [tgt_image_pyramid, src_image_concat_pyramid,
                        fwd_full_flow_pyramid, bwd_full_flow_pyramid,
                        fwd_full_error_pyramid, bwd_full_error_pyramid,
                        tgt_image_tile_pyramid, src_image_concat_pyramid]


