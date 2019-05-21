from geonet_nets import *
from utils import *
import tensorflow as tf

@tf.function
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


@tf.function
def spatial_normalize(disp):
    _, curr_h, curr_w, curr_c = disp.get_shape().as_list()
    disp_mean = tf.reduce_mean(disp, axis=[1,2,3], keep_dims=True)
    disp_mean = tf.tile(disp_mean, [1, curr_h, curr_w, curr_c])
    return disp/disp_mean


@tf.function
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


@tf.function
def image_similarity(alpha_recon_image, x, y):
    return alpha_recon_image * SSIM(x, y) + (1.0 - alpha_recon_image) * tf.abs(x-y)


@tf.function
def build_rigid_flow_warping(num_scales, num_source, alpha_recon_image, src_image_concat_pyramid, tgt_image_tile_pyramid, pred_depth, intrinsics, pred_poses):
    bs = tf.shape(intrinsics)[0] # bs : 4
    # build rigid flow (fwd: tgt->src, bwd: src->tgt)
    fwd_rigid_flow_pyramid = []
    bwd_rigid_flow_pyramid = []
    for s in range(num_scales):
        for i in range(num_source):
            fwd_rigid_flow = compute_rigid_flow(tf.squeeze(pred_depth[s][:bs], axis=3),
                             pred_poses[:,i,:], intrinsics[:,s,:,:], False)
            bwd_rigid_flow = compute_rigid_flow(tf.squeeze(pred_depth[s][bs*(i+1):bs*(i+2)], axis=3),
                             pred_poses[:,i,:], intrinsics[:,s,:,:], True)
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


@tf.function
def gradient_x(img):
    gx = img[:,:,:-1,:] - img[:,:,1:,:]
    return gx


@tf.function
def gradient_y(img):
    gy = img[:,:-1,:,:] - img[:,1:,:,:]
    return gy


@tf.function
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


@tf.function
def losses(mode, num_scales, num_source, rigid_warp_weight, disp_smooth_weight,
           tgt_image_pyramid, src_image_concat_pyramid, fwd_rigid_error_pyramid, bwd_rigid_error_pyramid, pred_disp):

    total_loss = 0

    for s in range(num_scales):
        # rigid_warp_loss
        rigid_warp_loss = 0
        disp_smooth_loss = 0
        if mode == 'train_rigid' and rigid_warp_weight > 0:
            rigid_warp_loss = rigid_warp_weight*(num_source/2) * \
                            (tf.reduce_mean(fwd_rigid_error_pyramid[s]) + \
                             tf.reduce_mean(bwd_rigid_error_pyramid[s]))

        # disp_smooth_loss
        if mode == 'train_rigid' and disp_smooth_weight > 0:
            disp_smooth_loss = disp_smooth_weight/(2**s) * compute_smooth_loss(pred_disp[s],
                            tf.concat([tgt_image_pyramid[s], src_image_concat_pyramid[s]], axis=0))

        if mode == 'train_rigid':
            total_loss = (rigid_warp_loss + disp_smooth_loss)

    return total_loss


class GeoNet(Model):
    def __init__(self, num_scales, num_source, alpha_recon_image):
        super(GeoNet, self).__init__()
        self.num_scales = num_scales
        self.num_source = num_source
        self.alpha_recon_image = alpha_recon_image

        self.pose_net = PoseNet(num_source=self.num_source)
        self.disp_net = DispNet()

    def call(self, inputs, training=None, mask=None):
        # Data preprocess
        tgt_image = inputs[0]
        src_image_stack = inputs[1]
        intrinsics = inputs[2]

        tgt_image_pyramid = scale_pyramid(tgt_image, self.num_scales)
        tgt_image_tile_pyramid = [tf.tile(img, [self.num_source, 1, 1, 1]) for img in tgt_image_pyramid]

        # src images concated along batch dimension
        if src_image_stack != None:
            src_image_concat = tf.concat([src_image_stack[:,:,:,3*i:3*(i+1)] \
                                    for i in range(self.num_source)], axis=0)
            src_image_concat_pyramid = scale_pyramid(src_image_concat, self.num_scales)

        ### DepthNet part
        # build dispnet_inputs
        if training == False: #self.opt['mode'] == 'test_depth':
            # for test_depth mode we only predict the depth of the target image
            dispnet_inputs = tgt_image
        else:
            # multiple depth predictions; tgt: disp[:bs,:,:,:] src.i: disp[bs*(i+1):bs*(i+2),:,:,:]
            dispnet_inputs = tgt_image
            for i in range(self.num_source):
                dispnet_inputs = tf.concat([dispnet_inputs, src_image_stack[:, :, :, 3 * i:3 * (i + 1)]], axis=0)

        # DepthNet Forward
        pred_disp = self.disp_net(dispnet_inputs, training=training)

        # DepthNet result
        pred_depth = [1. / d for d in pred_disp]

        ### PoseNet part
        # build posenet_inputs
        posenet_inputs = tf.concat([tgt_image, src_image_stack], axis=3)

        # build posenet
        pred_poses = self.pose_net(posenet_inputs, training=training)

        # print("111 : ", pred_poses) # (4, 2, 6)
        fwd_rigid_error_pyramid, bwd_rigid_error_pyramid, fwd_rigid_warp_pyramid, bwd_rigid_warp_pyramid, fwd_rigid_flow_pyramid, bwd_rigid_flow_pyramid = build_rigid_flow_warping(
                                                                                    self.num_scales,
                                                                                    self.num_source,
                                                                                    self.alpha_recon_image,
                                                                                    src_image_concat_pyramid,
                                                                                    tgt_image_tile_pyramid,
                                                                                    pred_depth,
                                                                                    intrinsics,
                                                                                    pred_poses)

        return [tgt_image_pyramid, src_image_concat_pyramid, pred_disp, pred_depth, pred_poses, fwd_rigid_error_pyramid, bwd_rigid_error_pyramid, fwd_rigid_warp_pyramid, bwd_rigid_warp_pyramid, fwd_rigid_flow_pyramid, bwd_rigid_flow_pyramid]