import numpy as np
import cv2
import os

UNKNOWN_FLOW_THRESH= 1e7
# obdframe= [front wheel rps, steering wheel angle]

def get_relative_velosity(obd_frame):
    angle = -obd_frame[1] * 0.05 * np.pi / 180
    fps = 9.64764
    speed = obd_frame[0] * 1000000 / 3600 / fps # mm per frame
    speed_x = speed * np.sin(angle)
    speed_y = speed * np.cos(angle)
    return -speed_x, -speed_y
# image_size_odometry= [376,1241,3]
# image_size_raw = [374.1242.3]

def real_position_of_p(pixel_position,image_size,camera_height=1.65, vertical_AOV= 35, horizental_AOV= 90):
    distance = camera_height * 1 / np.tan(vertical_AOV / 2 * np.pi/180)
    # px : pixel_position[0] - width of image/ 2
    # py : pixel_position[1] - height of image/2
    px = pixel_position[0]-int(image_size[1]/2)
    py = pixel_position[1]-int(image_size[0]/2)
    ry = image_size[0]/2*distance/py
    rx = 2 * distance*np.tan(horizental_AOV/2 * np.pi/180)/image_size[1] * px * ry / distance
    rz = -camera_height
    return rx, ry, rz

def flow_to_image(flow):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))

    print("max flow: %.4f\nflow range:\nu = %.3f .. %.3f\nv = %.3f .. %.3f" % (maxrad, minu,maxu, minv, maxv))

    u = u/(maxrad + np.finfo(float).eps)
    v = v/(maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)

def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u**2+v**2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a+1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols+1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel,1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1-rad[idx]*(1-col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx)))

    return img

def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
    colorwheel[col:col+YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
    colorwheel[col:col+CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col+MR, 0] = 255

    return colorwheel

def get_pixelwise_relative_velosity(obdframe, pixel_position, image_size, camera_height=1.65, vertical_AOV=35, horizental_AOV= 90):
    distance = camera_height * 1 / np.tan(vertical_AOV / 2 * np.pi / 180)
    py = pixel_position[1] - int(image_size[0] / 2)
    dX_dt, dY_dt = get_relative_velosity(obdframe)
    R_X, R_Y, R_Z = real_position_of_p(pixel_position, image_size, camera_height=camera_height, vertical_AOV=vertical_AOV, horizental_AOV= horizental_AOV)
    dPy_dy = -image_size[0] / 2 * distance / (R_Y**2)
    dPy_dt = dPy_dy * dY_dt
    dPx_dx = image_size[0] * py / image_size[1] / distance
    dPx_dy = R_X * image_size[0] / image_size[1] / distance* dPy_dy
    dPx_dt = dPx_dx * dX_dt + dPx_dy * dY_dt
    return dPx_dt, dPy_dt

#x_point= 600
#y_point= 300
# obd_dir= './virtual_obd/00'
# def obd2flow(obd_dir, image_path,idx):
#     obd_path = os.path.join(obd_dir, os.listdir(obd_dir)[0])
#     read_obd = open(obd_path, 'r')
#     obdframe = read_obd.readlines()[idx].split()
#     obdframe[0] = float(obdframe[0])
#     obdframe[1] = float(obdframe[1])
#     # image_path = '../KITTI odometry/data_odometry_color/dataset/sequences/00/image_2/000000.png'
#     img = cv2.imread(image_path)
#     # img= cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#     img_size = np.shape(img)
#
#     flow_size = [img_size[0], img_size[1], 2]
#     flow = np.zeros(flow_size)
#
#     for y_index in range(int(img_size[0] / 2) + 1, img_size[0]):
#         for x_index in range(img_size[1]):
#             x_point = x_index
#             y_point = y_index
#             flow[y_point, x_point, 0], flow[y_point, x_point, 1] = get_pixelwise_relative_velosity(obdframe,
#                                                                                                    [x_point, y_point],
#                                                                                                    img_size)
#     return flow

def obd2flow(obd, img):
    obd[0] = float(obd[0])
    obd[1] = float(obd[1])

    img_size = np.shape(img)

    flow_size = [img_size[0], img_size[1], 2]
    flow = np.zeros(flow_size)

    for y_index in range(int(img_size[0] / 2) + 1, img_size[0]):
        for x_index in range(img_size[1]):
            x_point = x_index
            y_point = y_index
            flow[y_point, x_point, 0], flow[y_point, x_point, 1] = get_pixelwise_relative_velosity(obd,
                                                                                                   [x_point, y_point],
                                                                                                   img_size)
    return flow

def obd_to_flow(obd):
    print("func : ", np.shape(obd))
    flow_size = [128, 416, 2]
    flow = np.zeros(flow_size)

    for y_index in range(int(flow_size[0] / 2) + 1, flow_size[0]):
        for x_index in range(flow_size[1]):
            x_point = x_index
            y_point = y_index
            flow[y_point, x_point , 0], flow[y_point, x_point, 1] = get_pixelwise_relative_velosity(obd, [x_point, y_point],flow_size)

    return flow


def warp_image(im, flow):
    """
    Use optical flow to warp image to the next
    :param im: image to warp
    :param flow: optical flow
    :return: warped image
    """
    from scipy import interpolate
    image_height = im.shape[0]
    image_width = im.shape[1]
    flow_height = flow.shape[0]
    flow_width = flow.shape[1]
    n = image_height * image_width
    (iy, ix) = np.mgrid[0:image_height, 0:image_width]
    (fy, fx) = np.mgrid[0:flow_height, 0:flow_width]
    fx = fx.astype(np.float64)
    fy = fy.astype(np.float64)
    fx += flow[:,:,0]
    fy += flow[:,:,1]
    mask = np.logical_or(fx <0 , fx > flow_width)
    mask = np.logical_or(mask, fy < 0)
    mask = np.logical_or(mask, fy > flow_height)
    fx = np.minimum(np.maximum(fx, 0), flow_width)
    fy = np.minimum(np.maximum(fy, 0), flow_height)
    points = np.concatenate((ix.reshape(n,1), iy.reshape(n,1)), axis=1)
    xi = np.concatenate((fx.reshape(n, 1), fy.reshape(n,1)), axis=1)
    warp = np.zeros((image_height, image_width, im.shape[2]))
    for i in range(im.shape[2]):
        channel = im[:, :, i]
        values = channel.reshape(n, 1)
        new_channel = interpolate.griddata(points, values, xi, method='cubic')
        new_channel = np.reshape(new_channel, [flow_height, flow_width])
        new_channel[mask] = 1
        warp[:, :, i] = new_channel.astype(np.uint8)
    return warp.astype(np.uint8)


# def flow_warp(src_img, flow):
#   """ inverse warp a source image to the target image plane based on flow field
#   Args:
#     src_img: the source  image [batch, height_s, width_s, 3]
#     flow: target image to source image flow [batch, height_t, width_t, 2]
#   Returns:
#     Source image inverse warped to the target image plane [batch, height_t, width_t, 3]
#   """
#   height, width, = src_img.shape[0], src_img.shape[1]
#   temp = np.meshgrid(height, width)
#   print(np.shape(temp))
#   tgt_pixel_coords = np.transpose(temp, (0, 2, 3, 1))
#   src_pixel_coords = tgt_pixel_coords + flow
#   output_img = bilinear_sampler(src_img, src_pixel_coords)
#   return output_img
#
#
# def bilinear_sampler(imgs, coords):
#   """Construct a new image by bilinear sampling from the input image.
#
#   Points falling outside the source image boundary have value 0.
#
#   Args:
#     imgs: source image to be sampled from [batch, height_s, width_s, channels]
#     coords: coordinates of source pixels to sample from [batch, height_t,
#       width_t, 2]. height_t/width_t correspond to the dimensions of the output
#       image (don't need to be the same as height_s/width_s). The two channels
#       correspond to x and y coordinates respectively.
#   Returns:
#     A new sampled image [batch, height_t, width_t, channels]
#   """
#   def _repeat(x, n_repeats):
#     rep = np.transpose(
#         np.expand_dims(np.ones(shape=np.stack([
#             n_repeats,
#         ])), 1), [1, 0])
#     rep = np.cast(rep, 'float32')
#     x = np.matmul(np.reshape(x, (-1, 1)), rep)
#     return np.reshape(x, [-1])
#
#   with tf.name_scope('image_sampling'):
#     coords_x, coords_y = tf.split(coords, [1, 1], axis=3)
#     inp_size = imgs.get_shape()
#     coord_size = coords.get_shape()
#     out_size = coords.get_shape().as_list()
#     out_size[3] = imgs.get_shape().as_list()[3]
#
#     coords_x = tf.cast(coords_x, 'float32')
#     coords_y = tf.cast(coords_y, 'float32')
#
#     x0 = tf.floor(coords_x)
#     x1 = x0 + 1
#     y0 = tf.floor(coords_y)
#     y1 = y0 + 1
#
#     y_max = tf.cast(tf.shape(imgs)[1] - 1, 'float32')
#     x_max = tf.cast(tf.shape(imgs)[2] - 1, 'float32')
#     zero = tf.zeros([1], dtype='float32')
#
#     x0_safe = tf.clip_by_value(x0, zero, x_max)
#     y0_safe = tf.clip_by_value(y0, zero, y_max)
#     x1_safe = tf.clip_by_value(x1, zero, x_max)
#     y1_safe = tf.clip_by_value(y1, zero, y_max)
#
#     ## bilinear interp weights, with points outside the grid having weight 0
#     # wt_x0 = (x1 - coords_x) * tf.cast(tf.equal(x0, x0_safe), 'float32')
#     # wt_x1 = (coords_x - x0) * tf.cast(tf.equal(x1, x1_safe), 'float32')
#     # wt_y0 = (y1 - coords_y) * tf.cast(tf.equal(y0, y0_safe), 'float32')
#     # wt_y1 = (coords_y - y0) * tf.cast(tf.equal(y1, y1_safe), 'float32')
#
#     wt_x0 = x1_safe - coords_x
#     wt_x1 = coords_x - x0_safe
#     wt_y0 = y1_safe - coords_y
#     wt_y1 = coords_y - y0_safe
#
#     ## indices in the flat image to sample from
#     dim2 = tf.cast(inp_size[2], 'float32')
#     dim1 = tf.cast(inp_size[2] * inp_size[1], 'float32')
#     base = tf.reshape(
#         _repeat(
#             tf.cast(tf.range(coord_size[0]), 'float32') * dim1,
#             coord_size[1] * coord_size[2]),
#         [out_size[0], out_size[1], out_size[2], 1])
#
#     base_y0 = base + y0_safe * dim2
#     base_y1 = base + y1_safe * dim2
#     idx00 = tf.reshape(x0_safe + base_y0, [-1])
#     idx01 = x0_safe + base_y1
#     idx10 = x1_safe + base_y0
#     idx11 = x1_safe + base_y1
#
#     ## sample from imgs
#     imgs_flat = tf.reshape(imgs, tf.stack([-1, inp_size[3]]))
#     imgs_flat = tf.cast(imgs_flat, 'float32')
#     im00 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx00, 'int32')), out_size)
#     im01 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx01, 'int32')), out_size)
#     im10 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx10, 'int32')), out_size)
#     im11 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx11, 'int32')), out_size)
#
#     w00 = wt_x0 * wt_y0
#     w01 = wt_x0 * wt_y1
#     w10 = wt_x1 * wt_y0
#     w11 = wt_x1 * wt_y1
#
#     output = tf.add_n([
#         w00 * im00, w01 * im01,
#         w10 * im10, w11 * im11
#     ])
#     return output

# if __name__ == "__main__":
#
