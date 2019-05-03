import cv2
import numpy as np
import os
import readFlowFile
from utils import compute_rigid_flow
import tensorflow as tf

def put_optical_flow_arrows_on_image(image, optical_flow_image, threshold=100.0, skip_amount=100):
    # Don't affect original image
    image = image.copy()

    # Turn grayscale to rgb if needed
    if len(image.shape) == 2:
        image = np.stack((image,) * 3, axis=2)

    # Get start and end coordinates of the optical flow
    flow_start = np.stack(np.meshgrid(range(optical_flow_image.shape[1]), range(optical_flow_image.shape[0])), 2)
    flow_end = (optical_flow_image[flow_start[:, :, 1], flow_start[:, :, 0], :1] * 3 + flow_start).astype(np.int32)

    # Threshold values
    norm = np.linalg.norm(flow_end - flow_start, axis=2)
    norm[norm > threshold] = 0

    # Draw all the nonzero values
    nz = np.nonzero(norm)
    for i in range(0, len(nz[0]), skip_amount):
        y, x = nz[0][i], nz[1][i]
        cv2.arrowedLine(image,
                        pt1=tuple(flow_start[y, x]),
                        pt2=tuple(flow_end[y, x]),
                        color=(0, 255, 0),
                        thickness=1,
                        tipLength=.2)
    return image


if __name__ == "__main__":
    sess = tf.Session()

    image = cv2.imread(os.path.join("E:\\KITTI_dump\\00\\000003.jpg"))
    image = image[:, 416:832, :]
    # print(np.shape(image))
    # exit()
    # flo = readFlowFile.read(os.path.join("E:\\KITTI_dump\\00\\000001_obd_flow_00.flo"))
    pose = [-0.093743,-0.056761,1.716275,-0.001059,-0.004129,0.002312]
    # pose = [-0.046840,-0.028362,0.857581,-0.000529,-0.002063,0.001156] # relative
    # pose = [-0.00680046,  0.  ,        0.86374217 , 0.     ,     0.00217138 , 0.        ] # obd->pose

    # pose = [-0.046840, -0.028362, 0.857581, 0.0, 0.0, 0.0]  # relative

    # pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # relative

    intrinsic = [240.970276,0.,203.539246,0.,244.716949,63.052151,0.,0.,1.]
    pose = tf.convert_to_tensor(pose, dtype=tf.float32)
    intrinsic = tf.convert_to_tensor(intrinsic, dtype=tf.float32)
    intrinsic = tf.reshape(intrinsic, (3, 3))

    # print(sess.run(intrinsic))
    # print(sess.run(pose))


    pose = tf.expand_dims(pose, axis=0)
    intrinsic = tf.expand_dims(intrinsic, axis=0)
    image = tf.convert_to_tensor(image, dtype=tf.uint8)
    image = tf.expand_dims(image, axis=0)

    flow = compute_rigid_flow(image, pose, intrinsic)

    # print(sess.run(C))


    temp_image = tf.squeeze(image, axis=0).eval(session=sess)
    temp_flow = tf.squeeze(flow, axis=0).eval(session=sess)
    res = put_optical_flow_arrows_on_image(temp_image, temp_flow) # flo
    cv2.imshow("img", res)
    cv2.waitKey(0)