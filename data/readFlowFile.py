#   compute colored image to visualize optical flow file .flo

#   According to the matlab code of Deqing Sun and c++ source code of Daniel Scharstein
#   Contact: dqsun@cs.brown.edu
#   Contact: schar@middlebury.edu

#   Author: Johannes Oswald, Technical University Munich
#   Contact: johannes.oswald@tum.de
#   Date: 26/04/2017

#	For more information, check http://vision.middlebury.edu/flow/

import numpy as np
import os
import tensorflow as tf
TAG_FLOAT = 202021.25


def read(file):

    assert type(file) is str, "file is not str %r" % str(file)
    assert os.path.isfile(file) is True, "file does not exist %r" % str(file)
    assert file[-4:] == '.flo', "file ending is not .flo %r" % file[-4:]
    f = open(file,'r') # f = open(file,'rb')
    flo_number = np.fromfile(f, np.float32, count=1)[0]
    assert flo_number == TAG_FLOAT, 'Flow number %r incorrect. Invalid .flo file' % flo_number
    w = np.fromfile(f, np.int32, count=1)
    h = np.fromfile(f, np.int32, count=1)
    # if error try: data = np.fromfile(f, np.float32, count=2*w[0]*h[0])
    data = np.fromfile(f, np.float32, count=2 * w[0] * h[0])
    # data = np.fromfile(f, np.float32, count=2*w*h)
    # Reshape data into 3D array (columns, rows, bands)
    flow = np.resize(data, (int(h), int(w), 2))
    f.close()

    return flow


def read_byte(byte_flow):
    # flo_number = byte_flow[0:4]
    # flo_width = byte_flow[4:8]
    # flo_height = byte_flow[8:12]
    # flo_data = byte_flow[12:]

    flo_number = byte_flow[0]
    flo_width = byte_flow[1]
    flo_height = byte_flow[2]
    flo_data = byte_flow[3:]

    # assert flo_number == TAG_FLOAT, 'Flow number %r incorrect. Invalid .flo file' % flo_number
    w = 128 # tf.cast(flo_width, dtype=tf.float32)
    h = 416 # tf.cast(flo_height, dtype=tf.float32)
    data = tf.cast(flo_data, dtype=tf.float32)

    # Reshape data into 3D array (columns, rows, bands)
    flow = tf.reshape(data, shape=[h, w, 2])

    return flow
