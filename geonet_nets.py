from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.python.keras import layers, Model, regularizers
import numpy as np


class SenetLikeGuidanceLayer(layers.Layer):
    def __init__(self, num_outputs):
        super(SenetLikeGuidanceLayer, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        self.fc1 = layers.Dense(self.num_outputs, use_bias=True, activation=tf.nn.relu)
        self.fc2 = layers.Dense(self.num_outputs, use_bias=True, activation=tf.nn.sigmoid)

    def call(self, image_feature, *scale_feature):
        squeeze = tf.reduce_mean(scale_feature, axis=[1, 2])

        excitation = self.fc1(squeeze)
        excitation = self.fc2(excitation)
        excitation = tf.reshape(excitation, [-1, 1, 1, self.num_outputs])

        scale = image_feature * excitation  # input_x -> images_feature
        return scale


class PoseNet(Model):
    def __init__(self, num_source):
        super(PoseNet, self).__init__()
        self.num_source = num_source
        kernel_regularizer = regularizers.l2(0.0001)
        padding = 'same'
        activation_fn = tf.nn.relu

        self.conv1 = layers.Conv2D(16, 7, 2, padding=padding, kernel_regularizer=kernel_regularizer,
                                       activation=activation_fn)
        self.bn1 = layers.BatchNormalization()

        self.conv2 = layers.Conv2D(32, 5, 2, padding=padding, kernel_regularizer=kernel_regularizer,
                                       activation=activation_fn)
        self.bn2 = layers.BatchNormalization()

        self.conv3 = layers.Conv2D(64, 3, 2, padding=padding, kernel_regularizer=kernel_regularizer,
                                       activation=activation_fn)
        self.bn3 = layers.BatchNormalization()

        self.conv4 = layers.Conv2D(128, 3, 2, padding=padding, kernel_regularizer=kernel_regularizer,
                                       activation=activation_fn)
        self.bn4 = layers.BatchNormalization()

        self.conv5 = layers.Conv2D(256, 3, 2, padding=padding, kernel_regularizer=kernel_regularizer,
                                       activation=activation_fn)
        self.bn5 = layers.BatchNormalization()

        self.conv6 = layers.Conv2D(256, 3, 2, padding=padding, kernel_regularizer=kernel_regularizer,
                                       activation=activation_fn)
        self.bn6 = layers.BatchNormalization()

        self.conv7 = layers.Conv2D(256, 3, 2, padding=padding, kernel_regularizer=kernel_regularizer,
                                       activation=activation_fn)
        self.bn7 = layers.BatchNormalization()

        self.pose_pred = layers.Conv2D(6 * self.num_source, (1, 1), (1, 1), padding=padding, kernel_regularizer=kernel_regularizer)

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.conv4(x)
        x = self.bn4(x, training=training)
        x = self.conv5(x)
        x = self.bn5(x, training=training)
        x = self.conv6(x)
        x = self.bn6(x, training=training)
        x = self.conv7(x)
        x = self.bn7(x, training=training)
        pose_pred = self.pose_pred(x)
        pose_avg = tf.reduce_mean(pose_pred, [1, 2])
        pose_final = 0.01 * tf.reshape(pose_avg, [-1, self.num_source, 6])
        return pose_final


class Conv(layers.Layer):
    def __init__(self, num_layers, kernel_size, stride, activation=tf.nn.relu, padding='valid', batch_normalize=True):
        super(Conv, self).__init__()
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.stride = stride
        self.activation = activation
        self.padding = padding
        self.batch_normalize = batch_normalize

    def build(self, input_shape):
        self.conv = layers.Conv2D(self.num_layers, self.kernel_size, self.stride, self.padding, activation=self.activation)
        if self.batch_normalize:
            self.bn = layers.BatchNormalization()

    def call(self, x, training=None, mask=None):
        p = np.floor((self.kernel_size - 1) / 2).astype(np.int32)
        p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
        p_x = self.conv(p_x)
        if self.batch_normalize:
            p_x = self.bn(p_x, training=training)
        return p_x


class Maxpool(layers.Layer):
    def __init__(self, pool_size, stride=2, padding='valid'):
        super(Maxpool, self).__init__()
        self.pool_size = pool_size
        self.stride = stride
        self.padding = padding

    def build(self, input_shape):
        self.maxpool = layers.MaxPool2D(self.pool_size, self.stride, self.padding)

    def call(self, x, training=None, mask=None):
        p = np.floor((self.pool_size - 1) / 2).astype(np.int32)
        p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
        p_x = self.maxpool(p_x)
        return p_x


class ResConv(layers.Layer):
    def __init__(self, num_layers, stride):
        super(ResConv, self).__init__()
        self.num_layers = num_layers
        self.stride = stride

    def build(self, input_shape):
        self.conv1 = Conv(self.num_layers, 1, 1, batch_normalize=True)
        self.conv2 = Conv(self.num_layers, 3, self.stride, batch_normalize=True)
        self.conv3 = Conv(4 * self.num_layers, 1, 1, activation=None, batch_normalize=True)
        self.conv_shortcut = Conv(4 * self.num_layers, 1, self.stride, activation=None, batch_normalize=True)

    def call(self, x, training=None, mask=None):
        do_proj = tf.shape(x)[3] != self.num_layers or self.stride == 2
        shortcut = []
        conv1 = self.conv1(x, training=training)
        conv2 = self.conv2(conv1, training=training)
        conv3 = self.conv3(conv2, training=training)
        if do_proj:
            shortcut = self.conv_shortcut(x, training=training)
        else:
            shortcut = x
        return tf.nn.relu(conv3 + shortcut)


class ResBlock(layers.Layer):
    def __init__(self, num_layers, num_blocks):
        super(ResBlock, self).__init__()
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.resconv1 = []

    def build(self, input_shape=None):
        for i in range(self.num_blocks - 1):
            self.resconv1.append(ResConv(self.num_layers, 1))
        self.resconv2 = ResConv(self.num_layers, 2)

    def call(self, x, training=None, mask=None):
        for i in range(self.num_blocks - 1):
            x = self.resconv1[i](x, training=training)
        x = self.resconv2(x, training=training)
        return x


# #NO tf.function
# def upsample_nn(x, ratio):
#     h = x.get_shape()[1]
#     w = x.get_shape()[2]
#     return tf.image.resize(x, [h * ratio, w * ratio], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
#
#
# #No @tf.function
# def resize_like(inputs, ref):
#     iH, iW = inputs.get_shape()[1], inputs.get_shape()[2]
#     rH, rW = ref.get_shape()[1], ref.get_shape()[2]
#     if iH == rH and iW == rW:
#         return inputs
#     return tf.image.resize(inputs, [rH, rW], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)


class UpSample(layers.Layer):
    def __init__(self, ratio):
        super(UpSample, self).__init__()
        self.ratio = ratio

    def build(self, input_shape):
        pass

    def call(self, x, training=None, mask=None):
        h = x.get_shape()[1]
        w = x.get_shape()[2]
        return tf.image.resize(x, [h * self.ratio, w * self.ratio], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)


class ResizeLike(layers.Layer):
    def __init__(self):
        super(ResizeLike, self).__init__()

    def build(self, input_shape):
        pass

    def call(self, x, training=None, mask=None):
        inputs, ref = x
        iH, iW = inputs.get_shape()[1], inputs.get_shape()[2]
        rH, rW = ref.get_shape()[1], ref.get_shape()[2]
        return tf.image.resize(inputs, [rH, rW], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)


class UpConv(layers.Layer):
    def __init__(self, num_out_layers, kernel_size, scale):
        super(UpConv, self).__init__()
        self.num_out_layers = num_out_layers
        self.kernel_size = kernel_size
        self.scale = scale

    def build(self, input_shape):
        self.upsample = UpSample(self.scale)
        self.conv = Conv(self.num_out_layers, self.kernel_size, 1, padding='valid', batch_normalize=True)

    def call(self, x, training=None, mask=None):
        x = self.upsample(x)
        x = self.conv(x, training=training)
        return x


class GetDispResnet50(layers.Layer):
    def __init__(self):
        super(GetDispResnet50, self).__init__()
        self.DISP_SCALING_RESNET50 = 5

    def build(self, input_shape=None):
        self.conv = Conv(1, 3, 1, activation=tf.nn.sigmoid, batch_normalize=False)

    def call(self, x, training=None, mask=None):
        return self.DISP_SCALING_RESNET50 * self.conv(x, training=training) + 0.01


class DepthNet(Model):
    def __init__(self):
        super(DepthNet, self).__init__()
        activation_fn = tf.nn.relu

        self.conv = Conv(64, 7, 2, padding='valid', activation=activation_fn)
        self.maxpool = Maxpool(3, padding='valid')
        self.resblock1 = ResBlock(64, 3)
        self.resblock2 = ResBlock(128, 4)
        self.resblock3 = ResBlock(256, 6)
        self.resblock4 = ResBlock(512, 3)

        self.upconv6 = UpConv(512, 3, 2)
        self.resize_like6 = ResizeLike()
        self.conv6 = Conv(512, 3, 1, padding='valid', activation=activation_fn)

        self.upconv5 = UpConv(256, 3, 2)
        self.resize_like5 = ResizeLike()
        self.conv5 = Conv(256, 3, 1, padding='valid', activation=activation_fn)

        self.upconv4 = UpConv(128, 3, 2)
        self.resize_like4 = ResizeLike()
        self.conv4 = Conv(128, 3, 1, padding='valid', activation=activation_fn)
        self.get_disp_resnet50_4 = GetDispResnet50()
        self.upsample4 = UpSample(2)

        self.upconv3 = UpConv(64, 3, 2)
        self.conv3 = Conv(64, 3, 1, padding='valid', activation=activation_fn)
        self.get_disp_resnet50_3 = GetDispResnet50()
        self.upsample3 = UpSample(2)

        self.upconv2 = UpConv(32, 3, 2)
        self.conv2 = Conv(32, 3, 1, padding='valid', activation=activation_fn)
        self.get_disp_resnet50_2 = GetDispResnet50()
        self.upsample2 = UpSample(2)

        self.upconv1 = UpConv(16, 3, 2)
        self.conv1 = Conv(16, 3, 1, padding='valid', activation=activation_fn)
        self.get_disp_resnet50_1 = GetDispResnet50()

    def call(self, inputs, training=None, mask=None):
        conv1 = self.conv(inputs, training=training)
        pool1 = self.maxpool(conv1)
        conv2 = self.resblock1(pool1, training=training)
        conv3 = self.resblock2(conv2, training=training)
        conv4 = self.resblock3(conv3, training=training)
        conv5 = self.resblock4(conv4, training=training)

        skip1 = conv1 # 64, 208
        skip2 = pool1 # 22 70
        skip3 = conv2 # 11 35
        skip4 = conv3 # 6 18
        skip5 = conv4 # 3 9

        # Decoding
        upconv6 = self.upconv6(conv5, training=training)
        upconv6 = self.resize_like6([upconv6, skip5])
        concat6 = tf.concat([upconv6, skip5], 3)
        iconv6 = self.conv6(concat6, training=training)

        upconv5 = self.upconv5(iconv6, training=training)
        upconv5 = self.resize_like5([upconv5, skip4])
        concat5 = tf.concat([upconv5, skip4], 3)
        iconv5 = self.conv5(concat5, training=training)

        upconv4 = self.upconv4(iconv5, training=training)
        upconv4 = self.resize_like4([upconv4, skip3]) #resize_like(upconv4, skip3) # If it is a @tf_function, It makes None channel
        # print("upconv4-re : ", upconv4)
        concat4 = tf.concat([upconv4, skip3], 3)
        # print("concat4 : ", concat4)
        iconv4 = self.conv4(concat4, training=training)
        pred4 = self.get_disp_resnet50_4(iconv4, training=training)
        upred4 = self.upsample4(pred4)

        upconv3 = self.upconv3(iconv4, training=training)
        concat3 = tf.concat([upconv3, skip2, upred4], 3)
        iconv3 = self.conv3(concat3, training=training)
        pred3 = self.get_disp_resnet50_3(iconv3, training=training)
        upred3 = self.upsample3(pred3)

        upconv2 = self.upconv2(iconv3, training=training)
        concat2 = tf.concat([upconv2, skip1, upred3], 3)
        iconv2 = self.conv2(concat2, training=training)
        pred2 = self.get_disp_resnet50_2(iconv2, training=training)
        upred2 = self.upsample2(pred2)

        upconv1 = self.upconv1(iconv2, training=training)
        concat1 = tf.concat([upconv1, upred2], 3)
        iconv1 = self.conv1(concat1, training=training)
        pred1 = self.get_disp_resnet50_1(iconv1, training=training)

        return [pred1, pred2, pred3, pred4]






