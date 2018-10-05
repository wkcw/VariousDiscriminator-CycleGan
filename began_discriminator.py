import tensorflow as tf
import numpy as np
import Autoencoder

def leaky_rectify(x, leakiness=0.01):
    assert leakiness <= 1
    ret = tf.maximum(x, leakiness * x)
    return ret


def custom_conv2d(input_layer, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, in_dim=None,
                  padding='SAME', scope="conv2d"):
    with tf.variable_scope(scope):
        w = tf.get_variable('w', [k_h, k_w, in_dim or input_layer.shape[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_layer, w,
                            strides=[1, d_h, d_w, 1], padding=padding)
        b = tf.get_variable("b", shape=output_dim, initializer=tf.constant_initializer(0.))
        conv = tf.nn.bias_add(conv, b)
        return conv


def custom_fc(input_layer, output_size, scope='Linear',
              in_dim=None, stddev=0.02, bias_start=0.0):
    shape = input_layer.shape
    if len(shape) > 2:
        input_layer = tf.reshape(input_layer, [-1, int(np.prod(shape[1:]))])
    shape = input_layer.shape
    with tf.variable_scope(scope):
        matrix = tf.get_variable("weight",
                                 [in_dim or shape[1], output_size],
                                 dtype=tf.float32,
                                 initializer=tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(bias_start))
        return tf.nn.bias_add(tf.matmul(input_layer, matrix), bias)


def decoder(Z, num_filters, hidden_size, image_size):

    #layer_1 = custom_fc(Z, 8 * 8 * num_filters, scope='l1')

    #layer_1 = tf.reshape(layer_1, [-1, 8, 8, num_filters])  # '-1' is batch size

    conv_1 = custom_conv2d(Z, num_filters, k_h=3, k_w=3, d_h=1, d_w=1, scope='c1')
    conv_1 = tf.nn.elu(conv_1)

    conv_2 = custom_conv2d(conv_1, num_filters, k_h=3, k_w=3, d_h=1, d_w=1, scope='c2')
    conv_2 = tf.nn.elu(conv_2)

    layer_2 = tf.image.resize_nearest_neighbor(conv_2, [16, 16])

    conv_3 = custom_conv2d(layer_2, num_filters, k_h=3, k_w=3, d_h=1, d_w=1, scope='c3')
    conv_3 = tf.nn.elu(conv_3)

    conv_4 = custom_conv2d(conv_3, num_filters, k_h=3, k_w=3, d_h=1, d_w=1, scope='c4')
    conv_4 = tf.nn.elu(conv_4)

    layer_3 = tf.image.resize_nearest_neighbor(conv_4, [32, 32])

    conv_5 = custom_conv2d(layer_3, num_filters, k_h=3, k_w=3, d_h=1, d_w=1, scope='c5')
    conv_5 = tf.nn.elu(conv_5)

    conv_6 = custom_conv2d(conv_5, num_filters, k_h=3, k_w=3, d_h=1, d_w=1, scope='c6')
    conv_6 = tf.nn.elu(conv_6)

    layer_4 = tf.image.resize_nearest_neighbor(conv_6, [64, 64])

    conv_7 = custom_conv2d(layer_4, num_filters, k_h=3, k_w=3, d_h=1, d_w=1, scope='c7')
    conv_7 = tf.nn.elu(conv_7)

    conv_8 = custom_conv2d(conv_7, num_filters, k_h=3, k_w=3, d_h=1, d_w=1, scope='c8')
    conv_8 = tf.nn.elu(conv_8)

    if image_size == 64:
        im = conv_8
    else:
        layer_5 = tf.image.resize_nearest_neighbor(conv_8, [128, 128])

        conv_9 = custom_conv2d(layer_5, num_filters, k_h=3, k_w=3, d_h=1, d_w=1, scope='c9')
        conv_9 = tf.nn.elu(conv_9)

        conv_10 = custom_conv2d(conv_9, num_filters, k_h=3, k_w=3, d_h=1, d_w=1, scope='c10')
        im = tf.nn.elu(conv_10)

    im = custom_conv2d(im, 3, k_h=3, k_w=3, d_h=1, d_w=1, scope='im')
    im = tf.tanh(im)
    return im


def began_discriminator(D_I, num_filters, hidden_size, image_size,
                        scope_name="began_discriminator", reuse_scope=False):


    with tf.variable_scope(scope_name) as scope:
        if reuse_scope:
            scope.reuse_variables()

        conv_0 = custom_conv2d(D_I, 3, k_h=3, k_w=3, d_h=1, d_w=1, scope='ec0')
        conv_0 = tf.nn.elu(conv_0)

        conv_1 = custom_conv2d(conv_0, num_filters, k_h=3, k_w=3, d_h=1, d_w=1, scope='ec1')
        conv_1 = tf.nn.elu(conv_1)

        conv_2 = custom_conv2d(conv_1, num_filters, k_h=3, k_w=3, d_h=1, d_w=1, scope='ec2')
        conv_2 = tf.nn.elu(conv_2)

        layer_2 = custom_conv2d(conv_2, 2 * num_filters, k_h=3, k_w=3, d_h=2, d_w=2, scope='el2')
        layer_2 = tf.nn.elu(layer_2)

        conv_3 = custom_conv2d(layer_2, 2 * num_filters, k_h=3, k_w=3, d_h=1, d_w=1, scope='ec3')
        conv_3 = tf.nn.elu(conv_3)

        conv_4 = custom_conv2d(conv_3, 2 * num_filters, k_h=3, k_w=3, d_h=1, d_w=1, scope='ec4')
        conv_4 = tf.nn.elu(conv_4)

        layer_3 = custom_conv2d(conv_4, 3 * num_filters, k_h=3, k_w=3, d_h=2, d_w=2, scope='el3')
        layer_3 = tf.nn.elu(layer_3)

        conv_5 = custom_conv2d(layer_3, 3 * num_filters, k_h=3, k_w=3, d_h=1, d_w=1, scope='ec5')
        conv_5 = tf.nn.elu(conv_5)

        conv_6 = custom_conv2d(conv_5, 3 * num_filters, k_h=3, k_w=3, d_h=1, d_w=1, scope='ec6')
        conv_6 = tf.nn.elu(conv_6)

        layer_4 = custom_conv2d(conv_6, 4 * num_filters, k_h=3, k_w=3, d_h=2, d_w=2, scope='el4')
        layer_4 = tf.nn.elu(layer_4)

        conv_7 = custom_conv2d(layer_4, 4 * num_filters, k_h=3, k_w=3, d_h=1, d_w=1, scope='ec7')
        conv_7 = tf.nn.elu(conv_7)

        conv_8 = custom_conv2d(conv_7, 4 * num_filters, k_h=3, k_w=3, d_h=1, d_w=1, scope='ec8')
        conv_8 = tf.nn.elu(conv_8)

        if image_size == 64:
            enc = conv_8
            #enc = custom_fc(conv_8, hidden_size, scope='enc')
        else:
            layer_5 = custom_conv2d(conv_8, 5 * num_filters, k_h=3, k_w=3, d_h=2, d_w=2, scope='el5')
            layer_5 = tf.nn.elu(layer_5)

            conv_9 = custom_conv2d(layer_5, 5 * num_filters, k_h=3, k_w=3, d_h=1, d_w=1, scope='ec9')
            conv_9 = tf.nn.elu(conv_9)

            conv_10 = custom_conv2d(conv_9, 5 * num_filters, k_h=3, k_w=3, d_h=1, d_w=1, scope='ec10')
            conv_10 = tf.nn.elu(conv_10)
            enc = custom_fc(conv_10, hidden_size, scope='enc')

        # add elu before decoding?
        return decoder(enc, num_filters=num_filters,
                       hidden_size=hidden_size, image_size=image_size)


class BeganDiscriminator:
    def __init__(self, name, hidden_size=64, image_size=64, num_filters=128, fullyconv=False):
        self.name = name
        self.reuse = False
        self.hidden_size = hidden_size
        self.image_size = image_size
        self.num_filters = num_filters
        self.fullyconv= fullyconv
        self.encoder = Autoencoder.Encoder(name='Encoder', is_training=True, image_size=self.image_size)
        self.decoder = Autoencoder.Decoder(name='Decoder', is_training=True, image_size=self.image_size)

    def __call__(self, input):

        if not self.fullyconv:
            with tf.variable_scope(self.name):

                # began discriminator
                output = began_discriminator(input, hidden_size=self.hidden_size,
                                             image_size=self.image_size,
                                             num_filters=self.num_filters, reuse_scope=self.reuse)

            self.reuse = True
            self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

            return output

        else:
            with tf.variable_scope(self.name):

                # use fully-convolutional auto-encoder as began discriminator
                output = self.decoder(self.encoder(input))

            self.reuse = True
            self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

            return output