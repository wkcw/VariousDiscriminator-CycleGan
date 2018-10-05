import tensorflow as tf
import ops


class Discriminator:
    def __init__(self, name, is_training, norm='instance', use_sigmoid=False, patchgan='True', gan_mode='lsgan', rf=70, ndf=32):
        self.name = name
        self.is_training = is_training
        self.norm = norm
        self.reuse = False
        self.patchgan = patchgan
        self.rf=rf
        self.ndf = ndf
        self.gan_mode = gan_mode
        if self.gan_mode == 'gan':
            self.use_sigmoid = True
        else:
            self.use_sigmoid = False

    def __call__(self, input):
        """
        Args:
          input: batch_size x image_size x image_size x 3
        Returns:
          output: 4D tensor batch_size x out_size x out_size x 1 (default 1x5x5x1)
                  filled with 0.9 if real, 0.0 if fake
        """

        with tf.variable_scope(self.name):
            if self.patchgan:

                if self.rf == 70:
                    print "using rf=70 patchgan"
                    C32 = ops.Ck(input, 64, stride=1, reuse=self.reuse, norm=None,
                                 is_training=self.is_training, name='C32')
                    C64 = ops.Ck(C32, 64, reuse=self.reuse, norm=self.norm,
                                 is_training=self.is_training, name='C64')  # (?, w/2, h/2, 64)
                    C128 = ops.Ck(C64, 128, reuse=self.reuse, norm=self.norm,
                                  is_training=self.is_training, name='C128')  # (?, w/4, h/4, 128)
                    C256 = ops.Ck(C128, 256, reuse=self.reuse, norm=self.norm,
                                  is_training=self.is_training, name='C256')  # (?, w/8, h/8, 256)
                    C512_0 = ops.Ck(C256, 512, reuse=self.reuse, norm=self.norm,
                                    is_training=self.is_training, name='C512_0')  # (?, w/16, h/16, 512)

                    # apply a convolution to produce a 1 dimensional output (1 channel?)
                    # use_sigmoid = False if use_lsgan = True
                    output = ops.last_conv(C512_0, reuse=self.reuse,
                                           use_sigmoid=self.use_sigmoid, name='output', kernelsize=4)  # (?, w/16, h/16, 1)
                elif self.rf == 40:
                    print "using rf=40 patchgan"
                    C32 = ops.Ck(input, 32, stride=1, reuse=self.reuse, norm=None,
                                 is_training=self.is_training, name='C32', kernelsize=3)
                    C64 = ops.Ck(C32, 64, stride=1, reuse=self.reuse, norm=self.norm,
                                 is_training=self.is_training, name='C64', kernelsize=3)  # (?, w/2, h/2, 64)
                    C128 = ops.Ck(C64, 128, reuse=self.reuse, norm=self.norm,
                                  is_training=self.is_training, name='C128')  # (?, w/4, h/4, 128)
                    C256 = ops.Ck(C128, 256, reuse=self.reuse, norm=self.norm,
                                  is_training=self.is_training, name='C256')  # (?, w/8, h/8, 256)
                    C512_0 = ops.Ck(C256, 512, reuse=self.reuse, norm=self.norm,
                                    is_training=self.is_training, name='C512_0')  # (?, w/16, h/16, 512)
                    output = ops.last_conv(C512_0, reuse=self.reuse,
                                           use_sigmoid=self.use_sigmoid, name='output', kernelsize=3)  # (?, w/16, h/16, 1)

                elif self.rf == 21:
                    print "using rf=21 patchgan"
                    C32 = ops.Ck(input, 32, stride=1, reuse=self.reuse, norm=None,
                                 is_training=self.is_training, name='C32', kernelsize=3)
                    C64 = ops.Ck(C32, 64, stride=1, reuse=self.reuse, norm=self.norm,
                                 is_training=self.is_training, name='C64', kernelsize=3)  # (?, w/2, h/2, 64)
                    C128 = ops.Ck(C64, 128, reuse=self.reuse, norm=self.norm,
                                  is_training=self.is_training, name='C128', kernelsize=3)  # (?, w/4, h/4, 128)
                    C256 = ops.Ck(C128, 256, reuse=self.reuse, norm=self.norm,
                                  is_training=self.is_training, name='C256', kernelsize=2)  # (?, w/8, h/8, 256)
                    C512_0 = ops.Ck(C256, 512, reuse=self.reuse, norm=self.norm,
                                    is_training=self.is_training, name='C512_0', kernelsize=2)  # (?, w/16, h/16, 512)
                    output = ops.last_conv(C512_0, reuse=self.reuse,
                                           use_sigmoid=self.use_sigmoid, name='output',
                                           kernelsize=2)  # (?, w/16, h/16, 1)

                elif self.rf == 56:
                    print "using rf=56 patchgan"
                    C32 = ops.Ck(input, 64, stride=1, reuse=self.reuse, norm=None,
                                 is_training=self.is_training, name='C32', kernelsize=4)
                    C64 = ops.Ck(C32, 128, stride=2, reuse=self.reuse, norm=self.norm,
                                 is_training=self.is_training, name='C64', kernelsize=4)  # (?, w/2, h/2, 64)
                    C128 = ops.Ck(C64, 256, stride=1, reuse=self.reuse, norm=self.norm,
                                  is_training=self.is_training, name='C128', kernelsize=3)  # (?, w/4, h/4, 128)
                    C256 = ops.Ck(C128, 512, stride=2, reuse=self.reuse, norm=self.norm,
                                  is_training=self.is_training, name='C256', kernelsize=4)  # (?, w/8, h/8, 256)
                    C512_0 = ops.Ck(C256, 1024, stride=2, reuse=self.reuse, norm=self.norm,
                                    is_training=self.is_training, name='C512_0', kernelsize=4)  # (?, w/16, h/16, 512)
                    output = ops.last_conv(C512_0, reuse=self.reuse,
                                           use_sigmoid=self.use_sigmoid, name='output', kernelsize=3)  # (?, w/16, h/16, 1)
                elif self.rf == 0:
                    print "using rf=0 wholegan"
                    C32 = ops.Ck(input, 32, stride=1, reuse=self.reuse, norm=None,
                                 is_training=self.is_training, name='C32')
                    C64 = ops.Ck(C32, 64, stride=2, reuse=self.reuse, norm=self.norm,
                                 is_training=self.is_training, name='C64')  # (?, w/2, h/2, 64)
                    C128 = ops.Ck(C64, 128, stride=2, reuse=self.reuse, norm=self.norm,
                                  is_training=self.is_training, name='C128')  # (?, w/4, h/4, 128)
                    C256 = ops.Ck(C128, 256, stride=2, reuse=self.reuse, norm=self.norm,
                                  is_training=self.is_training, name='C256')  # (?, w/8, h/8, 256)
                    C512 = ops.Ck(C256, 512, stride=2, reuse=self.reuse, norm=self.norm,
                                  is_training=self.is_training, name='C512')  # (?, w/16, h/16, 512)
                    # C512_0 = ops.Ck(C512, 1024, stride=1, reuse=self.reuse, norm=self.norm,
                    #            is_training=self.is_training, name='C512_0')
                    batchsize = C512.get_shape()[0].value
                    reshaped = tf.reshape(C512, [batchsize, -1])
                    output = tf.layers.dense(reshaped, 1, name='fc8', reuse=self.reuse)
                else:
                    print "unknow rf"
                    exit(0)
                if self.gan_mode == 'gan_logits':
                    batchsize = output.get_shape()[0].value
                    output = tf.reshape(output, [batchsize, -1])  # may cause error when batchsize is not 1
            else:
                # convolution layers
                C64 = ops.Ck(input, 64, stride=2, reuse=self.reuse, norm=None,
                             is_training=self.is_training, name='C64')  # (?, w/2, h/2, 64)
                C128 = ops.Ck(C64, 128, stride=2, reuse=self.reuse, norm=self.norm,
                              is_training=self.is_training, name='C128')  # (?, w/4, h/4, 128)
                C256 = ops.Ck(C128, 256, stride=2, reuse=self.reuse, norm=self.norm,
                              is_training=self.is_training, name='C256')  # (?, w/8, h/8, 256)
                C512 = ops.Ck(C256, 512, stride=2, reuse=self.reuse, norm=self.norm,
                                is_training=self.is_training, name='C512')  # (?, w/16, h/16, 512)
                #C512_0 = ops.Ck(C512, 1024, stride=1, reuse=self.reuse, norm=self.norm,
                #            is_training=self.is_training, name='C512_0')
                batchsize = C512.get_shape()[0].value
                reshaped = tf.reshape(C512, [batchsize, -1])
                output = tf.layers.dense(reshaped, 1, name='fc8', reuse=self.reuse)
                if self.use_sigmoid:
                    output = tf.sigmoid(output)


        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

        return output

    def generate_map(self, input):
        map = self.__call__(input)
        # image = tf.image.encode_jpeg(tf.squeeze(image, [0]))
        return map