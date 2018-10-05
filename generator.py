import tensorflow as tf
import ops
import utils


class Generator:
  def __init__(self, name, is_training, ngf=64, norm='instance', image_size=128, activation='relu', resblock_num=6, skip_connection=False):
    self.name = name
    self.reuse = False
    self.ngf = ngf
    self.norm = norm
    self.is_training = is_training
    self.image_size = image_size
    self.activation = activation
    self.resblock_num = resblock_num
    self.skip_connection = skip_connection

  def __call__(self, input):
    """
    Args:
      input: batch_size x width x height x 3
    Returns:
      output: same size as input
    """
    with tf.variable_scope(self.name):
      # conv layers
      c7s1_32 = ops.c7s1_k(input, self.ngf, is_training=self.is_training, norm=self.norm,
          activation=self.activation, reuse=self.reuse, name='c7s1_32')                             # (?, w, h, 32)
      d64 = ops.dk(c7s1_32, 2*self.ngf, is_training=self.is_training, norm=self.norm,
          activation=self.activation, reuse=self.reuse, name='d64')                                 # (?, w/2, h/2, 64)
      d128 = ops.dk(d64, 4*self.ngf, is_training=self.is_training, norm=self.norm,
          activation=self.activation, reuse=self.reuse, name='d128')                                # (?, w/4, h/4, 128)

      if self.image_size <= 128:
        # use 6 residual blocks for 128x128 images
        res_output = ops.n_res_blocks(d128, reuse=self.reuse, n=self.resblock_num, activation=self.activation)      # (?, w/4, h/4, 128)
      else:
        # 9 blocks for higher resolution
        res_output = ops.n_res_blocks(d128, reuse=self.reuse, n=9, activation=self.activation)      # (?, w/4, h/4, 128)

      # fractional-strided convolution
      if self.skip_connection:
        print "using skip connection (unet)"
        d128_resoutput = tf.concat([d128, res_output], axis=3)
        u64 = ops.uk(d128_resoutput, 2 * self.ngf, is_training=self.is_training, norm=self.norm,
                     reuse=self.reuse, name='u64', activation=self.activation)
        d64_u64 = tf.concat([d64, u64], axis=3)
        u32 = ops.uk(d64_u64, self.ngf, is_training=self.is_training, norm=self.norm,
                     reuse=self.reuse, name='u32', output_size=self.image_size,
                     activation=self.activation)  # (?, w, h, 32)
        c7s1_32_u32 = tf.concat([c7s1_32, u32], axis=3)
        output = ops.c7s1_k(c7s1_32_u32, 3, norm=None,
                            activation='tanh', reuse=self.reuse, name='output')  # (?, w, h, 3)
      else:
        u64 = ops.uk(res_output, 2*self.ngf, is_training=self.is_training, norm=self.norm,
            reuse=self.reuse, name='u64', activation=self.activation)                                 # (?, w/2, h/2, 64)
        u32 = ops.uk(u64, self.ngf, is_training=self.is_training, norm=self.norm,
            reuse=self.reuse, name='u32', output_size=self.image_size,activation=self.activation)         # (?, w, h, 32)

        # conv layer
        # Note: the paper said that ReLU and _norm were used
        # but actually tanh was used and no _norm here
        output = ops.c7s1_k(u32, 3, norm=None,
            activation='tanh', reuse=self.reuse, name='output')           # (?, w, h, 3)
    # set reuse=True for next call
    self.reuse = True
    self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    return output

  def sample(self, input):
    image = utils.batch_convert2int(self.__call__(input))
    #image = self.__call__(input)
    #image = tf.image.encode_jpeg(tf.squeeze(image, [0]))
    return image
