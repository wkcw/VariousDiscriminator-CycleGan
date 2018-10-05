import tensorflow as tf
import ops
import utils
from reader import Reader
from improved_discriminator import Discriminator
from began_discriminator import BeganDiscriminator
from generator import Generator
from Autoencoder import *

REAL_LABEL = 0.9
LAMBDA = 10


class CycleGAN:
    def __init__(self,
                 X_train_file='',
                 Y_train_file='',
                 batch_size=1,
                 image_size=256,
                 BEGAN=False,
                 gamma=0.75,
                 patchgan=True,
                 receptive_field=[40,70],
                 ndf=32,
                 rf_weight=[1, 1],
                 gan_mode='lsgan',
                 norm='instance',
                 dis_norm='instance',
                 gen_activation='leakyrelu',
                 resblock_num=1,
                 unet=False,
                 lambda1=10.0,
                 lambda2=10.0,
                 sigma1=20.0,  # ae
                 sigma2=20.0,  # ae
                 lossmode=2,
                 learning_rate=2e-4,
                 dis_lrcoef=2,
                 gen_lrcoef=1,
                 exponential_decay=False,
                 decay_rate=0.01,
                 beta1=0.5,
                 beta2=0.999,
                 ngf=64,
                 shuffle=True,
                 maxloss='mean',
                 ):
        """
        Args:
          X_train_file: string, X tfrecords file for training
          Y_train_file: string Y tfrecords file for training
          batch_size: integer, batch size
          image_size: integer, image size
          lambda1: integer, weight for forward cycle loss (X->Y->X)
          lambda2: integer, weight for backward cycle loss (Y->X->Y)
          use_lsgan: boolean
          norm: 'instance' or 'batch'
          learning_rate: float, initial learning rate for Adam
          beta1: float, momentum term of Adam
          ngf: number of gen filters in first conv layer
        """
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.BEGAN = BEGAN
        self.gamma = gamma
        self.lossmode = lossmode
        # self.cycle_L2norm = cycle_L2norm
        # self.ae_cycle_L2norm = ae_cycle_L2norm
        self.patchgan = patchgan
        self.rf_weight = rf_weight
        self.gan_mode = gan_mode
        if gan_mode == 'gan':
            use_sigmoid = True
        else:
            use_sigmoid = False
        self.batch_size = batch_size
        self.image_size = image_size
        self.learning_rate = learning_rate
        self.dis_lrcoef = dis_lrcoef
        self.gen_lrcoef = gen_lrcoef
        self.exponential_decay = exponential_decay
        self.decay_rate = decay_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.X_train_file = X_train_file
        self.Y_train_file = Y_train_file
        self.shuffle = shuffle
        self.maxloss = maxloss

        self.is_training = tf.placeholder_with_default(True, shape=[], name='is_training')

        self.G = Generator('G', self.is_training, ngf=ngf, norm=norm, image_size=image_size, activation=gen_activation,
                           resblock_num=resblock_num, skip_connection=unet)
        self.F = Generator('F', self.is_training, ngf=ngf, norm=norm, image_size=image_size, activation=gen_activation,
                           resblock_num=resblock_num, skip_connection=unet)

        if not self.BEGAN:
            self.D_Y_1 = Discriminator('D_Y_1',
                                       self.is_training, norm=dis_norm, use_sigmoid=use_sigmoid, patchgan=self.patchgan,
                                       gan_mode=self.gan_mode, rf=receptive_field[0])
            self.D_Y_2 = Discriminator('D_Y_2',
                                       self.is_training, norm=dis_norm, use_sigmoid=use_sigmoid, patchgan=self.patchgan,
                                       gan_mode=self.gan_mode, rf=receptive_field[1])
            self.D_X_1 = Discriminator('D_X_1',
                                       self.is_training, norm=dis_norm, use_sigmoid=use_sigmoid, patchgan=self.patchgan,
                                       gan_mode=self.gan_mode, rf=receptive_field[0], ndf=ndf)
            self.D_X_2 = Discriminator('D_X_2',
                                       self.is_training, norm=dis_norm, use_sigmoid=use_sigmoid, patchgan=self.patchgan,
                                       gan_mode=self.gan_mode, rf=receptive_field[1])
        else:
            self.D_Y = BeganDiscriminator('D_Y', hidden_size=256, image_size=self.image_size, num_filters=128,
                                          fullyconv=True)
            self.D_X = BeganDiscriminator('D_X', hidden_size=256, image_size=self.image_size, num_filters=128,
                                          fullyconv=True)

        if self.sigma1 != 0:
            self.Encoder_X = Encoder(name='X_encoder', is_training=True)
        if self.sigma2 != 0:
            self.Encoder_Y = Encoder(name='Y_encoder', is_training=True)

        self.fake_x = tf.placeholder(tf.float32,
                                     shape=[batch_size, image_size, image_size, 3])
        self.fake_y = tf.placeholder(tf.float32,
                                     shape=[batch_size, image_size, image_size, 3])

    def model(self):


        X_reader = Reader(self.X_train_file, name='X',
                          image_size=self.image_size, batch_size=self.batch_size, shuffle=self.shuffle)
        x = X_reader.feed()
        Y_reader = Reader(self.Y_train_file, name='Y',
                          image_size=self.image_size, batch_size=self.batch_size, shuffle=self.shuffle)
        y = Y_reader.feed()



        fake_y = self.G(x)
        reconstructed_x = self.F(self.G(x))
        fake_x = self.F(y)
        reconstructed_y = self.G(self.F(y))

        x2y_cycle_loss, y2x_cycle_loss, cycle_loss = self.cycle_consistency_loss(reconstructed_x, reconstructed_y, x, y,
                                                                                 self.lossmode)
        if self.sigma1 != 0 and self.sigma2 != 0:
            x2y_ae_cycle_loss, y2x_ae_cycle_loss, ae_cycle_loss = self.autoencoder_cycle_loss(reconstructed_x,
                                                                                              reconstructed_y, x, y,
                                                                                              self.Encoder_X,
                                                                                              self.Encoder_Y,
                                                                                              self.lossmode)
        if self.batch_size ==2:
            x2y_batch_loss = tf.abs(tf.subtract(fake_y[1], fake_y[0]))
            y2x_batch_loss = tf.abs(tf.subtract(fake_x[1], fake_x[0]))

        # if not using BEGAN
        if not self.BEGAN:
            print "rf_weight[0]", self.rf_weight[0], "rf_weight[1]", self.rf_weight[1]
            # X -> Y
            D_Y_output_real_1 = self.D_Y_1(y)
            D_Y_output_fake_forG_1 = self.D_Y_1(fake_y)
            D_Y_output_fake_forD_1 = self.D_Y_1(self.fake_y)
            G_gan_loss_1 = self.generator_loss(D_Y_output_fake_forG_1, gan_mode=self.gan_mode, maxloss=self.maxloss)
            D_Y_loss_1 = self.discriminator_loss(self.D_Y_1, D_Y_output_fake_forD_1, D_Y_output_real_1, self.fake_y, y,
                                               gan_mode=self.gan_mode, maxloss='mean')
            D_Y_output_real_2 = self.D_Y_2(y)
            D_Y_output_fake_forG_2 = self.D_Y_2(fake_y)
            D_Y_output_fake_forD_2 = self.D_Y_2(self.fake_y)
            G_gan_loss_2 = self.generator_loss(D_Y_output_fake_forG_2, gan_mode=self.gan_mode)
            D_Y_loss_2 = self.discriminator_loss(self.D_Y_2, D_Y_output_fake_forD_2, D_Y_output_real_2, self.fake_y, y,
                                               gan_mode=self.gan_mode)
            G_loss = tf.div(self.rf_weight[0]*G_gan_loss_1+self.rf_weight[1]*G_gan_loss_2, self.rf_weight[0]+self.rf_weight[1])*1
            D_Y_loss = tf.div(self.rf_weight[0]*D_Y_loss_1+self.rf_weight[1]*D_Y_loss_2, self.rf_weight[0]+self.rf_weight[1])*1
            if self.lambda1 != 0:
                G_loss = G_loss + x2y_cycle_loss
            if self.sigma1 != 0:
                G_loss = G_loss + ae_cycle_loss

            # Y -> X
            D_X_output_real_1 = self.D_X_1(x)
            D_X_output_fake_forG_1 = self.D_X_1(fake_x)
            D_X_output_fake_forD_1 = self.D_X_1(self.fake_x)
            F_gan_loss_1 = self.generator_loss(D_X_output_fake_forG_1, gan_mode=self.gan_mode, maxloss=self.maxloss)
            D_X_loss_1 = self.discriminator_loss(self.D_X_1, D_X_output_fake_forD_1, D_X_output_real_1, self.fake_x, x,
                                               gan_mode=self.gan_mode, maxloss='mean')
            D_X_output_real_2 = self.D_X_2(x)
            D_X_output_fake_forG_2 = self.D_X_2(fake_x)
            D_X_output_fake_forD_2 = self.D_X_2(self.fake_x)
            F_gan_loss_2 = self.generator_loss(D_X_output_fake_forG_2, gan_mode=self.gan_mode)
            D_X_loss_2 = self.discriminator_loss(self.D_X_2, D_X_output_fake_forD_2, D_X_output_real_2, self.fake_x, x,
                                                 gan_mode=self.gan_mode)
            F_loss = tf.div(self.rf_weight[0]*F_gan_loss_1+self.rf_weight[1]*F_gan_loss_2, self.rf_weight[0]+self.rf_weight[1])*2
            D_X_loss = tf.div(self.rf_weight[0]*D_X_loss_1+self.rf_weight[1]*D_X_loss_2, self.rf_weight[0]+self.rf_weight[1])*2
            if self.lambda2 != 0:
                F_loss = F_loss + y2x_cycle_loss
            if self.sigma2 != 0:
                F_loss = F_loss + ae_cycle_loss


            # summary
            tf.summary.scalar('loss/G_1', G_gan_loss_1)
            tf.summary.scalar('loss/D_Y_1', D_Y_loss_1)
            tf.summary.scalar('loss/G_2', G_gan_loss_2)
            tf.summary.scalar('loss/D_Y_2', D_Y_loss_2)
            tf.summary.scalar('loss/F_1', F_gan_loss_1)
            tf.summary.scalar('loss/D_X_1', D_X_loss_1)
            tf.summary.scalar('loss/F_2', F_gan_loss_2)
            tf.summary.scalar('loss/D_X_2', D_X_loss_2)
            if self.lambda1 != 0:
                tf.summary.scalar('loss/X2Y_cycle', x2y_cycle_loss)
                tf.summary.scalar('loss/Y2X_cycle', y2x_cycle_loss)
            if self.sigma1 != 0:
                tf.summary.scalar('loss/X2Y_autoencoder', x2y_ae_cycle_loss)
                tf.summary.scalar('loss/Y2X_autoencoder', y2x_ae_cycle_loss)
            tf.summary.scalar('D_outputs/D_Y_fake_1', tf.reduce_mean(tf.squeeze(D_Y_output_fake_forG_1)))
            tf.summary.scalar('D_outputs/D_Y_real_1', tf.reduce_mean(tf.squeeze(D_Y_output_real_1)))
            tf.summary.scalar('D_outputs/D_Y_fake_2', tf.reduce_mean(tf.squeeze(D_Y_output_fake_forG_2)))
            tf.summary.scalar('D_outputs/D_Y_real_2', tf.reduce_mean(tf.squeeze(D_Y_output_real_2)))
            tf.summary.scalar('D_outputs/D_X_fake_1', tf.reduce_mean(tf.squeeze(D_X_output_fake_forG_1)))
            tf.summary.scalar('D_outputs/D_X_real_1', tf.reduce_mean(tf.squeeze(D_X_output_real_1)))
            tf.summary.scalar('D_outputs/D_X_fake_2', tf.reduce_mean(tf.squeeze(D_X_output_fake_forG_2)))
            tf.summary.scalar('D_outputs/D_X_real_2', tf.reduce_mean(tf.squeeze(D_X_output_real_2)))
            return G_loss, D_Y_loss, F_loss, D_X_loss, fake_y, fake_x, y, x, reconstructed_y, reconstructed_x, \
                   D_Y_output_real_1, D_Y_output_fake_forG_1, D_Y_output_real_2, D_Y_output_fake_forG_2, \
                   D_X_output_real_1, D_X_output_fake_forG_1, D_X_output_real_2, D_X_output_fake_forG_2

        # if using BEGAN
        else:
            print "using BEGAN"
            D_X_real_out = self.D_X(x)
            D_Y_real_out = self.D_Y(y)
            D_Y_gen_out = self.D_Y(fake_y)
            D_X_gen_out = self.D_X(fake_x)
            self.G_k_t = tf.get_variable('G_kt', [], initializer=tf.constant_initializer(0), trainable=False)
            self.F_k_t = tf.get_variable('F_kt', [], initializer=tf.constant_initializer(0), trainable=False)
            gamma = self.gamma
            D_Y_loss, G_loss, G_k_tp, G_convergence_measure, G_mu_real, G_mu_gen = self.began_loss(D_real_in=y,
                                                                                                   D_real_out=D_Y_real_out,
                                                                                                   D_gen_in=fake_y,
                                                                                                   D_gen_out=D_Y_gen_out,
                                                                                                   k_t=self.G_k_t,
                                                                                                   gamma=gamma)
            D_X_loss, F_loss, F_k_tp, F_convergence_measure, F_mu_real, F_mu_gen = self.began_loss(D_real_in=x,
                                                                                                   D_real_out=D_X_real_out,
                                                                                                   D_gen_in=fake_x,
                                                                                                   D_gen_out=D_X_gen_out,
                                                                                                   k_t=self.F_k_t,
                                                                                                   gamma=gamma)
            return G_loss, G_k_tp, G_convergence_measure, D_Y_loss, F_loss, F_k_tp, F_convergence_measure, D_X_loss, fake_y, fake_x, y, x, reconstructed_y, reconstructed_x, \
                   G_mu_real, G_mu_gen, F_mu_real, F_mu_gen

            # tf.summary.image('X/generated', utils.batch_convert2int(self.G(x)))
            # tf.summary.image('X/reconstruction', utils.batch_convert2int(self.F(self.G(x))))
            # tf.summary.image('X/reconstruction', utils.batch_convert2int(reconstructed_x))
            # tf.summary.image('Y/generated', utils.batch_convert2int(self.F(y)))
            # tf.summary.image('Y/reconstruction', utils.batch_convert2int(self.G(self.F(y))))
            # tf.summary.image('Y/reconstruction', utils.batch_convert2int(reconstructed_y))

    def optimize(self, G_loss, D_Y_loss, F_loss, D_X_loss):
        def make_optimizer(loss, variables, name='Adam', lrcoef=1):
            """ Adam optimizer with learning rate 0.0002 for the first 100k steps (~100 epochs)
                and a linearly decaying rate that goes to zero over the next 100k steps
            """
            global_step = tf.Variable(0, trainable=False)
            starter_learning_rate = self.learning_rate * lrcoef
            beta1 = self.beta1
            beta2 = self.beta2
            if not self.exponential_decay:
                end_learning_rate = self.learning_rate * 0.01 * lrcoef
                start_decay_step = 10000
                decay_steps = 140000
                learning_rate = (
                    tf.where(
                        tf.greater_equal(global_step, start_decay_step),
                        tf.train.polynomial_decay(starter_learning_rate, global_step - start_decay_step,
                                                  decay_steps, end_learning_rate,
                                                  power=1.0),
                        starter_learning_rate
                    )

                )
            else:
                decay_rate = self.decay_rate
                start_decay_step = 10000
                decay_steps = 140000
                learning_rate = (
                    tf.where(
                        tf.greater_equal(global_step, start_decay_step),
                        tf.train.exponential_decay(starter_learning_rate, global_step - start_decay_step,
                                                   decay_steps, decay_rate),
                        starter_learning_rate
                    )

                )
            tf.summary.scalar('learning_rate/{}'.format(name), learning_rate)

            learning_step = (
                tf.train.AdamOptimizer(learning_rate, beta1=beta1, beta2=beta2, name=name)
                    .minimize(loss, global_step=global_step, var_list=variables)
            )
            return learning_step
        G_optimizer = make_optimizer(G_loss, self.G.variables, name='Adam_G', lrcoef=self.gen_lrcoef)
        D_Y_optimizer = make_optimizer(D_Y_loss, self.D_Y_1.variables+self.D_Y_2.variables, name='Adam_D_Y', lrcoef=self.dis_lrcoef)
        F_optimizer = make_optimizer(F_loss, self.F.variables, name='Adam_F', lrcoef=self.gen_lrcoef)
        D_X_optimizer = make_optimizer(D_X_loss, self.D_X_1.variables+self.D_X_2.variables, name='Adam_D_X', lrcoef=self.dis_lrcoef)

        with tf.control_dependencies([G_optimizer, F_optimizer]):
            G_optimizers = tf.no_op(name='G_optimizers')
        with tf.control_dependencies([D_Y_optimizer, D_X_optimizer]):
            D_optimizers = tf.no_op(name='D_optimizers')
        return G_optimizers, D_optimizers

    def discriminator_loss(self, D, D_output_fake, D_output_real, fake, real, gan_mode='lsgan', maxloss='mean'):
        """ Note: default: D(y).shape == (batch_size,5,5,1),
                           fake_buffer_size=50, batch_size=1
        Args:
          G: generator object
          D: discriminator object
          y: 4D tensor (batch_size, image_size, image_size, 3)
        Returns:
          loss: scalar
        """
        if gan_mode == 'lsgan':
            if maxloss == 'mean':
                # use mean squared error
                error_real = tf.reduce_mean(tf.squared_difference(D_output_real, REAL_LABEL))
                error_fake = tf.reduce_mean(tf.square(D_output_fake))
            elif maxloss == 'max':
                # use max squared error
                error_real = tf.reduce_max(tf.squared_difference(D_output_real, REAL_LABEL))
                error_fake = tf.reduce_max(tf.square(D_output_fake))
            '''
            elif maxloss == 'softmax':
                loss_map = (tf.squared_difference(D_output_real, REAL_LABEL) +
                            tf.square(D_output_fake)) / 2
                loss_map_shape = loss_map.get_shape()
                reshaped_loss_map = tf.reshape(loss_map, shape=[loss_map_shape[0], -1])
                softmax_weight = tf.nn.softmax(reshaped_loss_map, dim=1)
                error = tf.reduce_sum(softmax_weight * reshaped_loss_map)
                loss = error / 2
                return loss
            '''


        elif gan_mode == 'lcgan':
            # use mean cubic error
            error_real = tf.reduce_mean(tf.pow(tf.abs(tf.subtract(D_output_real, REAL_LABEL)), 3))
            error_fake = tf.reduce_mean(tf.pow(tf.abs(D_output_fake), 3))
        elif gan_mode == 'gan':
            # use cross entropy
            error_real = -tf.reduce_mean(ops.safe_log(D_output_real))
            error_fake = -tf.reduce_mean(ops.safe_log(1 - D_output_fake))
        elif gan_mode == 'gan_logits':
            if self.patchgan:
                constant08 = tf.constant(0.8, shape=(self.batch_size, 64))
                constant02 = tf.constant(0.2, shape=(self.batch_size, 64))
                error_real = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(constant08, D_output_real))
                error_fake = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(constant02, D_output_fake))
            else:
                constant08 = tf.constant(0.8, shape=(self.batch_size, 1))
                constant02 = tf.constant(0.2, shape=(self.batch_size, 1))
                error_real = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(constant08, D_output_real))
                error_fake = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(constant02, D_output_fake))
        elif gan_mode == 'wgangp':
            alpha = tf.random_uniform(
                shape=[self.batch_size, 1],
                minval=0.,
                maxval=1.
            )
            real_result = D_output_real
            fake_result = D_output_fake
            d_loss = tf.reduce_mean(fake_result - real_result)  # This optimizes the discriminator.
            differences = fake - real
            interpolates = real + tf.multiply(alpha, differences)
            gradients = tf.gradients(D(interpolates), [interpolates])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[3]))
            gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
            d_loss += LAMBDA * gradient_penalty
            return d_loss
        else:
            print 'unknown gan mode %s' % gan_mode
            exit(0)

        loss = (error_real + error_fake) / 2
        return loss

    def generator_loss(self, D_output_fake, gan_mode='lsgan', maxloss='mean'):
        """  fool discriminator into believing that G(x) is real
        """
        if gan_mode == 'lsgan':
            if maxloss == 'mean':
                # use mean squared error
                loss = tf.reduce_mean(tf.squared_difference(D_output_fake, REAL_LABEL))
            elif maxloss == 'max':
                # use max squared error
                loss = tf.reduce_max(tf.squared_difference(D_output_fake, REAL_LABEL))
            elif maxloss == 'softmax':
                #use softmax squared error
                loss_map = (tf.squared_difference(D_output_fake, REAL_LABEL))
                batchsize = loss_map.get_shape()[0].value
                reshaped_loss_map = tf.reshape(loss_map, shape=[batchsize, -1])
                softmax_weight = tf.nn.softmax(reshaped_loss_map, dim=1)
                loss = tf.reduce_sum(softmax_weight * reshaped_loss_map)
            elif maxloss == 'focal':
                loss_map = (tf.squared_difference(D_output_fake, REAL_LABEL) +
                            tf.square(D_output_fake)) / 2
                loss_map_shape = loss_map.get_shape()
                D_output_fake_shape = D_output_fake.get_shape()
                prob_weight = (1 - D_output_fake) * 1.5  # here debug the prob coef
                print 'loss_map_shape:', loss_map_shape
                print 'D_output_fake_shape:', D_output_fake_shape
                loss = tf.reduce_mean(prob_weight * loss_map)

        elif gan_mode == 'lcgan':
            loss = tf.reduce_mean(tf.pow(tf.abs(tf.subtract(D_output_fake, REAL_LABEL)), 3))
        elif gan_mode == 'gan':
            # heuristic, non-saturating loss
            loss = -tf.reduce_mean(ops.safe_log(D_output_fake)) / 2
        elif gan_mode == 'gan_logits':
            if self.patchgan:
                constant05 = tf.constant(0.5, shape=(self.batch_size, 64))
                loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(constant05, D_output_fake))
            else:
                constant05 = tf.constant(0.5, shape=(self.batch_size, 1))
                loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(constant05, D_output_fake))
        elif gan_mode == 'wgangp':
            fake_result = D_output_fake
            g_loss = - tf.reduce_mean(fake_result)  # This optimizes the generator.
            return g_loss
        else:
            print 'unknown gan mode %s' % gan_mode
            exit(0)
        return loss

    def cycle_consistency_loss(self, reconstructed_x, reconstructed_y, x, y, loss_mode=2, ):
        """
        cycle consistency loss (L1 norm)
        loss_mode = 1 : L1_norm
        loss_mode = 2 : L2_norm
        loss_mode = 3 : huber_loss
        """
        if loss_mode == 1:
            forward_loss = tf.reduce_mean(tf.abs(reconstructed_x - x))
            backward_loss = tf.reduce_mean(tf.abs(reconstructed_y - y))
        elif loss_mode == 2:
            forward_loss = tf.reduce_mean(tf.square(reconstructed_x - x))
            backward_loss = tf.reduce_mean(tf.square(reconstructed_y - y))
        elif loss_mode == 3:
            forward_loss = tf.reduce_mean(tf.losses.huber_loss(x, reconstructed_x, weights=5, delta=0.2))
            backward_loss = tf.reduce_mean(tf.losses.huber_loss(y, reconstructed_y, weights=5, delta=0.2))
        elif loss_mode == 0:
            print 'cycle softmax'
            forward_loss_map = tf.square(reconstructed_x - x)
            backward_loss_map = tf.square(reconstructed_y - y)
            batchsize = forward_loss_map.get_shape()[0].value
            cycle_softmax_coef = 0.75

            reshaped_forward_loss_map = tf.reshape(forward_loss_map, shape=[batchsize, -1])
            forward_softmax_weight = tf.nn.softmax(reshaped_forward_loss_map*cycle_softmax_coef, dim=1)
            forward_loss = tf.reduce_sum(forward_softmax_weight * reshaped_forward_loss_map)

            reshaped_backward_loss_map = tf.reshape(backward_loss_map, shape=[batchsize, -1])
            backward_softmax_weight = tf.nn.softmax(reshaped_backward_loss_map*cycle_softmax_coef, dim=1)
            backward_loss = tf.reduce_sum(backward_softmax_weight * reshaped_backward_loss_map)

        else:
            print 'Unknown cycle loss mode'
            exit(0)
        loss = self.lambda1 * forward_loss + self.lambda2 * backward_loss
        return self.lambda1 * forward_loss, self.lambda2 * backward_loss, loss

    def autoencoder_cycle_loss(self, reconstructed_x, reconstructed_y, x, y, Encoder_X, Encoder_Y, loss_mode=2):
        """
        autoencoder cycle consistency loss
        if L2_norm==True use L2 norm else use L2 norm
        """
        if loss_mode == 1:
            ae_forward_loss = tf.reduce_mean(tf.abs(Encoder_X(reconstructed_x) - Encoder_X(x)))
            ae_backward_loss = tf.reduce_mean(tf.abs(Encoder_Y(reconstructed_y) - Encoder_Y(y)))
        elif loss_mode == 2:
            ae_forward_loss = tf.reduce_mean(tf.square(Encoder_X(reconstructed_x) - Encoder_X(x)))
            ae_backward_loss = tf.reduce_mean(tf.square(Encoder_Y(reconstructed_y) - Encoder_Y(y)))
        elif loss_mode == 3:
            ae_forward_loss = tf.reduce_mean(tf.losses.huber_loss(Encoder_X(x), Encoder_X(reconstructed_x)))
            ae_backward_loss = tf.reduce_mean(tf.losses.huber_loss(Encoder_Y(y), Encoder_Y(reconstructed_y)))
        else:
            print 'Unknown ae loss mode'
            exit(0)
        loss = self.sigma1 * ae_forward_loss + self.sigma2 * ae_backward_loss
        return self.sigma1 * ae_forward_loss, self.sigma2 * ae_backward_loss, loss

    def began_loss(self, D_real_in, D_real_out, D_gen_in, D_gen_out, k_t, gamma=0.75):
        def pixel_autoencoder_loss(out, inp):
            '''
            The autoencoder loss used is the L1 norm (note that this
            is based on the pixel-wise distribution of losses
            that the authors assert approximates the Normal distribution)
            args:
                out:  discriminator output
                inp:  discriminator input
            returns:
                L1 norm of pixel-wise loss
            '''
            eta = 1  # paper uses L1 norm
            diff = tf.abs(out - inp)
            if eta == 1:
                return tf.reduce_mean(diff)
            else:
                return tf.reduce_mean(tf.pow(diff, eta))

        mu_real = pixel_autoencoder_loss(D_real_out, D_real_in)
        mu_gen = pixel_autoencoder_loss(D_gen_out, D_gen_in)
        D_loss = mu_real - k_t * mu_gen
        G_loss = mu_gen
        lam = 0.001  # 'learning rate' for k. Berthelot et al. use 0.001
        k_tp = k_t + lam * (gamma * mu_real - mu_gen)
        convergence_measure = mu_real + np.abs(gamma * mu_real - mu_gen)
        return D_loss, G_loss, k_tp, convergence_measure, mu_real, mu_gen
