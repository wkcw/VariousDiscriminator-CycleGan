import tensorflow as tf
from variousD_aemodel import CycleGAN
from reader import Reader
from datetime import datetime
import os
import logging
from utils import ImagePool, concat_save, concat_3_save, data_list_reader, scipysave
from scipy import misc

import numpy as np

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('batch_size', 1, 'batch size, default: 1')
tf.flags.DEFINE_integer('image_size', 128, 'image size, default: 256')

tf.flags.DEFINE_bool('patchgan', True, 'use patchgan, default: False')
tf.flags.DEFINE_string('receptive_field', '40_70', 'receptive field of patchgan, default: 40_70')
tf.flags.DEFINE_integer('ndf', 32, 'number of channel in 1st layer of 40 discriminator, default: 32')
tf.flags.DEFINE_string('rf_weight', '1_1',
                       'weight for various receptive field discriminator of patchgan, default: 1_1')

tf.flags.DEFINE_string('gan_mode', 'lsgan',
                       'use lsgan/gan/gan_logits/wgangp , default: lsgan')
tf.flags.DEFINE_string('dis_norm', 'instance',
                       '[instance, batch, layer] use instance/batch/layer norm, default: instance')
tf.flags.DEFINE_string('norm', 'instance',
                       '[instance, batch] use instance norm or batch norm, default: instance')
tf.flags.DEFINE_string('gen_activation', 'leakyrelu',
                       '[leakyrelu, relu] activation func for generator, default: leakyrelu')
tf.flags.DEFINE_bool('unet', False,
                       'whether to use unet, default: False')
tf.flags.DEFINE_integer('resblock_num', 1, 'number of residual blocks in G, default: 1')

tf.flags.DEFINE_integer('lambda1', 0,
                        'weight for forward cycle loss (X->Y->X), default: 10.0')
tf.flags.DEFINE_integer('lambda2', 0,
                        'weight for backward cycle loss (Y->X->Y), default: 10.0')
tf.flags.DEFINE_integer('sigma1', 0,
                        'weight for forward autoencoder cycle loss (X->Y->X), default: 10.0')
tf.flags.DEFINE_integer('sigma2', 0,
                        'weight for backward autoencoder cycle loss (Y->X->Y), default: 10.0')
tf.flags.DEFINE_integer('lossmode', 2, '1:L1 2:L2 3:huber, default: 2')

tf.flags.DEFINE_float('learning_rate', 1e-4,
                      'initial learning rate for Adam, default: 0.0002')
tf.flags.DEFINE_integer('dis_pretrain', 1,
                        'learning rate coef for discriminator, default: 2')
tf.flags.DEFINE_integer('dis_lrcoef', 1,
                        'learning rate coef for discriminator, default: 2')
tf.flags.DEFINE_integer('gen_lrcoef', 2,
                        'learning rate coef for discriminator, default: 2')

tf.flags.DEFINE_bool('exponential_decay', True,
                     'use expponentitial decay as learning rate decay policy, default: False')
tf.flags.DEFINE_float('decay_rate', 0.01,
                      'decay_rate, default: 0.01')

tf.flags.DEFINE_float('beta1', 0.5,
                      'momentum term of Adam, default: 0.5')
tf.flags.DEFINE_float('beta2', 0.999,
                      'momentum term beta2 of Adam, default: 0.5')
tf.flags.DEFINE_float('pool_size', 50,
                      'size of image buffer that stores previously generated images, default: 50')
tf.flags.DEFINE_integer('ngf', 64,
                        'number of gen filters in first conv layer, default: 64')
tf.flags.DEFINE_bool('shuffle', True,
                     'shuffle data, default: True')

tf.flags.DEFINE_string('maxloss', 'mean',
                     'true for max, false for mean , default: False')

tf.flags.DEFINE_bool('BEGAN', False,
                       'whether to use BEGAN, default: False')
tf.flags.DEFINE_float('gamma', 1.0, 'gamma for BEGAN, default:1.0')

tf.flags.DEFINE_string('X', 'data/tfrecords/apple.tfrecords',
                       'X tfrecords file for training, default: data/tfrecords/apple.tfrecords')
tf.flags.DEFINE_string('Y', 'data/tfrecords/orange.tfrecords',
                       'Y tfrecords file for training, default: data/tfrecords/orange.tfrecords')

tf.flags.DEFINE_string('ae_x_ckpt_dir', 'autoencoder_checkpoint/X_encoder',
                       'Autoencoder checkpoint for X, default: autoencoder_checkpoint/X_encoder')
tf.flags.DEFINE_string('ae_y_ckpt_dir', 'autoencoder_checkpoint/Y_encoder',
                       'Autoencoder checkpoint for Y, default: autoencoder_checkpoint/Y_encoder')
tf.flags.DEFINE_string('whole_ckpt_dir', 'no',
                       'Autoencoder checkpoint for Y, default: no')

tf.flags.DEFINE_string('testimage_x_dir', 'test_image/X500',
                       'test images of x directory, default: test_image/X500')
tf.flags.DEFINE_string('testimage_y_dir', 'test_image/Y500',
                       'test images of y directory, default: test_image/Y500')
tf.flags.DEFINE_string('testoutput_dir', 'test_output',
                       'test output images directory, default: ./test_output')
tf.flags.DEFINE_string('generated_image_dir', 'generated_image',
                       'generated images directory, default: ./test_output')
tf.flags.DEFINE_string('gpu', 'gpu',
                       'gpu name for f, default: gpu (cause error)')






def typeall(varlist):
    for var in varlist:
        print type(var)
    return

def compute_change(x, y):
    return np.sum(np.abs(x-y))


def make_suffix(flag_list):
    suffix = str(flag_list.pop(0))
    for flag in flag_list:
        suffix = suffix + '_'
        suffix = suffix + str(flag)
    return suffix

def train():
    if FLAGS.gpu == 'gpu':
        print "Unspecified gpu name!"
        exit(0)
    current_time = datetime.now().strftime("%Y%m%d-%H%M")
    suffix = make_suffix([FLAGS.batch_size,FLAGS.image_size,FLAGS.patchgan,FLAGS.gan_mode,FLAGS.dis_norm,FLAGS.norm,FLAGS.unet,FLAGS.lambda1,FLAGS.lambda2,
        FLAGS.sigma1,FLAGS.sigma2,FLAGS.lossmode,FLAGS.learning_rate,FLAGS.exponential_decay,FLAGS.ngf,FLAGS.decay_rate, FLAGS.dis_lrcoef, FLAGS.gen_lrcoef, FLAGS.BEGAN, FLAGS.gamma, FLAGS.receptive_field, FLAGS.rf_weight])
    checkpoints_dir = "./checkpoints/{}".format(current_time) + FLAGS.gpu + suffix
    testoutput_images_dir = FLAGS.testoutput_dir + '/{}'.format(current_time) + FLAGS.gpu + suffix
    generated_images_dir = FLAGS.generated_image_dir + '/{}'.format(current_time) + FLAGS.gpu + suffix
    try:
        os.makedirs(checkpoints_dir)
    except os.error, e:
        pass
    try:
        os.makedirs(testoutput_images_dir)
    except os.error, e:
        pass
    try:
        os.makedirs(generated_images_dir)
    except os.error, e:
        pass

    testimage_x_list = data_list_reader(FLAGS.testimage_x_dir, FLAGS.image_size)
    testimage_y_list = data_list_reader(FLAGS.testimage_y_dir, FLAGS.image_size)
    test_length = len(testimage_x_list)

    graph = tf.Graph()
    with graph.as_default():
        cycle_gan = CycleGAN(
            X_train_file=FLAGS.X,
            Y_train_file=FLAGS.Y,
            batch_size=FLAGS.batch_size,
            image_size=FLAGS.image_size,
            BEGAN=FLAGS.BEGAN,
            gamma=FLAGS.gamma,
            patchgan=FLAGS.patchgan,
            receptive_field=[int(i) for i in FLAGS.receptive_field.split('_')],
            ndf=FLAGS.ndf,
            rf_weight=[float(i) for i in FLAGS.rf_weight.split('_')],
            gan_mode=FLAGS.gan_mode,
            dis_norm=FLAGS.dis_norm,
            norm=FLAGS.norm,
            gen_activation=FLAGS.gen_activation,
            resblock_num = FLAGS.resblock_num,
            unet = FLAGS.unet,
            lambda1=FLAGS.lambda1,
            lambda2=FLAGS.lambda1,
            sigma1=FLAGS.sigma1,
            sigma2=FLAGS.sigma2,
            lossmode=FLAGS.lossmode,
            learning_rate=FLAGS.learning_rate,
            dis_lrcoef=FLAGS.dis_lrcoef,
            gen_lrcoef=FLAGS.gen_lrcoef,
            exponential_decay=FLAGS.exponential_decay,
            decay_rate = FLAGS.decay_rate,
            beta1=FLAGS.beta1,
            beta2=FLAGS.beta2,
            ngf=FLAGS.ngf,
            shuffle=FLAGS.shuffle,
            maxloss=FLAGS.maxloss
        )
        # G_loss, D_Y_loss, F_loss, D_X_loss, fake_y, fake_x, y, x = cycle_gan.model()
        if not FLAGS.BEGAN:
            G_loss, D_Y_loss, F_loss, D_X_loss, fake_y, fake_x, y, x, reconstructed_y, reconstructed_x, \
            D_Y_output_real_1, D_Y_output_fake_forG_1, D_Y_output_real_2, D_Y_output_fake_forG_2, \
            D_X_output_real_1, D_X_output_fake_forG_1, D_X_output_real_2, D_X_output_fake_forG_2 = cycle_gan.model()
            optimizers = cycle_gan.optimize(G_loss, D_Y_loss, F_loss, D_X_loss)
            G_optimizer, D_optimizer = optimizers
        else:
            G_loss, G_k_tp, G_convergence_measure, D_Y_loss, F_loss, F_k_tp, F_convergence_measure, D_X_loss,\
            fake_y, fake_x, y, x, reconstructed_y, reconstructed_x, G_mu_real, G_mu_gen, F_mu_real, F_mu_gen = cycle_gan.model()
            optimizers = cycle_gan.optimize(G_loss, D_Y_loss, F_loss, D_X_loss)
            G_optimizer, D_optimizer = optimizers

        #summary
        summary_op = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(checkpoints_dir, graph)

        #saver
        saver = tf.train.Saver(max_to_keep=5)
        if FLAGS.sigma1 != 0:
            saver_ae_x = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='X_encoder'))
            saver_ae_y = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Y_encoder'))

        # test model part
        test_x_input = tf.placeholder(tf.float32, shape=[FLAGS.image_size, FLAGS.image_size, 3],
                                      name='test_x_inputhold')
        test_x_output = cycle_gan.G.sample(tf.expand_dims(test_x_input, 0))
        test_y_input = tf.placeholder(tf.float32, shape=[FLAGS.image_size, FLAGS.image_size, 3],
                                      name='test_y_inputhold')
        test_y_output = cycle_gan.F.sample(tf.expand_dims(test_y_input, 0))

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    with tf.Session(graph=graph, config=config) as sess:
        sess.run(tf.global_variables_initializer())
        if FLAGS.sigma1 != 0:
            aex_latest_ckpt = tf.train.latest_checkpoint(FLAGS.ae_x_ckpt_dir)
            aey_latest_ckpt = tf.train.latest_checkpoint(FLAGS.ae_y_ckpt_dir)
            saver_ae_x.restore(sess, aex_latest_ckpt)
            saver_ae_y.restore(sess, aey_latest_ckpt)
        if FLAGS.whole_ckpt_dir != 'no':
            whole_latest_ckpt = tf.train.latest_checkpoint(FLAGS.whole_ckpt_dir)
            saver.restore(sess, whole_latest_ckpt)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        #GAN part
        if not FLAGS.BEGAN:
            np.set_printoptions(precision=3)
            try:
                step = 0
                fake_Y_pool = ImagePool(FLAGS.pool_size)
                fake_X_pool = ImagePool(FLAGS.pool_size)

                concat_img_list = []

                fake_y_val, fake_x_val = sess.run([fake_y, fake_x])

                x_last_test_predict_list = []
                for last_i in range(len(testimage_x_list)):
                    x_last_test_predict_list.append(0)
                y_last_test_predict_list = []
                for last_i in range(len(testimage_x_list)):
                    y_last_test_predict_list.append(0)

                while not coord.should_stop():

                    if step <= 25:
                        for i in range(FLAGS.dis_pretrain):
                            _, fake_y_val, fake_x_val = sess.run([D_optimizer, fake_y, fake_x],
                                                                 feed_dict={cycle_gan.fake_y: fake_Y_pool.query(fake_y_val),
                                                                            cycle_gan.fake_x: fake_X_pool.query(
                                                                                fake_x_val)})
                        _, G_loss_val, D_Y_loss_val, F_loss_val, D_X_loss_val, fake_y_val, fake_x_val,\
                        real_y, real_x, reconstructed_y_val, reconstructed_x_val, summary, \
                        D_Y_output_real_1_val, D_Y_output_fake_forG_1_val, D_Y_output_real_2_val, D_Y_output_fake_forG_2_val, \
                        D_X_output_real_1_val, D_X_output_fake_forG_1_val, D_X_output_real_2_val, D_X_output_fake_forG_2_val  = \
                            sess.run([optimizers, G_loss, D_Y_loss, F_loss, D_X_loss, fake_y, fake_x, y, x, reconstructed_y,
                                      reconstructed_x, summary_op,
                                      D_Y_output_real_1, D_Y_output_fake_forG_1, D_Y_output_real_2,
                                      D_Y_output_fake_forG_2,
                                      D_X_output_real_1, D_X_output_fake_forG_1, D_X_output_real_2,
                                      D_X_output_fake_forG_2],
                                     feed_dict={cycle_gan.fake_y: fake_Y_pool.query(fake_y_val),
                                                cycle_gan.fake_x: fake_X_pool.query(fake_x_val)})

                    if step > 25:
                        # train with recontruction
                        _, G_loss_val, D_Y_loss_val, F_loss_val, D_X_loss_val, real_y, real_x,\
                        fake_y_val, fake_x_val, reconstructed_y_val, reconstructed_x_val, summary, \
                        D_Y_output_real_1_val, D_Y_output_fake_forG_1_val, D_Y_output_real_2_val, D_Y_output_fake_forG_2_val, \
                        D_X_output_real_1_val, D_X_output_fake_forG_1_val, D_X_output_real_2_val, D_X_output_fake_forG_2_val = (
                            sess.run(
                                [optimizers, G_loss, D_Y_loss, F_loss, D_X_loss, y, x,
                                 fake_y, fake_x, reconstructed_y, reconstructed_x, summary_op,
                                 D_Y_output_real_1, D_Y_output_fake_forG_1, D_Y_output_real_2, D_Y_output_fake_forG_2,
                                 D_X_output_real_1, D_X_output_fake_forG_1, D_X_output_real_2, D_X_output_fake_forG_2],
                                feed_dict={cycle_gan.fake_y: fake_Y_pool.query(fake_y_val),
                                           cycle_gan.fake_x: fake_X_pool.query(fake_x_val)}
                            )
                        )

                    # concatenate generated images and save them as pic
                    if step % 1000 < 10:
                        concat_img_list.append(
                            (real_y[0:1, :, :, :], fake_x_val[0:1, :, :, :], reconstructed_y_val[0:1, :, :, :]))
                        concat_img_list.append(
                            (real_x[0:1, :, :, :], fake_y_val[0:1, :, :, :], reconstructed_x_val[0:1, :, :, :]))
                    #
                    train_writer.add_summary(summary, step)
                    train_writer.flush()

                    if step % 1000 == 9:
                        if len(concat_img_list[0]) == 3:
                            concat_img = concat_3_save(concat_img_list)
                        if len(concat_img_list[0]) == 2:
                            concat_img = concat_save(concat_img_list)
                        misc.imsave(generated_images_dir + "/concat_img_step{}.jpg".format(str(step).zfill(6)), concat_img)
                        concat_img_list = []

                    if step % 100 == 0:
                        logging.info('-----------Step %d:-------------' % step)
                        logging.info('  G_loss   : {}'.format(G_loss_val))
                        logging.info('  D_Y_loss : {}'.format(D_Y_loss_val))
                        logging.info('-----------D_Y_output_real_1')
                        print np.squeeze(D_Y_output_real_1_val)
                        logging.info('-----------D_Y_output_fake_forG_1')
                        print np.squeeze(D_Y_output_fake_forG_1_val)
                        logging.info('-----------D_Y_output_real_2')
                        print np.squeeze(D_Y_output_real_2_val)
                        logging.info('-----------D_Y_output_fake_forG_2')
                        print np.squeeze(D_Y_output_fake_forG_2_val)
                        logging.info('  F_loss   : {}'.format(F_loss_val))
                        logging.info('  D_X_loss : {}'.format(D_X_loss_val))
                        logging.info('-----------D_X_output_real_1')
                        print np.squeeze(D_X_output_real_1_val)
                        logging.info('-----------D_X_output_fake_forG_1')
                        print np.squeeze(D_X_output_fake_forG_1_val)
                        logging.info('-----------D_X_output_real_2')
                        print np.squeeze(D_X_output_real_2_val)
                        logging.info('-----------D_X_output_fake_forG_2')
                        print np.squeeze(D_X_output_fake_forG_2_val)

                    if step % 10000 == 0:
                        try:
                            os.makedirs(testoutput_images_dir + '/step%d' % step)
                        except os.error, e:
                            pass

                        x_change = 0
                        y_change = 0
                        for i in range(test_length):
                            #x part
                            processed_image = testimage_x_list[i] / 127.5 - 1
                            predicted_image = sess.run(test_x_output, feed_dict={test_x_input: processed_image})
                            predicted_image = np.squeeze(predicted_image)
                            current_image_change = compute_change(predicted_image, x_last_test_predict_list[i])
                            x_change = x_change + current_image_change
                            x_last_test_predict_list[i] = np.copy(predicted_image)
                            save_x_image = np.concatenate((testimage_x_list[i], predicted_image), axis=1)
                            #y part
                            processed_image = testimage_y_list[i] / 127.5 - 1
                            predicted_image = sess.run(test_y_output, feed_dict={test_y_input: processed_image})
                            predicted_image = np.squeeze(predicted_image)
                            current_image_change = compute_change(predicted_image, y_last_test_predict_list[i])
                            y_change = y_change + current_image_change
                            y_last_test_predict_list[i] = np.copy(predicted_image)
                            save_y_image = np.concatenate((testimage_y_list[i], predicted_image), axis=1)
                            #save part
                            save_image = np.concatenate((save_x_image, save_y_image), axis=0)
                            scipysave(testoutput_images_dir + '/step{0}/{1}.jpg'.format(step, str(i).zfill(4)), save_image)

                        save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
                        logging.info("Model saved in file: %s" % save_path)
                        logging.info("Testset XXXX : Change on testset: {}".format(x_change))
                        logging.info("Testset YYYY : Change on testset: {}".format(y_change))
                    step += 1

            except KeyboardInterrupt:
                logging.info('Interrupted')
                coord.request_stop()
            except Exception as e:
                coord.request_stop(e)
            finally:
                save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
                logging.info("Model saved in file: %s" % save_path)
                # When done, ask the threads to stop.
                coord.request_stop()
                coord.join(threads)
        #BEGAN part
        else:
            try:
                step = 0
                #fake_Y_pool = ImagePool(FLAGS.pool_size)
                #fake_X_pool = ImagePool(FLAGS.pool_size)

                concat_img_list = []

                #fake_y_val, fake_x_val = sess.run([fake_y, fake_x])
                G_k_t_val, F_k_t_val = sess.run([cycle_gan.G_k_t, cycle_gan.F_k_t])
                while not coord.should_stop():

                    if step <= 25:
                        for i in range(FLAGS.dis_pretrain):
                            _, G_k_t_val, F_k_t_val = sess.run([D_optimizer, G_k_tp, F_k_tp],
                                     feed_dict={cycle_gan.G_k_t: min(max(G_k_t_val, 0), 1),
                                                cycle_gan.F_k_t: min(max(F_k_t_val, 0), 1)})
                        _, G_loss_val, D_Y_loss_val, F_loss_val, D_X_loss_val, real_y, real_x, \
                        fake_y_val, fake_x_val, summary, \
                        G_convergence_measure_val, F_convergence_measure_val, G_k_t_val, F_k_t_val, \
                        G_mu_real_val, G_mu_gen_val, F_mu_real_val, F_mu_gen_val = \
                            sess.run(
                                [optimizers, G_loss, D_Y_loss, F_loss, D_X_loss, y, x,
                                 fake_y, fake_x, summary_op,
                                 G_convergence_measure, F_convergence_measure, G_k_tp, F_k_tp,
                                G_mu_real, G_mu_gen, F_mu_real, F_mu_gen],
                                     feed_dict={cycle_gan.G_k_t: min(max(G_k_t_val, 0), 1),
                                                cycle_gan.F_k_t: min(max(F_k_t_val, 0), 1)})

                    if step > 25:
                        # train with recontruction
                        _, G_loss_val, D_Y_loss_val, F_loss_val, D_X_loss_val, real_y, real_x, \
                        fake_y_val, fake_x_val, summary, \
                        G_convergence_measure_val, F_convergence_measure_val, G_k_t_val, F_k_t_val, \
                        G_mu_real_val, G_mu_gen_val, F_mu_real_val, F_mu_gen_val = (
                            sess.run(
                                [optimizers, G_loss, D_Y_loss, F_loss, D_X_loss, y, x,
                                 fake_y, fake_x, summary_op,
                                 G_convergence_measure, F_convergence_measure, G_k_tp, F_k_tp,
                                G_mu_real, G_mu_gen, F_mu_real, F_mu_gen],
                                feed_dict={cycle_gan.G_k_t: min(max(G_k_t_val, 0), 1),
                                           cycle_gan.F_k_t: min(max(F_k_t_val, 0), 1)}
                            )
                        )

                    # concatenate generated images and save them as pic
                    if step % 1000 < 10:
                        concat_img_list.append(
                            (real_y[0:1, :, :, :], fake_x_val[0:1, :, :, :]))#, reconstructed_y_val[0:1, :, :, :]))
                        concat_img_list.append(
                            (real_x[0:1, :, :, :], fake_y_val[0:1, :, :, :]))#, reconstructed_x_val[0:1, :, :, :]))
                    #
                    train_writer.add_summary(summary, step)
                    train_writer.flush()

                    if step % 1000 == 9:
                        if len(concat_img_list[0]) == 3:
                            concat_img = concat_3_save(concat_img_list)
                        if len(concat_img_list[0]) == 2:
                            concat_img = concat_save(concat_img_list)
                        misc.imsave(generated_images_dir + "/concat_img_step{}.jpg".format(str(step).zfill(6)),
                                    concat_img)
                        concat_img_list = []

                    if step % 100 == 0:
                        logging.info('-----------Step %d:-------------' % step)
                        logging.info('  G_loss   : {}'.format(G_loss_val))
                        logging.info('  D_Y_loss : {}'.format(D_Y_loss_val))
                        logging.info('  F_loss   : {}'.format(F_loss_val))
                        logging.info('  D_X_loss : {}'.format(D_X_loss_val))
                        logging.info('  G_convergence_measure : {}'.format(G_convergence_measure_val))
                        logging.info('  F_convergence_measure : {}'.format(F_convergence_measure_val))
                        logging.info('  G_k_t : {}'.format(G_k_t_val))
                        logging.info('  F_k_t : {}'.format(F_k_t_val))
                        logging.info('  G_mu_real : {}'.format(G_mu_real_val))
                        logging.info('  G_mu_gen : {}'.format(G_mu_gen_val))
                        logging.info('  F_mu_real : {}'.format(F_mu_real_val))
                        logging.info('  F_mu_gen : {}'.format(F_mu_gen_val))

                    if step % 10000 == 0:
                        try:
                            os.makedirs(testoutput_images_dir + '/step%d' % step)
                        except os.error, e:
                            pass

                        for i in range(test_length):
                            processed_image = testimage_x_list[i] / 127.5 - 1
                            predicted_image = sess.run(test_x_output, feed_dict={test_x_input: processed_image})
                            predicted_image = np.squeeze(predicted_image)
                            save_x_image = np.concatenate((testimage_x_list[i], predicted_image), axis=1)
                            processed_image = testimage_y_list[i] / 127.5 - 1
                            predicted_image = sess.run(test_y_output, feed_dict={test_y_input: processed_image})
                            predicted_image = np.squeeze(predicted_image)
                            save_y_image = np.concatenate((testimage_y_list[i], predicted_image), axis=1)
                            save_image = np.concatenate((save_x_image, save_y_image), axis=0)
                            scipysave(testoutput_images_dir + '/step{0}/{1}.jpg'.format(step, str(i).zfill(4)),
                                      save_image)

                        save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
                        logging.info("Model saved in file: %s" % save_path)

                    step += 1

            except KeyboardInterrupt:
                logging.info('Interrupted')
                coord.request_stop()
            except Exception as e:
                coord.request_stop(e)
            finally:
                save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
                logging.info("Model saved in file: %s" % save_path)
                # When done, ask the threads to stop.
                coord.request_stop()
                coord.join(threads)


def main(unused_argv):
    train()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    tf.app.run()
