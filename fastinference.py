"""
A fast version of the original inference.
Constructing one graph to infer all the samples.
Originaly one graph for each sample.
"""

import tensorflow as tf
import os
from model import CycleGAN
import utils
import scipy.misc
import numpy as np

try:
  from os import scandir
except ImportError:
  # Python 2 polyfill module
  from scandir import scandir

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('model', '', 'model path (.pb)')
tf.flags.DEFINE_string('input', 'data/apple', 'input image path')
tf.flags.DEFINE_string('output', 'samples/apple', 'output image path')
tf.flags.DEFINE_integer('image_size', 128, 'image size, default: 128')

def data_reader(input_dir):
  file_paths = []

  for img_file in scandir(input_dir):
    if img_file.name.endswith('.jpg') and img_file.is_file():
      file_paths.append(img_file.path)
  return file_paths

def inference():
  graph = tf.Graph()

  with graph.as_default():

    with tf.gfile.FastGFile(FLAGS.model, 'rb') as model_file:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(model_file.read())

    input_image = tf.placeholder(tf.float32,shape=[FLAGS.image_size, FLAGS.image_size, 3])
    [output_image] = tf.import_graph_def(graph_def,
                                         input_map={'input_image': input_image},
                                         return_elements=['output_image:0'],
                                         name='output')
    #print type(output_image), output_image
    file_list = data_reader(FLAGS.input)
    whole = len(file_list)
    cnt = 0
    with tf.Session(graph=graph) as sess:
      for file in file_list:
        tmp_image = scipy.misc.imread(file)
        tmp_image = scipy.misc.imresize(tmp_image, (FLAGS.image_size, FLAGS.image_size, 3))
        processed_image = tmp_image / 127.5 - 1
        processed_image = np.asarray(processed_image, dtype=np.float32)
        predicted_image = sess.run(output_image, feed_dict={input_image: processed_image})
        predicted_image = np.squeeze(predicted_image)
        #print tmp_image.shape, predicted_image.shape
        save_image = np.concatenate((tmp_image, predicted_image), axis=1)
        print cnt
        output_file_name = file.split('/')[-1]
        try:
          os.makedirs(FLAGS.output)
        except os.error, e:
          pass
        scipy.misc.imsave(FLAGS.output + '/{}'.format(output_file_name), save_image)
        cnt += 1
        if cnt//whole > 0.05:
          print cnt//whole, 'done'



def main(unused_argv):
  inference()

if __name__ == '__main__':
  tf.app.run()
