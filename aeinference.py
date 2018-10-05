
import tensorflow as tf
import os
from model import CycleGAN
import utils

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

    file_list = data_reader(FLAGS.input)
    whole = len(file_list)
    cnt = 0
    with tf.Session(graph=graph) as sess:
      for file in file_list:
        with tf.gfile.FastGFile(file, 'rb') as f:
          image_data = f.read()
          input_image = tf.image.decode_jpeg(image_data, channels=3)
          input_image = tf.image.resize_images(input_image, size=(FLAGS.image_size, FLAGS.image_size))
          input_image = utils.convert2float(input_image)
          input_image.set_shape([FLAGS.image_size, FLAGS.image_size, 3])
          #input_image_list.append(input_image)
        print cnt
        [output_image] = tf.import_graph_def(graph_def,
                              input_map={'input_image': input_image},
                              return_elements=['output_image:0'],
                              name='output')
        print cnt
        generated = output_image.eval()
        print cnt
        output_file_name = file.split('/')[-1]
        with open(FLAGS.output + '/fake_{}'.format(output_file_name), 'wb') as f:
          f.write(generated)
        cnt += 1
        if cnt/whole > 0.05:
          print cnt/whole, 'done'



def main(unused_argv):
  inference()

if __name__ == '__main__':
  tf.app.run()
