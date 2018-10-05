import tensorflow as tf
import random
import numpy as np
import scipy.misc

try:
  from os import scandir
except ImportError:
  # Python 2 polyfill module
  from scandir import scandir


def data_reader(input_dir):
  file_paths = []

  for img_file in scandir(input_dir):
    if img_file.name.endswith('.jpg') and img_file.is_file():
      file_paths.append(img_file.path)
  return file_paths

def data_list_reader(input_dir, image_size):
  filepaths_list = sorted(data_reader(input_dir))
  image_list = []
  for file in filepaths_list:
    tmp_image = scipy.misc.imread(file)
    if image_size is not None:
      tmp_image = scipy.misc.imresize(tmp_image, (image_size, image_size, 3))
    image_list.append(tmp_image)
  return image_list

def convert2int(image):
  """ Transfrom from float tensor ([-1.,1.]) to int image ([0,255])
  """
  return tf.image.convert_image_dtype((image+1.0)/2.0, tf.uint8)

def convert2float(image):
  """ Transfrom from int image ([0,255]) to float tensor ([-1.,1.])
  """
  #image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  return (image/127.5) - 1.0

def batch_convert2int(images):
  """
  Args:
    images: 4D float tensor (batch_size, image_size, image_size, depth)
  Returns:
    4D int tensor
  """
  return tf.map_fn(convert2int, images, dtype=tf.uint8)

def batch_convert2float(images):
  """
  Args:
    images: 4D int tensor (batch_size, image_size, image_size, depth)
  Returns:
    4D float tensor
  """
  return tf.map_fn(convert2float, images, dtype=tf.float32)

class ImagePool:
  """ History of generated images
      Same logic as https://github.com/junyanz/CycleGAN/blob/master/util/image_pool.lua
  """
  def __init__(self, pool_size):
    self.pool_size = pool_size
    self.images = []

  def query(self, image, threshold=0.5):
    if self.pool_size == 0:
      return image

    if len(self.images) < self.pool_size:
      self.images.append(image)
      return image
    else:
      p = random.random()
      if p > threshold:
        # use old image
        random_id = random.randrange(0, self.pool_size)
        tmp = self.images[random_id].copy()
        self.images[random_id] = image.copy()
        return tmp
      else:
        return image

def npConvert2int(image):
  return np.squeeze(np.asarray((image + 1.) * 127.5, dtype=np.uint8))

def concat_save(concat_img_list):
  real_x, fake_y = concat_img_list.pop(0)
  real_x = real_x[0:1, :, :, :]
  fake_y = fake_y[0:1, :, :, :]
  output = np.concatenate((real_x, fake_y), axis=1)
  for real_x, fake_y in concat_img_list:
    real_x = real_x[0:1, :, :, :]
    fake_y = fake_y[0:1, :, :, :]
    tmp_img = np.concatenate((real_x, fake_y), axis=1)
    output = np.concatenate((output, tmp_img), axis=2)
  return npConvert2int(output)

'''
def concat_3_save(concat_img_list):
  real_x, fake_y, reconstructed_x = concat_img_list.pop(0)
  real_x = npConvert2int(real_x)[0:1, :, :, :]
  fake_y = npConvert2int(fake_y)[0:1, :, :, :]
  reconstructed_x = npConvert2int(reconstructed_x)[0:1, :, :, :]
  print real_x.shape, fake_y.shape, reconstructed_x.shape
  output = np.concatenate((real_x, fake_y, reconstructed_x), axis=0)
  print output.shape
  for real_x, fake_y, reconstructed_x in concat_img_list:
    real_x = npConvert2int(real_x)[0:1, :, :, :]
    fake_y = npConvert2int(fake_y)[0:1, :, :, :]
    reconstructed_x = npConvert2int(reconstructed_x)[0:1, :, :, :]
    tmp_img = np.concatenate((real_x, fake_y, reconstructed_x), axis=0)
    output = np.concatenate((output, tmp_img), axis=1)
    print output.shape

  return output
'''
def concat_3_save(concat_img_list):
  real_x, fake_y, reconstructed_x = concat_img_list.pop(0)
  real_x = real_x[0:1, :, :, :]
  fake_y = fake_y[0:1, :, :, :]
  reconstructed_x = reconstructed_x[0:1, :, :, :]
  output = np.concatenate((real_x, fake_y, reconstructed_x), axis=1)
  for real_x, fake_y, reconstructed_x in concat_img_list:
    real_x = real_x[0:1, :, :, :]
    fake_y = fake_y[0:1, :, :, :]
    reconstructed_x = reconstructed_x[0:1, :, :, :]
    tmp_img = np.concatenate((real_x, fake_y, reconstructed_x), axis=1)
    output = np.concatenate((output, tmp_img), axis=2)
  return npConvert2int(output)

def scipysave(name, image):
  scipy.misc.imsave(name, image)

