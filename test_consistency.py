#coding=utf-8
import scipy.misc
import os
import numpy as np
try:
  from os import scandir
except ImportError:
  # Python 2 polyfill module
  from scandir import scandir
from matplotlib import pyplot as plt

def data_reader(input_dir):
  file_paths = []

  for img_file in scandir(input_dir):
    if img_file.name.endswith('.jpg') and img_file.is_file():
      file_paths.append(img_file.path)
  return file_paths

def data_list_reader(input_dir, image_size=None):
  filepaths_list = sorted(data_reader(input_dir))
  image_list = []
  for file in filepaths_list:
    tmp_image = scipy.misc.imread(file)
    if image_size is not None:
      tmp_image = scipy.misc.imresize(tmp_image, (image_size, image_size, 3))
    image_list.append(tmp_image)
  return image_list

def scipysave(name, image):
  scipy.misc.imsave(name, image)

def compute_change(x, y):
    return np.sum(np.square(x-y))


testimage_dir = ''
testimage_list = data_list_reader(testimage_dir)

realimage_dir = ''
realimage_list = data_list_reader(realimage_dir, 128)


total_distance=0
for i in range(len(testimage_list)):
    print i
    '''
    plt.subplot(311)
    plt.imshow(testimage_list[i][:,128:])
    plt.subplot(312)
    plt.imshow(realimage_list[i])
    #plt.show()
    '''
    tmp_change = compute_change(testimage_list[i][:,128:]/1.0, realimage_list[i]/1.0)
    '''
    plt.subplot(313)
    plt.imshow(testimage_list[i][:, 128:] - realimage_list[i])
    plt.show()
    '''
    #exit(0)
    print tmp_change/128/128/3
    total_distance += tmp_change
print total_distance

#cnt=0
#total_distance = 0
#for testimage in testimage_list:
#        tmp_distance = compute_change(testimage, realimage)
#        if tmp_distance < this_image_min:
#            this_image_min = tmp_distance
#    total_distance = total_distance + this_image_min
#    print cnt
#    cnt += 1

