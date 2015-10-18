__author__ = 'sorkhei'

import numpy as np
import struct

import matplotlib.pyplot as plt
import matplotlib as mpl
import os


def read_mnist_training_data(N=60000):
    """
    :param N: the number of digits to be read, default is value is set to maximum = 60000
    :return: a list of tuples (X, y). X is a 28 by 28 image and y the corresponding label, a number between 0 and 9
    """
    files = os.listdir(os.getcwd())
    if 'train-images-idx3-ubyte' not in files or 'train-labels-idx1-ubyte' not in files:
        exit('training data not found')
    train_image = open('train-images-idx3-ubyte', 'rb')
    train_label = open('train-labels-idx1-ubyte', 'rb')
    _, _ = struct.unpack('>II', train_label.read(8))
    labels = np.fromfile(train_label, dtype=np.int8)
    _, _, img_row, img_col = struct.unpack('>IIII', train_image.read(16))
    images = np.fromfile(train_image, dtype=np.uint8).reshape(len(labels), img_row, img_col)
    image_label_list = [(images[i], labels[i]) for i in xrange(len(labels))]
    return image_label_list[0:N]


def visualize(image):
    """
    :param image: is a 28 by 28 image which is going to be displayed
    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    plt.show()

