'''
Basic operation on tensorflow

Updated on 2017.07.31
Author : Yeonwoo Jeong
'''
from tensorflow.examples.tutorials.mnist import input_data
from config import MNIST_PATH
from utils import struct
import tensorflow as tf
import numpy as np

class mnist_for_gan(object):
    """Proper mnist manager for only GAN"""

    def __init__(self, style = 'conv'):
        '''
        Args:
            style - string
                'conv' to return 4D array setting
                otherwise return 2D array setting
        '''
        self.data = input_data.read_data_sets(MNIST_PATH, one_hot = True)
        self.style = style
        if self.style == 'conv':
            self.size = 28
            self.channel = 1
        else:
            self.size = 784

    def __call__(self, batch_size):
        batch_image, _ = self.data.train.next_batch(batch_size)
        if self.style == 'conv':
            batch_image = np.reshape(batch_image, [-1, self.size, self.size, self.channel])
        return batch_image

def mnistloader():
    '''
    Return :
        struct is defined in utils.py

        train - mnist train set with struct 
        test - mnist test set with struct
        val - mnist validation set with struct
    '''
    mnist = input_data.read_data_sets(MNIST_PATH, one_hot = True)
    train = struct()
    test = struct()
    val = struct()
    train.image = mnist.train.images
    train.label = mnist.train.labels
    test.image = mnist.test.images
    test.label = mnist.test.labels
    val.image = mnist.validation.images
    val.label = mnist.validation.labels
    return train, test, val

def get_shape(tensor):
    '''return the shape of tensor as list'''
    return tensor.get_shape().as_list()

def print_vars(string):
    '''print variables in collection named string'''
    print("Collection name %s"%string)
    print([v.name for v in tf.get_collection(string)])

def leaky_relu(x, leak = 0.2):
    '''Simple implementation of leaky relu'''
    f1 = 0.5*(1+leak)
    f2 = 0.5*(1-leak)
    return tf.add(tf.multiply(f1, x), tf.multiply(f2, tf.abs(x)))

def flatten(x):
    '''
    Flatten tensor into 2D tensor
    Args
        x - (n+1) tensor
            [batch_size, d_1, d_2, ..., d_n]
    Return
        flatten - 2D tensor
            [batch_size, d_1*d2*...*d_n]
    '''
    x_shape = get_shape(x)
    flatten = tf.reshape(x, [x_shape[0], -1])
    return flatten

def clip(x, vmin = 1e-7, vmax = 1-1e-7):
    '''
    Args:
        x - tensor
            tensor to be clipped
        vmin - float
            default to be 1e-7m clip value min
        vmax - float
            default to be 1-1e-7, clip value max
    '''
    return tf.clip_by_value(x, clip_value_min=vmin, clip_value_max=vmax)

def optimizer(loss, var_list, initial_lr = 1e-4):
    '''
    Args:
        loss - tensor
            loss that should be minimized
        var_list - list of tensor 
            tensors to be optimized 
        initial_lr - float
            default to be 1e-4
    Return:
        Adamoptimizer to reduce loss for var_list
    '''
    return tf.train.AdamOptimizer(initial_lr).minimize(loss, var_list = var_list)


def deconvolution(input_, filter_shape, output_shape = None, strides = [1,1,1,1], padding = True, activation = tf.nn.relu, batch_norm = False, istrain = False, scope = None):
    '''
    Args :
        input_ - 4D tensor
            [batch, height, width, inchannel]
        filter_shape - 1D array or list with 4 elements
            [height, width, outchannel, inchannel]
        output_shape - 1D array or list with 4 elements
            [-1, height, width, outchannel]
        strides - 1D array with elements
            default to be [1,1,1,1]
        padding - bool
            default to be True
                True 'VALID'
                False 'SAME'
        activation - activation function
            default to be tf.nn.relu
        scope - string
            default to be None
    '''
    input_shape = get_shape(input_)

    with tf.variable_scope(scope or "deconv"):
        if output_shape is None:
            output_shape = [input_shape[0], input_shape[1], input_shape[2], filter_shape[-2]]

        assert input_shape[-1]==filter_shape[-1], "inchannel value of input, and filter should be same"
        assert output_shape[-1]==filter_shape[-2], "outchannel value of output, and filter should be same"

        if padding:
            padding = 'SAME'
        else:
            padding = 'VALID'
        w = tf.get_variable(name="w", shape = filter_shape, initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False))
        deconv = tf.nn.conv2d_transpose(input_, w, output_shape = output_shape, strides=strides, padding=padding)
        if batch_norm:
            norm = tf.contrib.layers.batch_norm(deconv, center=True, scale=True, decay = 0.8, is_training=istrain, scope='batch_norm')
            return activation(norm)
        else:
            b = tf.get_variable(name="b", shape = output_shape[-1], initializer=tf.constant_initializer(0.01))
            return activation(deconv + b)

def convolution(input_, filter_shape, strides = [1,1,1,1], padding = False, activation = None, batch_norm = False, istrain = False, scope = None):
    '''
    Args:
        input_ - 4D tensor
            Normally NHWC format
        filter_shape - 1D array 4 elements
            [height, width, inchannel, outchannel]
        strides - 1D array 4 elements
            default to be [1,1,1,1]
        padding - bool
            Deteremines whether add padding or not
            True => add padding 'SAME'
            Fale => no padding  'VALID'
        activation - activation function
            default to be None
        batch_norm - bool
            default to be False
            used to add batch-normalization
        istrain - bool
            indicate the model whether train or not
        scope - string
            default to be None
    Return:
        4D tensor
        activation(batch(conv(input_)))
    '''
    with tf.variable_scope(scope or "conv"):
        if padding:
            padding = 'SAME'
        else:
            padding = 'VALID'
        w = tf.get_variable(name="w", shape = filter_shape, initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False)) 
        conv = tf.nn.conv2d(input_, w, strides=strides, padding=padding)
        if batch_norm:
            norm = tf.contrib.layers.batch_norm(conv, center=True, scale=True, decay = 0.8, is_training=istrain, scope='batch_norm')
            if activation is None:
                return norm
            return activation(norm)
        else:
            b = tf.get_variable(name="b", shape = filter_shape[-1], initializer=tf.constant_initializer(0.001))
            if activation is None:
                return conv + b
            return activation(conv + b)

def fc_layer(input_, output_size, activation = None, batch_norm = False, istrain = False, scope = None):
    '''
    fully convlolution layer
    Args :
        input_  - 2D tensor
            general shape : [batch, input_size]
        output_size - int
            shape of output 2D tensor
        activation - activation function
            defaults to be None
        batch_norm - bool
            defaults to be False
            if batch_norm to apply batch_normalization
        istrain - bool
            defaults to be False
            indicator for phase train or not
        scope - string
            defaults to be None then scope becomes "fc"
    '''
    with tf.variable_scope(scope or "fc"):
        w = tf.get_variable(name="w", shape = [get_shape(input_)[1], output_size], initializer=tf.contrib.layers.xavier_initializer()) 
        if batch_norm:
            norm = tf.contrib.layers.batch_norm(tf.matmul(input_, w) , center=True, scale=True, decay = 0.8, is_training=istrain, scope='batch_norm')
            if activation is None:
                return norm
            return activation(norm)
        else:
            b = tf.get_variable(name="b", shape = [output_size], initializer=tf.constant_initializer(0.0))
            if activation is None:
                return tf.nn.xw_plus_b(input_, w, b)
            return activation(tf.nn.xw_plus_b(input_, w, b))

def softmax_cross_entropy(logits, labels):
    '''softmax_cross_entropy, lables : correct label logits : predicts'''
    return tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)

def sigmoid_cross_entropy(logits, labels):
    '''softmax_cross_entropy, lables : correct label logits : predicts'''
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
