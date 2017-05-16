import cifar10.cifar10
import tensorflow as tf
import numpy as np
import time

max_steps=3000
batch_size=128
data_dir='cifar10_data/cifar-10-batches-bin'

def variable_with_weight_loss(shape, stddev, wl):
    var=tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if wl is not None:
        weight_loss=tf.multiply(tf.nn.l2_loss(var), wl, name='weight_loss')
        tf.add_to_collection('losses', weight_loss)
    return var

cifar10.maybe_download_and_extract()

