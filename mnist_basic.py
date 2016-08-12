# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A simple MNIST classifier which displays summaries in TensorBoard.
 This is an unimpressive MNIST model, but it is a good example of using
tf.name_scope to make a graph legible in the TensorBoard graph explorer, and of
naming summary tags so that they are grouped meaningfully in TensorBoard.
It demonstrates the functionality of every TensorBoard dashboard.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os
import matplotlib.pyplot as plt
import numpy
import scipy.stats
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import constant_op
from tensorflow.python.ops import math_ops

from tfutil import LayerManager, log

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
IMAGE_SIZE = 28
IMAGE_AREA = IMAGE_SIZE*IMAGE_SIZE
NUM_CLASSES = 10
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 64
PRIOR_BATCH_SIZE = 10

TRAIN = True
CONV = False
NUM_HIDDEN_LAYERS = 2
HIDDEN_LAYER_SIZE = 500

if CONV:
    small_image_size = IMAGE_SIZE // 4
    small_image_area = small_image_size * small_image_size
    HIDDEN_LAYER_SIZE = (HIDDEN_LAYER_SIZE // small_image_area) * small_image_area

def id_act(z):
    return z

def double_relu(z):
    return [tf.nn.relu(z), tf.nn.relu(-z)]

default_act = tf.nn.relu  # double_relu
do_bn = dict(bn=False)

def classifier(lm, data):
    last = data - 0.5
    if CONV:
        last = tf.reshape(last, [-1, IMAGE_SIZE, IMAGE_SIZE, 1])
        last = lm.conv_layer(last, 3, 3, 32, 'classifier/hidden/conv0', act=default_act, **do_bn)
        last = lm.conv_layer(last, 3, 3, 32, 'classifier/hidden/conv1', act=default_act, **do_bn)
        last = lm.conv_layer(last, 3, 3, 32, 'classifier/hidden/conv2', act=default_act, **do_bn)
        last = lm.max_pool(last, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        last = lm.conv_layer(last, 3, 3, 64, 'classifier/hidden/conv3', act=default_act, **do_bn)
        last = lm.conv_layer(last, 3, 3, 64, 'classifier/hidden/conv4', act=default_act, **do_bn)
        last = lm.conv_layer(last, 3, 3, 64, 'classifier/hidden/conv5', act=default_act, **do_bn)
        last = lm.max_pool(last, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        shape = last.get_shape().as_list()
        last = tf.reshape(last, [-1, shape[1] * shape[2] * shape[3]])
    for i in xrange(NUM_HIDDEN_LAYERS):
        last = lm.nn_layer(last, HIDDEN_LAYER_SIZE, 'classifier/hidden/fc{}'.format(i), act=default_act, **do_bn)
        last = tf.nn.dropout(last, 0.7)
    last = lm.nn_layer(last, NUM_CLASSES, 'classifier/output/logits', act=id_act)
    return last

def full_model(lm, data, labels):
    output_logits = classifier(lm, data)
    output_probs = tf.nn.softmax(output_logits)

    with tf.name_scope('error'):
        cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(output_logits, labels))
        lm.summaries.scalar_summary('cross_entropy', cross_entropy)
        percent_error = 100.0 * tf.reduce_mean(
            tf.cast(tf.not_equal(tf.arg_max(output_logits, dimension=1), labels), tf.float32))
        lm.summaries.scalar_summary('percent_error', percent_error)

    return output_probs, cross_entropy, percent_error

def train():
    log('loading MNIST')
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=False)
    TRAIN_SIZE=mnist.train.images.shape[0]

    lm = LayerManager(forward_biased_estimate=False)
    batch = tf.Variable(0)

    with tf.name_scope('input'):
        all_train_data_initializer = tf.placeholder(tf.float32, [TRAIN_SIZE, IMAGE_AREA])
        all_train_labels_initializer = tf.placeholder(tf.int64, [TRAIN_SIZE])
        all_train_data = tf.Variable(all_train_data_initializer, trainable=False, collections=[])
        all_train_labels = tf.Variable(all_train_labels_initializer, trainable=False, collections=[])
        random_training_example = tf.train.slice_input_producer([all_train_data, all_train_labels])
        training_batch = tf.train.batch(random_training_example, batch_size=BATCH_SIZE, enqueue_many=False)
        fed_input_data = tf.placeholder(tf.float32, [None, IMAGE_AREA])
        fed_input_labels = tf.placeholder(tf.int64, [None])

    with tf.name_scope('posterior'):
        training_output, training_cross_entropy, training_percent_error = full_model(lm, *training_batch)
    training_merged = lm.summaries.merge_all_summaries()
    lm.is_training = False
    tf.get_variable_scope().reuse_variables()
    lm.summaries.reset()
    with tf.name_scope('test'):
        test_output, test_cross_entropy, test_percent_error = full_model(lm, fed_input_data, fed_input_labels)
    test_merged = lm.summaries.merge_all_summaries()

    saver = tf.train.Saver(tf.trainable_variables() + tf.get_collection('BatchNormInternal'))

    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, batch, 5000, 0.8, staircase=True)

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(training_cross_entropy, global_step=batch, var_list=lm.weight_factory.variables + lm.bias_factory.variables + lm.scale_factory.variables)

    def feed_dict(mode):
        """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
        if mode == 'test':
            return {fed_input_data: mnist.test.images, fed_input_labels: mnist.test.labels}
        else:
            return {}

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        sess.run([all_train_data.initializer, all_train_labels.initializer], feed_dict={all_train_data_initializer: mnist.train.images, all_train_labels_initializer: mnist.train.labels})
        sess.run(tf.initialize_variables(tf.get_collection('BatchNormInternal')))

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        if TRAIN:
            train_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/train', sess.graph)
            test_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/test')
            try:
                log('starting training')
                for i in xrange(FLAGS.max_steps):
                    if i % 1000 == 999: # Do test set
                        summary, err = sess.run([test_merged, test_percent_error], feed_dict=feed_dict('test'))
                        test_writer.add_summary(summary, i)
                        log('batch %s: Test classification error = %s%%' % (i, err))
                    if i % 5000 == 4999:
                        NUM_RUNS = 100
                        runs = []
                        for _ in xrange(NUM_RUNS):
                            new_output_probs, = sess.run([test_output], feed_dict=feed_dict('test'))
                            new_output = numpy.argmax(new_output_probs, 1)
                            runs.append(new_output)

                        all_runs = numpy.vstack(runs).T
                        ave_entropy = numpy.mean([scipy.stats.entropy(numpy.bincount(row), base=2.0) for row in all_runs])
                        log('batch %s: Average entropy = %.4f bits' % (i, ave_entropy))
                    if i % 100 == 99: # Record a summary
                        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata = tf.RunMetadata()
                        summary, _ = sess.run([training_merged, train_step],
                                              feed_dict=feed_dict('train'),
                                              options=run_options,
                                              run_metadata=run_metadata)
                        train_writer.add_summary(summary, i)
                        train_writer.add_run_metadata(run_metadata, 'batch%d' % i)
                    else:
                        sess.run([train_step], feed_dict=feed_dict('train'))
            finally:
                log('saving')
                saver.save(sess, FLAGS.train_dir, global_step=batch)
                log('done')

        coord.request_stop()
        coord.join(threads)
        sess.close()

def main(_):
    if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    tf.gfile.MakeDirs(FLAGS.summaries_dir)
    train()

if __name__ == '__main__':
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_integer('max_steps', 100000, 'Number of steps to run trainer.')
    flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
    flags.DEFINE_string('data_dir', '/tmp/data', 'Directory for storing data')
    flags.DEFINE_string('summaries_dir', '/tmp/mnist_basic/logs', 'Summaries directory')
    flags.DEFINE_string('train_dir', '/tmp/mnist_basic/save', 'Saves directory')

    tf.app.run()
