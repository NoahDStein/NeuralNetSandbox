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

import numpy
import scipy.stats
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from tfutil import LayerManager, log

ROWS = 10
COLS = 10
POSRANK = 13

TRAIN_SIZE = 60000
TEST_SIZE = 10000

SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 64
PRIOR_BATCH_SIZE = 10

TRAIN = True
NUM_HIDDEN_LAYERS = 2
HIDDEN_LAYER_SIZE = 500


def rand_nmf(rows, cols, pos_rank, num_examples):
    a = numpy.maximum(0, numpy.random.randn(num_examples, rows, pos_rank))
    b = numpy.maximum(0, numpy.random.randn(num_examples, pos_rank, cols))
    c = numpy.einsum('ijk,ikl->ijl', a, b)
    return a, b, c


def id_act(z):
    return z

def double_relu(z):
    return [tf.nn.relu(z), tf.nn.relu(-z)]

default_act = tf.nn.relu  # double_relu
do_bn = dict(bn=False)

def train():
    print("\nSource code of training file {}:\n\n{}".format(__file__, open(__file__).read()))

    # Import data
    log('simulating data')

    numpy.random.seed(3737)
    test_data_all = rand_nmf(ROWS, COLS, POSRANK, TEST_SIZE)
    if TRAIN:
        train_data_all = rand_nmf(ROWS, COLS, POSRANK, TRAIN_SIZE)
    else: # Don't waste time computing training data
        train_data_all = [numpy.zeros((TRAIN_SIZE, ROWS, POSRANK)), numpy.zeros((TRAIN_SIZE, POSRANK, COLS)), numpy.zeros((TRAIN_SIZE, ROWS, COLS))]
    log('done simulating')

    test_data = test_data_all[2].reshape(TEST_SIZE, ROWS*COLS)
    train_data = train_data_all[2].reshape(TRAIN_SIZE, ROWS*COLS)

    def factorizer(lm, data):
        last = [(data-train_data.mean())/train_data.std()]
        randomness = tf.random_normal(tf.shape(data))[:, :10]  # Stupid hack
        #last.append(randomness)
        for i in xrange(NUM_HIDDEN_LAYERS):
            new_layer = lm.nn_layer(last, HIDDEN_LAYER_SIZE, 'factorizer/hidden/fc{}'.format(i), act=default_act, **do_bn)
            #last.append(new_layer)
            last[0] = new_layer
        a = lm.nn_layer(last, ROWS * POSRANK, 'factorizer/output/a', act=tf.nn.relu)
        b = lm.nn_layer(last, POSRANK * COLS, 'factorizer/output/b', act=tf.nn.relu)
        return a, b

    def full_model(lm, data):
        a, b = factorizer(lm, data)
        chat = tf.reshape(tf.batch_matmul(tf.reshape(a, (-1, ROWS, POSRANK)), tf.reshape(b, (-1, POSRANK, COLS))),
                          (-1, ROWS * COLS))

        with tf.name_scope('error'):
            squared_error = tf.reduce_mean(tf.reduce_sum((chat - data) ** 2, reduction_indices=[1]))
            lm.summaries.scalar_summary('squared_error', squared_error)

        return a, b, chat, squared_error


    lm = LayerManager(forward_biased_estimate=False)
    batch = tf.Variable(0)

    with tf.name_scope('input'):
        all_train_data_initializer = tf.placeholder(tf.float32, [TRAIN_SIZE, ROWS*COLS])
        all_train_data = tf.Variable(all_train_data_initializer, trainable=False, collections=[])
        random_training_example = tf.train.slice_input_producer([all_train_data])
        training_batch = tf.train.batch(random_training_example, batch_size=BATCH_SIZE, enqueue_many=False)
        fed_input_data = tf.placeholder(tf.float32, [None, ROWS*COLS])

    with tf.name_scope('posterior'):
        _, _, _, train_squared_error = full_model(lm, training_batch)
    training_merged = lm.summaries.merge_all_summaries()
    lm.is_training = False
    tf.get_variable_scope().reuse_variables()
    lm.summaries.reset()
    with tf.name_scope('test'):
        _, _, _, test_squared_error = full_model(lm, fed_input_data)
    test_merged = lm.summaries.merge_all_summaries()

    saver = tf.train.Saver(tf.trainable_variables() + tf.get_collection('BatchNormInternal'))

    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, batch, 5000, 0.8, staircase=True)

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(train_squared_error, global_step=batch, var_list=lm.weight_factory.variables + lm.bias_factory.variables + lm.scale_factory.variables)

    def feed_dict(mode):
        """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
        if mode == 'test':
            return {fed_input_data: test_data}
        else:
            return {}

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        sess.run([all_train_data.initializer], feed_dict={all_train_data_initializer: train_data})
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
                        summary, err = sess.run([test_merged, test_squared_error], feed_dict=feed_dict('test'))
                        test_writer.add_summary(summary, i)
                        log('batch %s: Test error = %s' % (i, err))
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
    flags.DEFINE_string('summaries_dir', '/tmp/nmf/logs', 'Summaries directory')
    flags.DEFINE_string('train_dir', '/tmp/nmf/save', 'Saves directory')

    tf.app.run()
