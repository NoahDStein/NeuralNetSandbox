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

from tfutil import LayerManager, listify

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('max_steps', 100000, 'Number of steps to run trainer.')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_string('data_dir', '/tmp/data', 'Directory for storing data')
flags.DEFINE_string('summaries_dir', '/tmp/mnist/logs', 'Summaries directory')
flags.DEFINE_string('train_dir', '/tmp/mnist/save', 'Saves directory')

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
IMAGE_SIZE = 28
IMAGE_AREA = IMAGE_SIZE*IMAGE_SIZE
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 64
PRIOR_BATCH_SIZE = 10

TRAIN = True
BNAE = False
CONV = False
LATENT_DIM = 20
NUM_HIDDEN_LAYERS = 2
HIDDEN_LAYER_SIZE = 500

if CONV:
    small_image_size = IMAGE_SIZE // 4
    small_image_area = small_image_size * small_image_size
    HIDDEN_LAYER_SIZE = (HIDDEN_LAYER_SIZE // small_image_area) * small_image_area

def log(s):
    print('[%s] ' % time.asctime() + s)

# Adapted from tf's clip_ops.py, so this probably inherits the Apache license
def clip_rows_by_norm(t, clip_norm, name=None):
    with ops.op_scope([t, clip_norm], name, "clip_by_norm") as name:
        t = ops.convert_to_tensor(t, name="t")

        # Calculate L2-norm, clip elements by ratio of clip_norm to L2-norm
        l2norm_inv = math_ops.rsqrt(
            math_ops.reduce_sum(t * t, reduction_indices=[1], keep_dims=True))
        tclip = array_ops.identity(t * clip_norm * math_ops.minimum(
            l2norm_inv, tf.cast(constant_op.constant(1.0 / clip_norm), t.dtype)), name=name)

    return tclip


def train():
    log('loading MNIST')
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=False)
    TRAIN_SIZE=mnist.train.images.shape[0]

    lm = LayerManager(forward_biased_estimate=False)
    batch = tf.Variable(0)

    with tf.name_scope('input'):
        all_train_data_initializer = tf.placeholder(tf.float32, [TRAIN_SIZE, IMAGE_AREA])
        all_train_data = tf.Variable(all_train_data_initializer, trainable=False, collections=[])
        random_training_example = tf.train.slice_input_producer([all_train_data])
        training_batch = tf.train.batch([random_training_example], batch_size=BATCH_SIZE, enqueue_many=True)
        fed_input_data = tf.placeholder(tf.float32, [None, IMAGE_AREA])

    def id_act(z):
        return z

    def log_std_act(z):
        return tf.clip_by_value(z, -2.0, 2.0)

    def double_relu(z):
        return [tf.nn.relu(z), tf.nn.relu(-z)]

    default_act = tf.nn.relu  # double_relu
    do_bn = dict(bn=False)
    def encoder(data):
        last = data - 0.5
        if CONV:
            last = tf.reshape(last, [-1, IMAGE_SIZE, IMAGE_SIZE, 1])
            last = lm.conv_layer(last, 3, 3, 32, 'encoder/hidden/conv0', act=default_act, **do_bn)
            last = lm.conv_layer(last, 3, 3, 32, 'encoder/hidden/conv1', act=default_act, **do_bn)
            last = lm.conv_layer(last, 3, 3, 32, 'encoder/hidden/conv2', act=default_act, **do_bn)
            last = lm.max_pool(last, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            last = lm.conv_layer(last, 3, 3, 64, 'encoder/hidden/conv3', act=default_act, **do_bn)
            last = lm.conv_layer(last, 3, 3, 64, 'encoder/hidden/conv4', act=default_act, **do_bn)
            last = lm.conv_layer(last, 3, 3, 64, 'encoder/hidden/conv5', act=default_act, **do_bn)
            last = lm.max_pool(last, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            shape = last.get_shape().as_list()
            last = tf.reshape(last, [-1, shape[1] * shape[2] * shape[3]])
        for i in xrange(NUM_HIDDEN_LAYERS):
            last = lm.nn_layer(last, HIDDEN_LAYER_SIZE, 'encoder/hidden/fc{}'.format(i), act=default_act, **do_bn)
        if BNAE:
            latent_mean = lm.nn_layer(last, LATENT_DIM, 'latent/mean', act=id_act)#, bias=False, scale=False, **do_bn)
        else:
            latent_mean = lm.nn_layer(last, LATENT_DIM, 'latent/mean', act=id_act, **do_bn)
        latent_log_std = lm.nn_layer(last, LATENT_DIM, 'latent/log_std', act=log_std_act, **do_bn)
        return latent_mean, latent_log_std

    def decoder(code):
        last = code
        for i in xrange(NUM_HIDDEN_LAYERS):
            last = lm.nn_layer(last, HIDDEN_LAYER_SIZE, 'decoder/hidden/fc{}'.format(i), act=default_act, **do_bn)
        if CONV:
            last_num_filters = HIDDEN_LAYER_SIZE // ((IMAGE_SIZE // 4)*(IMAGE_SIZE // 4))
            last = [tf.reshape(val, [-1, IMAGE_SIZE // 4, IMAGE_SIZE // 4, last_num_filters]) for val in listify(last)]
            last = lm.conv_layer(last, 3, 3, 64, 'decoder/hidden/conv0', act=default_act, **do_bn)
            last = [tf.image.resize_images(val, IMAGE_SIZE // 2, IMAGE_SIZE // 2) for val in listify(last)]
            last = lm.conv_layer(last, 3, 3, 32, 'decoder/hidden/conv1', act=default_act, **do_bn)
            last = lm.conv_layer(last, 3, 3, 32, 'decoder/hidden/conv2', act=default_act, **do_bn)
            last = lm.conv_layer(last, 3, 3, 32, 'decoder/hidden/conv3', act=default_act, **do_bn)
            last = [tf.image.resize_images(val, IMAGE_SIZE, IMAGE_SIZE) for val in listify(last)]
            last = lm.conv_layer(last, 3, 3, 8, 'decoder/hidden/conv4', act=default_act, **do_bn)
            last = lm.conv_layer(last, 3, 3, 8, 'decoder/hidden/conv5', act=default_act, **do_bn)
            output_mean_logit = lm.conv_layer(last, 3, 3, 1, 'output/mean', act=id_act, **do_bn)
            output_log_std = lm.conv_layer(last, 3, 3, 1, 'output/log_std', act=log_std_act, **do_bn)
            output_mean_logit = tf.reshape(output_mean_logit, [-1, IMAGE_SIZE*IMAGE_SIZE])
            output_log_std = tf.reshape(output_log_std, [-1, IMAGE_SIZE*IMAGE_SIZE])
        else:
            output_mean_logit = lm.nn_layer(last, IMAGE_SIZE*IMAGE_SIZE, 'output/mean', act=id_act, **do_bn)
            output_log_std = lm.nn_layer(last, IMAGE_SIZE*IMAGE_SIZE, 'output/log_std', act=log_std_act, **do_bn)
        return output_mean_logit, output_log_std

    def full_model(data):
        latent_mean, latent_log_std = encoder(data)
        if BNAE:
            latent = latent_mean
        else:
            latent = lm.reparam_normal_sample(latent_mean, latent_log_std, 'latent/sample')
        #latent = clip_rows_by_norm(latent, numpy.sqrt(LATENT_DIM + 4.0*numpy.sqrt(2.0*LATENT_DIM)))
        output_mean_logit, output_log_std = decoder(latent)
        output_mean = tf.nn.sigmoid(output_mean_logit)

        with tf.name_scope('likelihood_bound'):
            minus_kl = 0.5 * tf.reduce_sum(
                1.0 + 2.0 * latent_log_std - tf.square(latent_mean) - tf.exp(2.0 * latent_log_std),
                reduction_indices=[1])
            # Normal
            # reconstruction_error = tf.reduce_sum(
            #     -0.5 * numpy.log(2 * numpy.pi) - output_log_std - 0.5 * tf.square(output_mean - data) / tf.exp(
            #         2.0 * output_log_std), reduction_indices=[1])

            # Laplace
            # reconstruction_error = tf.reduce_sum(-numpy.log(2.0) - output_log_std - abs(output_mean-data)/tf.exp(output_log_std), reduction_indices=[1])

            # Cross Entropy
            reconstruction_error = -tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(output_mean_logit, data), reduction_indices=[1])

        with tf.name_scope('total'):
            likelihood_bound = tf.reduce_mean(minus_kl + reconstruction_error)
            lm.summaries.scalar_summary('likelihood bound', tf.nn.relu(
                likelihood_bound))  # Easier to parse graphs if giant negative values of first few iterations are omitted
            # likelihood_bound = tf.reduce_mean(tf.clip_by_value(tf.cast(batch, tf.float32)/10000.0 - 2.0, 0.0, 1.0)*minus_kl + reconstruction_error)

        with tf.name_scope('error'):
            squared_error = tf.reduce_mean(tf.square(data - output_mean))
            lm.summaries.scalar_summary('squared_error', squared_error)

        with tf.name_scope('independence_error'):
            num_normal_constraints = LATENT_DIM**2  # Who knows what this should be
            unit = tf.nn.l2_normalize(tf.random_normal((LATENT_DIM, num_normal_constraints)), 0)
            z = tf.matmul(latent, unit)  # random orthogonal projection of latent
            center = tf.truncated_normal([num_normal_constraints])
            width = 0.4  # Who knows what this should be
            g = tf.nn.tanh((z - center) / (width/2))*tf.exp(tf.square(center)/2)  # any family of univariate functions of z
            gprime, = tf.gradients(g, z)

            # zero for all g iff z is unit normal by Stein's Lemma
            stein_lemma_err = tf.reduce_mean(z * g - gprime, reduction_indices=[0], keep_dims=True)

            #ind_err = tf.squeeze(tf.matmul(tf.nn.softmax(0.1*abs(stein_lemma_err)), tf.square(stein_lemma_err), transpose_b=True))

            ind_err = tf.reduce_mean(tf.square(stein_lemma_err))

            # nonlin = tf.nn.relu(tf.sign(tf.random_normal((LATENT_DIM,)))*latent - tf.random_normal((LATENT_DIM,)))
            # nonlin_mean = tf.reduce_mean(nonlin, reduction_indices=[0], keep_dims=True)
            # nonlin_cov = tf.matmul(nonlin, nonlin, transpose_a=True)/tf.cast(tf.shape(latent)[0], tf.float32) - tf.matmul(nonlin_mean, nonlin_mean, transpose_a=True)
            # ind_err = tf.reduce_sum(tf.square(nonlin_cov)) - tf.reduce_sum(tf.diag_part(tf.square(nonlin_cov)))
            lm.summaries.scalar_summary('ind_err', ind_err)


        lm.summaries.image_summary('posterior/mean', tf.reshape(output_mean, [-1, IMAGE_SIZE, IMAGE_SIZE, 1]), 10)

        cov = tf.matmul(latent, latent, transpose_a=True)/tf.cast(tf.shape(latent)[0], tf.float32)
        eye = tf.diag(tf.ones((LATENT_DIM,)))
        lm.summaries.image_summary('cov', tf.expand_dims(tf.expand_dims(cov, 0), -1), 1)
        lm.summaries.image_summary('cov_error', tf.expand_dims(tf.expand_dims(cov-eye, 0), -1), 1)
        if BNAE:
            error = squared_error + ind_err
        else:
            error = -likelihood_bound  # + 1000.0*ind_err
        return output_mean, output_log_std, error

    def prior_model():
        latent = tf.random_normal((PRIOR_BATCH_SIZE, LATENT_DIM))
        output_mean_logit, output_log_std = decoder(latent)
        output_mean = tf.nn.sigmoid(output_mean_logit)

        sample_image = lm.summaries.image_summary('prior/mean', tf.reshape(output_mean, [-1, IMAGE_SIZE, IMAGE_SIZE, 1]), 10)
        return output_mean, output_log_std, sample_image

    with tf.name_scope('posterior'):
        reconstruction, _, error = full_model(training_batch)
    training_merged = lm.summaries.merge_all_summaries()
    lm.is_training = False
    tf.get_variable_scope().reuse_variables()
    with tf.name_scope('prior'):
        prior_sample, _, prior_sample_image = prior_model()
    lm.summaries.reset()
    with tf.name_scope('test'):
        test_reconstruction, _, test_error = full_model(fed_input_data)
    test_merged = lm.summaries.merge_all_summaries()

    saver = tf.train.Saver(tf.trainable_variables() + tf.get_collection('BatchNormInternal'))

    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, batch, 5000, 0.8, staircase=True)

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(error, global_step=batch, var_list=lm.weight_factory.variables + lm.bias_factory.variables + lm.scale_factory.variables)

    def feed_dict(mode):
        """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
        if mode == 'test':
            return {fed_input_data: mnist.test.images}
        else:
            return {}

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        sess.run(all_train_data.initializer, feed_dict={all_train_data_initializer: mnist.train.images})
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
                        summary, err = sess.run([test_merged, test_error], feed_dict=feed_dict('test'))
                        test_writer.add_summary(summary, i)
                        log('batch %s: Test error = %s' % (i, err))
                    if i % 100 == 99: # Record a summary
                        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata = tf.RunMetadata()
                        summary, prior_sample_summary, _ = sess.run([training_merged, prior_sample_image, train_step],
                                              feed_dict=feed_dict('train'),
                                              options=run_options,
                                              run_metadata=run_metadata)
                        train_writer.add_summary(summary, i)
                        train_writer.add_summary(prior_sample_summary, i)
                        train_writer.add_run_metadata(run_metadata, 'batch%d' % i)
                    else:
                        sess.run([train_step], feed_dict=feed_dict('train'))
            finally:
                log('saving')
                saver.save(sess, FLAGS.train_dir, global_step=batch)
                log('done')
        else:
            # Visualization code, nothing to do with training

            dated_files = [(os.path.getmtime('/tmp/mnist/' + fn), os.path.basename(fn)) for fn in os.listdir('/tmp/mnist') if fn.startswith('save') and os.path.splitext(fn)[1] == '']
            dated_files.sort()
            dated_files.reverse()
            newest = dated_files[0][1]
            log('restoring %s updated at %s' % (dated_files[0][1], time.ctime(dated_files[0][0])))
            saver.restore(sess, '/tmp/mnist/' + newest)
            latent, latent_log_std = encoder(fed_input_data)
            #latent = lm.reparam_normal_sample(latent_mean, latent_log_std, 'latent/sample')
            latents, = sess.run([latent], feed_dict={fed_input_data: mnist.train.images})
            import sklearn.decomposition

            # f = sklearn.decomposition.FastICA()
            # latents = f.fit_transform(latents)

            xlow = numpy.percentile(latents, 0.1)
            xhigh = numpy.percentile(latents, 99.9)
            xhigh = max(xhigh, -xlow)
            xlow = -xhigh
            x = numpy.linspace(xlow, xhigh, 1000)
            ind = [0]
            fig = plt.figure()
            ax = fig.add_subplot(111)
            num_bins = 100
            def latent_hist(_):
                plt.cla()
                data = latents[:, ind[0]]
                ax.hist(data, num_bins, range=(xlow, xhigh))
                norm = scipy.stats.norm(loc=data.mean(), scale=data.std()).pdf(x)*latents.shape[0]*(xhigh-xlow)/num_bins

                plt.plot(x, norm)
                plt.title('Distribution of latent ICA component ' + str(ind[0]))
                plt.xlim(xlow, xhigh)
                plt.ylim(0, 2*norm.max())
                plt.draw()
                ind[0] += 1
                if ind[0] >= LATENT_DIM:
                    ind[0] = 0
            latent_hist(None)
            cid = fig.canvas.mpl_connect('button_press_event', latent_hist)
            plt.show()
            fig.canvas.mpl_disconnect(cid)

        coord.request_stop()
        coord.join(threads)
        sess.close()

def main(_):
    if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    tf.gfile.MakeDirs(FLAGS.summaries_dir)
    train()

if __name__ == '__main__':
    tf.app.run()
