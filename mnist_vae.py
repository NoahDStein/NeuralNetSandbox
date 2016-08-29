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

import matplotlib.pyplot as plt
import matplotlib, seaborn
import numpy
import scipy.stats
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os, sys

from tfutil import LayerManager, listify, log, restore_latest
from mnist_basic import classifier

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('max_steps', 100000, 'Number of steps to run trainer.')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_string('data_dir', '/tmp/data', 'Directory for storing data')
flags.DEFINE_string('summaries_dir', '/tmp/mnist_vae/logs', 'Summaries directory')
flags.DEFINE_string('train_dir', '/tmp/mnist_vae/save', 'Saves directory')
flags.DEFINE_string('viz_dir', '/tmp/mnist_vae/viz', 'Viz directory')

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
IMAGE_SIZE = 28
IMAGE_AREA = IMAGE_SIZE*IMAGE_SIZE
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 64
PRIOR_BATCH_SIZE = 10
NUM_RUNS_FOR_ENTROPY_ESTIMATES = 100

TRAIN = True
BNAE = False
CONV = False

IND_ERROR = False  # encourage normality of Q(z|X) across entire training set
# LATENT_DIM = 20
LATENT_DIM = 2
NUM_HIDDEN_LAYERS = 2

HIDDEN_LAYER_SIZE = 500

if CONV:
    small_image_size = IMAGE_SIZE // 4
    small_image_area = small_image_size * small_image_size
    HIDDEN_LAYER_SIZE = (HIDDEN_LAYER_SIZE // small_image_area) * small_image_area

def stat_summary(a):
    a = numpy.array(a)
    return [numpy.mean(a <= 0.0), numpy.mean(a <= 0.25), numpy.mean(a <= 0.5), numpy.mean(a <= 1.0), numpy.mean(a<=2.0)]
    #return [numpy.min(a), numpy.percentile(a, 25), numpy.percentile(a, 50), numpy.percentile(a, 75), numpy.max(a)]




def train():
    print("\nSource code of training file {}:\n\n{}".format(__file__, open(__file__).read()))

    log('loading MNIST')
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=False)
    TRAIN_SIZE=mnist.train.images.shape[0]

    lm = LayerManager(forward_biased_estimate=False)
    lm_classifier = LayerManager(forward_biased_estimate=False, is_training=False)
    batch = tf.Variable(0)

    prior_batch_size = tf.placeholder(tf.int64, [])

    with tf.name_scope('input'):
        all_train_data_initializer = tf.placeholder(tf.float32, [TRAIN_SIZE, IMAGE_AREA])
        all_train_data = tf.Variable(all_train_data_initializer, trainable=False, collections=[])
        random_training_example = tf.train.slice_input_producer([all_train_data])
        training_batch = tf.train.batch([random_training_example], batch_size=BATCH_SIZE, enqueue_many=True)
        fed_input_data = tf.placeholder(tf.float32, [None, IMAGE_AREA])

    def id_act(z):
        return z

    def log_std_act(z):
        return tf.clip_by_value(z, -5.0, 5.0)

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
        for i in range(NUM_HIDDEN_LAYERS):
            last = lm.nn_layer(last, HIDDEN_LAYER_SIZE, 'encoder/hidden/fc{}'.format(i), act=default_act, **do_bn)
        if BNAE:
            latent_mean = lm.nn_layer(last, LATENT_DIM, 'latent/mean', act=id_act, **do_bn) #, bias=False, scale=False)
        else:
            latent_mean = lm.nn_layer(last, LATENT_DIM, 'latent/mean', act=id_act, **do_bn)
        latent_log_std = lm.nn_layer(last, LATENT_DIM, 'latent/log_std', act=log_std_act, **do_bn)
        return latent_mean, latent_log_std

    def decoder(code):
        last = code
        for i in range(NUM_HIDDEN_LAYERS):
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
            lm.summaries.scalar_summary('likelihood bound', likelihood_bound)  # Easier to parse graphs if giant negative values of first few iterations are omitted
            # likelihood_bound = tf.reduce_mean(tf.clip_by_value(tf.cast(batch, tf.float32)/10000.0 - 2.0, 0.0, 1.0)*minus_kl + reconstruction_error)

        with tf.name_scope('error'):
            squared_error = tf.reduce_mean(tf.square(data - output_mean))
            lm.summaries.scalar_summary('squared_error', squared_error)

        with tf.name_scope('independence_error'):
            num_normal_constraints = 20*LATENT_DIM  # Who knows what this should be
            unit = tf.nn.l2_normalize(tf.random_normal((LATENT_DIM, num_normal_constraints)), 0)
            z = tf.matmul(latent, unit)  # random orthogonal projection of latent
            center = tf.truncated_normal([num_normal_constraints])
            width = 0.4  # Who knows what this should be
            g = tf.nn.tanh((z - center) / (width/2))*tf.exp(tf.square(center)/2)  # any family of univariate functions of z
            gprime, = tf.gradients(g, z)

            # zero for all g iff z is unit normal by Stein's Lemma
            stein_lemma_err = tf.reduce_mean(z * g - gprime, reduction_indices=[0], keep_dims=True)

            #ind_err = tf.squeeze(tf.matmul(tf.nn.softmax(0.1*abs(stein_lemma_err)), tf.square(stein_lemma_err), transpose_b=True))

            ind_err = tf.sqrt(tf.cast(tf.shape(latent)[0], tf.float32)) * tf.reduce_mean(tf.square(stein_lemma_err))

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
        weight_decay = sum([tf.reduce_sum(t**2) for t in lm.weight_factory.variables + lm.bias_factory.variables])
        if BNAE:
            error = squared_error
        else:
            error = -likelihood_bound
        if IND_ERROR:
            error += ind_err
        return output_mean, output_log_std, error

    def prior_model(latent=None):  # option to call with latent as numpy array of shape 1xLATENT_DIM
        if latent is None:
            latent = tf.random_normal((prior_batch_size, LATENT_DIM))
        else:
            latent = tf.convert_to_tensor(latent, dtype=tf.float32)
        output_mean_logit, output_log_std = decoder(latent)
        output_mean = tf.nn.sigmoid(output_mean_logit)

        sample_image = lm.summaries.image_summary('prior/mean', tf.reshape(output_mean, [-1, IMAGE_SIZE, IMAGE_SIZE, 1]), 10)
        return output_mean, output_log_std, sample_image


    classifier_logits = classifier(lm_classifier, fed_input_data)

    classifier_saver = tf.train.Saver([var for var in tf.trainable_variables() + tf.get_collection('BatchNormInternal') if var != batch])

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
    test_merged = lm.summaries.merge_all_summaries() + lm_classifier.summaries.merge_all_summaries()

    saver = tf.train.Saver(tf.trainable_variables() + tf.get_collection('BatchNormInternal'))

    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, batch, 5000, 0.8, staircase=True)

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(error, global_step=batch, var_list=lm.weight_factory.variables + lm.bias_factory.variables + lm.scale_factory.variables)

    def feed_dict(mode):
        """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
        if mode == 'test':
            return {fed_input_data: mnist.test.images, prior_batch_size: 1000}
        else:
            return {prior_batch_size: PRIOR_BATCH_SIZE}

    with tf.Session() as sess:
        train_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/train', sess.graph)
        test_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/test')

        sess.run(tf.initialize_all_variables())
        sess.run(all_train_data.initializer, feed_dict={all_train_data_initializer: mnist.train.images})
        sess.run(tf.initialize_variables(tf.get_collection('BatchNormInternal')))

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        restore_latest(classifier_saver, sess, '/tmp/mnist_basic')

        runs = []
        for _ in range(NUM_RUNS_FOR_ENTROPY_ESTIMATES):
            new_output_probs, = sess.run([classifier_logits], feed_dict={fed_input_data: mnist.test.images[:1000, :]})
            new_output = numpy.argmax(new_output_probs, 1)
            runs.append(new_output)

        all_runs = numpy.vstack(runs).T
        entropy_summary = stat_summary([scipy.stats.entropy(numpy.bincount(row), base=2.0) for row in all_runs])
        log('Summary of prediction entropy on 1000 test data samples = {}'.format(entropy_summary))


        if TRAIN:

            try:
                log('starting training')
                for i in range(FLAGS.max_steps):
                    if i % 1000 == 999: # Do test set
                        summary, err = sess.run([test_merged, test_error], feed_dict=feed_dict('test'))
                        test_writer.add_summary(summary, i)

                        log('batch %s: Test error = %s' % (i, err))
                    if i % 5000 == 4999:
                        prior_sample_data, = sess.run([prior_sample], feed_dict=feed_dict('test'))

                        runs = []
                        for _ in range(NUM_RUNS_FOR_ENTROPY_ESTIMATES):
                            new_output_probs, = sess.run([classifier_logits], feed_dict={fed_input_data: prior_sample_data})
                            new_output = numpy.argmax(new_output_probs, 1)
                            runs.append(new_output)

                        all_runs = numpy.vstack(runs).T
                        entropy_summary = stat_summary([scipy.stats.entropy(numpy.bincount(row), base=2.0) for row in all_runs])

                        log('batch {}: summary of prediction entropy on prior samples = {}'.format(i, entropy_summary))

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
            if not os.path.exists(FLAGS.viz_dir):
                os.makedirs(FLAGS.viz_dir)
            restore_latest(saver, sess, '/tmp/mnist_vae')

            def decode_uniform_samples_from_latent_space(_):
                fig, ax = plt.subplots()
                nx = ny = 20
                extent_x = extent_y = [-3, 3]
                extent = numpy.array(extent_x + extent_y)
                x_values = numpy.linspace(*(extent_x + [nx]))
                y_values = numpy.linspace(*(extent_y + [nx]))
                full_extent = extent * (nx + 1) / float(nx)
                canvas = numpy.empty((28 * ny, 28 * nx))
                for ii, yi in enumerate(x_values):
                    for j, xi in enumerate(y_values):
                        n = ii * nx + j + 1
                        sys.stdout.write("\rsampling p(X|z), sample %d/%d" % (n, nx*ny))
                        sys.stdout.flush()
                        np_z = numpy.array([[xi, yi]])
                        x_mean = sess.run(prior_model(latent=numpy.reshape(np_z, newshape=(1, LATENT_DIM)))[0])
                        canvas[(nx - ii - 1) * 28:(nx - ii) * 28, j * 28:(j + 1) * 28] = x_mean[0].reshape(28, 28)
                with seaborn.axes_style('ticks'):
                    seaborn.set_context(context='notebook', font_scale=1.75)
                    fig, ax = plt.subplots(figsize=(12, 9))
                ax.imshow(canvas, extent=full_extent)
                ax.xaxis.set_ticks(numpy.linspace(*(extent_x + [nx])))
                ax.yaxis.set_ticks(numpy.linspace(*(extent_y + [ny])))
                ax.set_xlabel('z_1')
                ax.set_ylabel('z_2')
                ax.set_title('P(X|z); decoding latent space; (CONV, BNAE, IND_ERROR) = (%d,%d,%d)' % (CONV, BNAE, IND_ERROR))
                plt.show()
                plt.savefig(os.path.join(FLAGS.viz_dir, 'P(X|z).png'))
                return fig, ax

            def latent_2d_scatter(latents, labels):  # if latent space is 2d we can visualize encoded data directly
                fig, ax = plt.subplots(figsize=(12, 9))
                cmap = matplotlib.colors.ListedColormap(seaborn.color_palette("hls", 10))
                order = numpy.random.permutation(range(latents.shape[0]))  # avoid, e.g., all 9's on scatter top layer
                im = ax.scatter(latents[order, 0], latents[order, 1], c=labels[order], cmap=cmap)
                ax.set_xlabel('z_1')
                ax.set_ylabel('z_2')
                ax.set_title('Q(z|X_train); (CONV, BNAE, IND_ERROR) = (%d,%d,%d)' % (CONV, BNAE, IND_ERROR))
                lims = [-5, 5]
                ax.set_xlim(*lims)
                ax.set_ylim(*lims)
                f.colorbar(im, ax=ax, label='Digit class')
                plt.show()
                plt.savefig('%s/Q(z|X).png' % FLAGS.viz_dir)
                return fig, ax
            if LATENT_DIM == 2:
                f, a = decode_uniform_samples_from_latent_space(None)
                plt.close(f)
            latent, latent_log_std = encoder(fed_input_data)
            latents, = sess.run([latent], feed_dict={fed_input_data: mnist.train.images})
            labels = mnist.train.labels
            if LATENT_DIM == 2:
                f, a = latent_2d_scatter(latents=latents, labels=labels)
                plt.close(f)
            #latent = lm.reparam_normal_sample(latent_mean, latent_log_std, 'latent/sample')
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
            plt.savefig('%s/latent_hist_%i.png' % (FLAGS.viz_dir, ind[0]))
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
