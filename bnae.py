from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy
import tensorflow as tf
import matplotlib.pyplot as plt

import time

from tfutil_deprecated import LayerManager

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('max_steps', 100000, 'Number of steps to run trainer.')
flags.DEFINE_float('learning_rate', 0.003, 'Initial learning rate.')
flags.DEFINE_string('data_dir', '/tmp/data', 'Directory for storing data')
flags.DEFINE_string('summaries_dir', '/tmp/bnae/logs', 'Summaries directory')
flags.DEFINE_string('train_dir', '/tmp/bnae/save', 'Saves directory')

TRAIN_SIZE = 60000
TEST_SIZE = 10000
SIG_LEN = 256
NUM_COMPONENTS = 3
BATCH_SIZE = 64
PRIOR_BATCH_SIZE = 5
TRAIN = True

LATENT_DIM = 8 # Very sensitive to this value: changing to 20 completely breaks periodicity of prior samples
NUM_HIDDEN_LAYERS = 3
HIDDEN_LAYER_SIZE = 200

def log(s):
    print('[%s] ' % time.asctime() + s)

def rand_periodic(num_components, num_signals, signal_length):
    time = numpy.arange(signal_length, dtype=numpy.float32).reshape(1, signal_length)
    period = numpy.random.rand(num_signals, 1) * 80 + 40
    counter = 2*numpy.pi*time / period
    sin_coeff = numpy.random.randn(num_components, num_signals)
    cos_coeff = numpy.random.randn(num_components, num_signals)
    arg = numpy.arange(1, num_components + 1).reshape(num_components, 1, 1) * counter
    return numpy.einsum('ij,ijk->jk', sin_coeff, numpy.sin(arg)) + numpy.einsum('ij,ijk->jk', cos_coeff, numpy.cos(arg))

def train():
    # Import data
    log('simulating data')

    numpy.random.seed(3737)
    test_data = rand_periodic(NUM_COMPONENTS, TEST_SIZE, SIG_LEN)
    if TRAIN:
        train_data = rand_periodic(NUM_COMPONENTS, TRAIN_SIZE, SIG_LEN)
    else: # Don't waste time computing training data
        train_data = numpy.zeros((TRAIN_SIZE, SIG_LEN))
    log('done simulating')

    lm = LayerManager(forward_biased_estimate=True)

    with tf.name_scope('input'):
        all_train_data_initializer = tf.placeholder(tf.float32, [TRAIN_SIZE, SIG_LEN])
        all_train_data = tf.Variable(all_train_data_initializer, trainable=False, collections=[])
        random_training_example = tf.train.slice_input_producer([all_train_data])
        training_batch = tf.train.batch([random_training_example], batch_size=BATCH_SIZE, enqueue_many=True)
        fed_input_data = tf.placeholder(tf.float32, [None, SIG_LEN])

    def id_act(z):
        return z

    def double_relu(z):
        return [tf.nn.relu(z), tf.nn.relu(-z)]

    def encoder(data):
        last = data
        for i in range(NUM_HIDDEN_LAYERS):
            last = lm.nn_layer(last, HIDDEN_LAYER_SIZE, 'encoder/hidden{}'.format(i), act=double_relu, scale=False, bn=True)
        latent = lm.nn_layer(last, LATENT_DIM, 'latent', act=id_act, bias=False, scale=False, bn=True)
        return latent

    def decoder(code):
        last = code
        for i in range(NUM_HIDDEN_LAYERS):
            last = lm.nn_layer(last, HIDDEN_LAYER_SIZE, 'decoder/hidden{}'.format(i), act=double_relu, scale=False, bn=True)
        # norm_freq = lm.nn_layer(last, 1, 'decoder/norm_freq', act=tf.nn.sigmoid)
        # output = tf.constant(0.0, dtype=tf.float32)
        # output_log_std = tf.constant(0.0, dtype=tf.float32)
        # for i in range(1, NUM_COMPONENTS+1):
        #     sin_weight = lm.nn_layer(last, 1, 'decoder/mean_sin_weight{}'.format(i))
        #     cos_weight = lm.nn_layer(last, 1, 'decoder/mean_cos_weight{}'.format(i))
        #     output = output + lm.parametrized_sinusoid(SIG_LEN, norm_freq*i, sin_weight, cos_weight)
        #     sin_weight = lm.nn_layer(last, 1, 'decoder/log_std_sin_weight{}'.format(i))
        #     cos_weight = lm.nn_layer(last, 1, 'decoder/log_std_cos_weight{}'.format(i))
        #     output_log_std = output_log_std + lm.parametrized_sinusoid(SIG_LEN, 2*norm_freq*i, sin_weight, cos_weight)
        # output_log_std = log_std_act(output_log_std)
        output = lm.nn_layer(last, SIG_LEN, 'output', act=id_act, scale=True, bn=True)
        return output

    def full_model(data):
        latent = encoder(data)
        output = decoder(latent)

        with tf.name_scope('error'):
            error = tf.reduce_mean(tf.square(data - output))
            lm.summaries.scalar_summary('error', error)
        num_copies = 85
        image = tf.reshape(
            tf.tile(tf.expand_dims(tf.transpose(tf.pack([data, output, data - output]), perm=[1, 0, 2]), 2),
                    [1, 1, num_copies, 1]), [-1, 3 * num_copies, SIG_LEN])
        lm.summaries.image_summary('posterior_sample', tf.expand_dims(image, -1), 5)
        cov = tf.matmul(latent, latent, transpose_a=True)/tf.cast(tf.shape(latent)[0], tf.float32)
        eye = tf.diag(tf.ones((LATENT_DIM,)))
        lm.summaries.image_summary('cov', tf.expand_dims(tf.expand_dims(cov, 0), -1), 1)
        lm.summaries.image_summary('cov_error', tf.expand_dims(tf.expand_dims(cov-eye, 0), -1), 1)
        return output, error

    def prior_model():
        latent = tf.random_normal((PRIOR_BATCH_SIZE, LATENT_DIM))
        output = decoder(latent)

        num_copies = 255
        image = tf.tile(tf.expand_dims(output, 1), [1, num_copies, 1])
        sample_image = lm.summaries.image_summary('prior_sample', tf.expand_dims(image, -1), 5)
        return output, sample_image

    with tf.name_scope('posterior'):
        reconstruction, error = full_model(training_batch)
    training_merged = lm.summaries.merge_all_summaries()
    lm.is_training = False
    tf.get_variable_scope().reuse_variables()
    with tf.name_scope('prior'):
        prior_sample, prior_sample_image = prior_model()
    lm.summaries.reset()
    with tf.name_scope('test'):
        test_reconstruction, test_error = full_model(fed_input_data)
    test_merged = lm.summaries.merge_all_summaries()

    saver = tf.train.Saver(tf.trainable_variables() + tf.get_collection('BatchNormInternal'))

    batch = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, batch, 5000, 0.8, staircase=True)

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(error, global_step=batch, var_list=lm.weight_factory.variables + lm.bias_factory.variables + lm.scale_factory.variables)

    def feed_dict(mode):
        """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
        if mode == 'test':
            return {fed_input_data: test_data}
        else:
            return {}

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        sess.run(all_train_data.initializer, feed_dict={all_train_data_initializer: train_data})
        sess.run(tf.initialize_variables(tf.get_collection('BatchNormInternal')))

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        if TRAIN:
            train_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/train', sess.graph)
            test_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/test')
            try:
                log('starting training')
                for i in range(FLAGS.max_steps):
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
            log('restoring')
            saver.restore(sess, FLAGS.train_dir + '-' + str(FLAGS.max_steps))
            fig = plt.figure()
            ax = fig.add_subplot(111)
            def plot_prior(_):
                prior_means, = sess.run([prior_sample], feed_dict=feed_dict('prior'))
                plt.cla()
                ax.plot(prior_means[0, :])
                plt.draw()
            plot_prior(None)
            cid = fig.canvas.mpl_connect('button_press_event', plot_prior)
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
