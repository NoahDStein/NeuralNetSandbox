# Adversarial autoencoder

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
flags.DEFINE_float('learning_rate', 0.0003, 'Initial learning rate.')
flags.DEFINE_string('data_dir', '/tmp/data', 'Directory for storing data')
flags.DEFINE_string('summaries_dir', '/tmp/aae/logs', 'Summaries directory')
flags.DEFINE_string('train_dir', '/tmp/aae/save', 'Saves directory')

TRAIN_SIZE = 60000
TEST_SIZE = 10000
SIG_LEN = 256
NUM_COMPONENTS = 3
BATCH_SIZE = 64
PRIOR_BATCH_SIZE = 5
PRETRAIN = False
TRAIN = True

LATENT_DIM = 20
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

    lm_ae = LayerManager()
    lm_disc = LayerManager()

    with tf.name_scope('input'):
        all_train_data_initializer = tf.placeholder(tf.float32, [TRAIN_SIZE, SIG_LEN])
        all_train_data = tf.Variable(all_train_data_initializer, trainable=False, collections=[])
        random_training_example = tf.train.slice_input_producer([all_train_data])
        training_batch = tf.train.batch([random_training_example], batch_size=BATCH_SIZE, enqueue_many=True)
        fed_input_data = tf.placeholder(tf.float32, [None, SIG_LEN])

    def log_std_act(log_std):
        return tf.clip_by_value(log_std, -4.0, 4.0)

    def id_act(z):
        return z

    def double_relu(z):
        return [tf.nn.relu(z), tf.nn.relu(-z)]

    def encoder(data):
        last = data
        for i in range(NUM_HIDDEN_LAYERS):
            last = lm_ae.nn_layer(last, HIDDEN_LAYER_SIZE, 'encoder/hidden{}'.format(i), act=double_relu)
        with tf.variable_scope('latent'):
            latent_mean = lm_ae.nn_layer(last, LATENT_DIM, 'mean', act=id_act)
            latent_log_std = lm_ae.nn_layer(last, LATENT_DIM, 'log_std', act=log_std_act)
        return latent_mean, latent_log_std

    def decoder(code):
        last = code
        for i in range(NUM_HIDDEN_LAYERS):
            last = lm_ae.nn_layer(last, HIDDEN_LAYER_SIZE, 'decoder/hidden{}'.format(i), act=double_relu)
        output_mean = lm_ae.nn_layer(last, SIG_LEN, 'output/mean', act=id_act)
        output_log_std = lm_ae.nn_layer(last, SIG_LEN, 'output/log_std', act=log_std_act)
        return output_mean, output_log_std

    def discriminator(latent):
        last = latent
        for i in range(1):
            last = lm_disc.nn_layer(last, 20, 'discriminator/hidden{}'.format(i), act=double_relu)
        output_logit = lm_disc.nn_layer(last, 1, 'discrimator/prediction', act=id_act)
        return output_logit

    def full_model(data):
        latent_mean, latent_log_std = encoder(data)
        #latent_sample = lm_ae.reparam_normal_sample(latent_mean, latent_log_std, 'sample')
        latent_sample = latent_mean
        output_mean, output_log_std = decoder(latent_sample)
        disc_neg_logit = discriminator(latent_sample)
        tf.get_variable_scope().reuse_variables()
        latent_prior_sample = tf.random_normal(tf.shape(latent_mean))
        latent_prior_sample.set_shape(latent_mean.get_shape().as_list())
        disc_pos_logit = discriminator(latent_prior_sample)

        reconstruction_error = tf.reduce_sum(
            -0.5 * numpy.log(2 * numpy.pi) - output_log_std - 0.5 * tf.square(output_mean - data) / tf.exp(
                2.0 * output_log_std), reduction_indices=[1])

        disc_cross_entropy =   0.5*tf.nn.sigmoid_cross_entropy_with_logits(disc_neg_logit, tf.zeros(tf.shape(disc_neg_logit))) \
                            + 0.5*tf.nn.sigmoid_cross_entropy_with_logits(disc_pos_logit, tf.ones( tf.shape(disc_pos_logit)))

        num_copies = 85
        image = tf.reshape(
            tf.tile(tf.expand_dims(tf.transpose(tf.pack([data, output_mean, data - output_mean]), perm=[1, 0, 2]), 2),
                    [1, 1, num_copies, 1]), [-1, 3 * num_copies, SIG_LEN])
        lm_ae.summaries.image_summary('posterior_sample', tf.expand_dims(image, -1), 5)
        rough_error = tf.reduce_mean(tf.square(tf.reduce_mean(tf.square(output_mean), reduction_indices=[1]) - tf.reduce_mean(tf.square(data), reduction_indices=[1])))
        return output_mean, tf.reduce_mean(reconstruction_error), tf.reduce_mean(disc_cross_entropy), rough_error

    def prior_model():
        latent_sample = tf.random_normal((PRIOR_BATCH_SIZE, LATENT_DIM))
        output_mean, output_log_std = decoder(latent_sample)

        num_copies = 255
        image = tf.tile(tf.expand_dims(output_mean, 1), [1, num_copies, 1])
        sample_image = lm_ae.summaries.image_summary('prior_sample', tf.expand_dims(image, -1), 5)
        return output_mean, sample_image

    with tf.name_scope('posterior'):
        posterior_mean, reconstruction_error, disc_cross_entropy, rough_error = full_model(training_batch)
    training_merged = lm_ae.summaries.merge_all_summaries()
    tf.get_variable_scope().reuse_variables()
    with tf.name_scope('prior'):
        prior_mean, prior_sample = prior_model()
    lm_ae.summaries.reset()
    with tf.name_scope('test'):
        test_posterior_mean, test_reconstruction_error, test_dist_cross_entropy, _ = full_model(fed_input_data)
    test_merged = lm_ae.summaries.merge_all_summaries()

    saver = tf.train.Saver(tf.trainable_variables())

    batch = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, batch, 5000, 0.8, staircase=True)

    pretrain_step = tf.train.AdamOptimizer(0.03).minimize(rough_error, var_list=lm_ae.scale_factory.variables)
    ae_train_step = tf.train.AdamOptimizer(learning_rate).minimize(-reconstruction_error, global_step=batch, var_list=lm_ae.weight_factory.variables + lm_ae.bias_factory.variables)
    ae_fool_disc_train_step = tf.train.AdamOptimizer(learning_rate/3.0).minimize(-disc_cross_entropy, global_step=batch, var_list=lm_ae.weight_factory.variables + lm_ae.bias_factory.variables)

    disc_train_step = tf.train.AdamOptimizer(learning_rate/3.0).minimize(disc_cross_entropy, global_step=batch, var_list=lm_disc.weight_factory.variables + lm_disc.bias_factory.variables)

    def feed_dict(mode):
        """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
        if mode == 'test':
            return {fed_input_data: test_data}
        else:
            return {}

    def validate(i, write_summary=True):
        summary, ce, re = sess.run([test_merged, test_dist_cross_entropy, test_reconstruction_error],
                                   feed_dict=feed_dict('test'))
        log('batch %s: Test set cross entropy = %s, test set reconstruction error = %s' % (i, ce, re))
        if write_summary:
            test_writer.add_summary(summary, i)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        sess.run(all_train_data.initializer, feed_dict={all_train_data_initializer: train_data})

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        if TRAIN:
            train_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/train', sess.graph)
            test_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/test')

            try:
                if PRETRAIN:
                    log('starting pre-training')
                    for i in range(3000):
                        err, _ = sess.run([rough_error, pretrain_step], feed_dict=feed_dict('train'))
                        if i % 100 == 99:
                            log('batch %s: single training batch rough error = %s' % (i, err))
                #validate(0, write_summary=False)
                log('initializing ae')
                for i in range(5000):
                    re, _ = sess.run([reconstruction_error, ae_train_step], feed_dict=feed_dict('train'))
                    if i % 1000 == 999:
                        log('batch %s: single training batch reconstruction error = %s' % (i, re))
                # #validate(0, write_summary=False)
                # log('initializing discriminator')
                # for i in range(5000):
                #     ce, _ = sess.run([disc_cross_entropy, disc_train_step], feed_dict('train'))
                #     if i % 1000 == 999:
                #         log('batch %s: single training batch cross entropy = %s' % (i, ce))
                #validate(0, write_summary=False)
                log('starting training')
                for i in range(FLAGS.max_steps):
                    if i % 1000 == 999: # Do test set
                        validate(i)
                    # if i % 10 == 0:
                    #     sess.run([disc_train_step], feed_dict('train'))
                    if i % 100 == 99: # Record a summary
                        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata = tf.RunMetadata()
                        summary, prior_sample_summary, _ = sess.run([training_merged, prior_sample, ae_train_step],
                                              feed_dict=feed_dict('train'),
                                              options=run_options,
                                              run_metadata=run_metadata)
                        train_writer.add_summary(summary, i)
                        train_writer.add_summary(prior_sample_summary, i)
                        train_writer.add_run_metadata(run_metadata, 'batch%d' % i)
                    else:
                        # sess.run([disc_train_step], feed_dict=feed_dict('train'))
                        # sess.run([ae_fool_disc_train_step], feed_dict=feed_dict('train'))
                        # sess.run([ae_train_step], feed_dict=feed_dict('train'))
                        sess.run([ae_train_step, ae_fool_disc_train_step, disc_train_step], feed_dict=feed_dict('train'))

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
                prior_means, = sess.run([prior_mean], feed_dict=feed_dict('prior'))
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
