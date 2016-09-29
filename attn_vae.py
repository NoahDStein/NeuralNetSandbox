from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy
import tensorflow as tf
import matplotlib.pyplot as plt
import os, sys
import time
import matplotlib, seaborn
from sklearn import manifold
import pickle
from tfutil_deprecated import LayerManager
nets_dir = './nets'
file_name = os.path.splitext(os.path.basename(__file__))[0]
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('max_steps', 100000, 'Number of steps to run trainer.')
flags.DEFINE_float('learning_rate', 0.0003, 'Initial learning rate.')
flags.DEFINE_string('data_dir', '%s/%s/data' % (nets_dir, file_name), 'Directory for storing data')
flags.DEFINE_string('summaries_dir', '%s/%s/logs' % (nets_dir, file_name), 'Summaries directory')
flags.DEFINE_string('train_dir', '%s/%s/save' % (nets_dir, file_name), 'Saves directory')
flags.DEFINE_string('viz_dir', '%s/%s/viz' % (nets_dir, file_name), 'Viz directory')
for d in [FLAGS.data_dir, FLAGS.summaries_dir, FLAGS.train_dir, FLAGS.viz_dir]:
    if not os.path.exists(d):
        os.makedirs(d)

TRAIN_SIZE = 60000
TEST_SIZE = 10000
SIG_LEN = 256
NUM_COMPONENTS = 3
BATCH_SIZE = 64
PRIOR_BATCH_SIZE = 5
PRETRAIN = False
TRAIN = False

LATENT_DIM = 10
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
        numpy.save('%s/train_data.npy' % FLAGS.data_dir, train_data)

    else: # Don't waste time computing training data
        train_data = numpy.load('%s/train_data.npy' % FLAGS.data_dir)
    log('done simulating')

    lm = LayerManager()

    prior_batch_size = tf.placeholder(tf.int64, [])

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
            last = lm.nn_layer(last, HIDDEN_LAYER_SIZE, 'encoder/hidden{}'.format(i), act=double_relu)
        with tf.variable_scope('latent'):
            latent_mean = lm.nn_layer(last, LATENT_DIM, 'mean', act=id_act)
            latent_log_std = lm.nn_layer(last, LATENT_DIM, 'log_std', act=log_std_act)
        return latent_mean, latent_log_std

    def decoder(code):
        last = code
        for i in range(NUM_HIDDEN_LAYERS):
            last = lm.nn_layer(last, HIDDEN_LAYER_SIZE, 'decoder/hidden{}'.format(i), act=double_relu)
        # norm_freq = lm.nn_layer(last, 1, 'decoder/norm_freq', act=tf.nn.sigmoid)
        # output_mean = tf.constant(0.0, dtype=tf.float32)
        # output_log_std = tf.constant(0.0, dtype=tf.float32)
        # for i in range(1, NUM_COMPONENTS+1):
        #     sin_weight = lm.nn_layer(last, 1, 'decoder/mean_sin_weight{}'.format(i))
        #     cos_weight = lm.nn_layer(last, 1, 'decoder/mean_cos_weight{}'.format(i))
        #     output_mean = output_mean + lm.parametrized_sinusoid(SIG_LEN, norm_freq*i, sin_weight, cos_weight)
        #     sin_weight = lm.nn_layer(last, 1, 'decoder/log_std_sin_weight{}'.format(i))
        #     cos_weight = lm.nn_layer(last, 1, 'decoder/log_std_cos_weight{}'.format(i))
        #     output_log_std = output_log_std + lm.parametrized_sinusoid(SIG_LEN, 2*norm_freq*i, sin_weight, cos_weight)
        # output_log_std = log_std_act(output_log_std)
        output_mean = lm.nn_layer(last, SIG_LEN, 'output/mean', act=id_act)
        output_log_std = lm.nn_layer(last, SIG_LEN, 'output/log_std', act=log_std_act)
        return output_mean, output_log_std

    def full_model(data):
        latent_mean, latent_log_std = encoder(data)
        latent_sample = lm.reparam_normal_sample(latent_mean, latent_log_std, 'sample')
        output_mean, output_log_std = decoder(latent_sample)

        with tf.name_scope('likelihood_bound'):
            minus_kl = 0.5 * tf.reduce_sum(1.0 + 2.0 * latent_log_std - tf.square(latent_mean) - tf.exp(2.0 * latent_log_std), reduction_indices=[1])
            # Normal
            reconstruction_error = tf.reduce_sum(-0.5 * numpy.log(2 * numpy.pi) - output_log_std - 0.5 * tf.square(output_mean - data) / tf.exp(2.0 * output_log_std), reduction_indices=[1])
            # Laplace
            # reconstruction_error = tf.reduce_sum(-numpy.log(2.0) - output_log_std - abs(output_mean-data)/tf.exp(output_log_std), reduction_indices=[1])

            likelihood_bound = tf.reduce_mean(minus_kl + reconstruction_error)
            lm.summaries.scalar_summary('truncated likelihood bound', tf.clip_by_value(likelihood_bound, -500, 1000))
        num_copies = 85
        image = tf.reshape(
            tf.tile(tf.expand_dims(tf.transpose(tf.pack([data, output_mean, data - output_mean]), perm=[1, 0, 2]), 2),
                    [1, 1, num_copies, 1]), [-1, 3 * num_copies, SIG_LEN])
        lm.summaries.image_summary('posterior_sample', tf.expand_dims(image, -1), 5)
        rough_error = tf.reduce_mean(tf.square(tf.reduce_mean(tf.square(output_mean), reduction_indices=[1]) - tf.reduce_mean(tf.square(data), reduction_indices=[1])))
        return output_mean, likelihood_bound, rough_error


    def prior_model(latent=None):  # option to call with latent as numpy array of shape 1xLATENT_DIM
        if latent is None:
            latent = tf.random_normal((prior_batch_size, LATENT_DIM))
        else:
            latent = tf.convert_to_tensor(latent, dtype=tf.float32)
        output_mean_logit, output_log_std = decoder(latent)
        output_mean = tf.nn.sigmoid(output_mean_logit)
        num_copies = 255
        image = tf.tile(tf.expand_dims(output_mean, 1), [1, num_copies, 1])
        sample_image = lm.summaries.image_summary('prior_sample', tf.expand_dims(image, -1), 5)
        return output_mean, sample_image  # for now, do nothing with std dev

    with tf.name_scope('posterior'):
        posterior_mean, likelihood_bound, rough_error = full_model(training_batch)
    training_merged = lm.summaries.merge_all_summaries()
    tf.get_variable_scope().reuse_variables()
    with tf.name_scope('prior'):
        prior_mean, prior_sample = prior_model()
    lm.summaries.reset()
    with tf.name_scope('test'):
        test_posterior_mean, test_likelihood_bound, _ = full_model(fed_input_data)
    test_merged = lm.summaries.merge_all_summaries()

    saver = tf.train.Saver(tf.trainable_variables())

    batch = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, batch, 5000, 0.8, staircase=True)

    if PRETRAIN:
        pretrain_step = tf.train.AdamOptimizer(0.03).minimize(rough_error, var_list=lm.scale_factory.variables)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(-likelihood_bound, global_step=batch, var_list=lm.weight_factory.variables + lm.bias_factory.variables)

    def feed_dict(mode):
        """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
        if mode == 'test':
            return {fed_input_data: test_data, prior_batch_size: PRIOR_BATCH_SIZE}
        else:
            return {prior_batch_size: PRIOR_BATCH_SIZE}

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
                            log('batch %s: Single training batch rough error = %s' % (i, err))
                log('starting training')
                for i in range(FLAGS.max_steps):
                    if i % 1000 == 999: # Do test set
                        summary, acc = sess.run([test_merged, test_likelihood_bound], feed_dict=feed_dict('test'))
                        test_writer.add_summary(summary, i)
                        log('batch %s: Test set likelihood bound = %s' % (i, acc))
                    if i % 100 == 99: # Record a summary
                        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata = tf.RunMetadata()
                        summary, prior_sample_summary, _ = sess.run([training_merged, prior_sample, train_step],
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
            # viz fns
            def plot_prior(n_priors=10):
                for i in xrange(n_priors):
                    prior_means, = sess.run([prior_mean], feed_dict=feed_dict('prior'))
                    plt.cla()
                    ax.plot(prior_means[0, :])
                    plt.draw()
                    cid = fig.canvas.mpl_connect('button_press_event', plot_prior)
                    plt.show()
                    plt.savefig('%s/plot_prior%d.png' % (FLAGS.viz_dir, i))
                    fig.canvas.mpl_disconnect(cid)
                return

            def decode_uniform_samples_from_latent_space(N=20, show_ticks=False):
                nx = ny = N
                with seaborn.axes_style('ticks'):
                    fig, ax = plt.subplots(nx, ny, figsize=(12, 12))
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
                        ax[ii, j].plot(x_mean.T)
                        ax[ii, j].set_xlim([0, SIG_LEN])
                        ax[ii, j].set_ylim([-1, 1])
                        ax[ii, j].set_title('(z1,z2)=(%d,%d)' % (ii, j))
                        if not show_ticks:
                            ax[ii, j].xaxis.set_ticklabels([])
                            ax[ii, j].yaxis.set_ticklabels([])
                plt.suptitle('P(X_mean|z); decoding latent space')
                plt.show()
                plt.savefig(os.path.join(FLAGS.viz_dir, 'P(X|z).png'))
                return fig, ax

            def latent_2d_scatter(latents, labels, tag=''):  # if latent space is 2d we can visualize encoded data directly
                fig, ax = plt.subplots(figsize=(12, 9))
                cmap = None  # todo: color data by some simple feature, e.g. max bin in fft?
                order = numpy.random.permutation(xrange(latents.shape[0]))
                im = ax.scatter(latents[order, 0], latents[order, 1])  # c=labels[order], cmap=cmap
                ax.set_xlabel('z_1')
                ax.set_ylabel('z_2')
                ax.set_title('Q(z|X%s)' % tag)
                lims = [-5, 5]
                ax.set_xlim(*lims)
                ax.set_ylim(*lims)
                # f.colorbar(im, ax=ax)
                plt.show()
                plt.savefig('%s/Q(z|X%s).png' % (FLAGS.viz_dir, tag))
                return fig, ax

            def plot_embedding(X, title=None):
                # embeddings are initialization and data dependent, so it's not obvious two can be meaningfully compared
                x_min, x_max = numpy.min(X, 0), numpy.max(X, 0)
                X = (X - x_min) / (x_max - x_min)
                plt.figure()
                ax = plt.subplot(111)
                ax.scatter(X[:, 0], X[:, 1])
                # plt.xticks([]), plt.yticks([])
                ax.set_xlabel('tnse axis 1')
                ax.set_ylabel('tnse axis 2')
                if title is not None:
                    plt.title(title)

            def subsample_data(data, N):  # randomly sample N elements of data
                return data[numpy.random.permutation(numpy.arange(data.shape[0]))[:N], :]

            # viz
            log('restoring')
            saver.restore(sess, FLAGS.train_dir + '-' + str(FLAGS.max_steps))
            if not os.path.exists(FLAGS.viz_dir):
                os.makedirs(FLAGS.viz_dir)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            plot_prior()
            latent, latent_log_std = encoder(fed_input_data)
            latents_test, = sess.run([latent], feed_dict=feed_dict('test'))
            latents_train = numpy.array(sess.run([latent], feed_dict={fed_input_data: train_data})).squeeze()
            if LATENT_DIM == 2:
                f, a = decode_uniform_samples_from_latent_space(10, show_ticks=True)
                plt.close(f)
                for t, l in zip(['train', 'test'], [latents_train, latents_test]):
                    latent_2d_scatter(latents=l, labels=None, tag='_%s' % t)
            else:
                latents_train_subsample = subsample_data(latents_train, latents_test.shape[0])  # tsne doesn't do well with N > 10000
                for t, l in zip(['test', 'train_subsample'], [latents_test, latents_train_subsample]):
                    if os.path.exists('%s/tsne_%s.npz' % (FLAGS.train_dir, t)):
                        z = numpy.load(open('%s/tsne_%s.pkl' % (FLAGS.train_dir, t), 'rb'))
                        X_tsne = z['embedding']
                        time_elapsed = z['embedding']
                    else:
                        tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
                        t0 = time.time()
                        X_tsne = tsne.fit_transform(l)
                        time_elapsed = t0 - time.time()
                        pickle.dump({'embedding': X_tsne, 'time_elapsed': time_elapsed}, open('%s/tsne.pkl' % FLAGS.train_dir, 'wb'))
                    plot_embedding(X_tsne,
                                   "t-SNE embedding of the %s data (time %.2fs)" %
                                   (t, time_elapsed))
                    plt.savefig('%s/tsne_%s.png' % (FLAGS.viz_dir, t))
                    plt.show()
            print('viz at %s' % FLAGS.viz_dir)
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
