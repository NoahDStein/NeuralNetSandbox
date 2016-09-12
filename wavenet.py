from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy
import tensorflow as tf

import time

from tfutil import LayerManager, restore_latest, modified_dynamic_shape

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('max_steps', 100000, 'Number of steps to run trainer.')
flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate.')
flags.DEFINE_string('data_dir', '/tmp/data', 'Directory for storing data')
flags.DEFINE_string('summaries_dir', 'data/wavenet/logs', 'Summaries directory')
flags.DEFINE_string('train_dir', 'data/wavenet/save', 'Saves directory')

TRAIN_SIZE = 60000
TEST_SIZE = 10000
SIG_LEN = 256
NUM_COMPONENTS = 3
BATCH_SIZE = 64
PRIOR_BATCH_SIZE = 5
RESTORE_BEFORE_TRAIN = False
TRAIN = True

NUM_HIDDEN_LAYERS = 5
RES_LAYERS = 8
HIDDEN_LAYER_SIZE = 32

QUANT_LEVELS = 256
QUANT_LOWER = -10.0
QUANT_UPPER = 10.0

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

def quantizer(val, lower, upper, levels):
    normalized = (tf.clip_by_value(val, lower, upper) - lower)/(upper - lower + 1e-6)
    return tf.cast(tf.floor(normalized*levels), tf.int64)

def delay(tensor, steps):
    if steps == 0:
        return tensor
    static_shape = tensor.get_shape()
    zeros = tf.zeros(modified_dynamic_shape(tensor, [None, abs(steps), None]), dtype=tensor.dtype)
    if steps > 0:
        shifted_tensor = tensor[:, :static_shape.as_list()[1]-steps, :]
        delayed_tensor = tf.concat(1, (zeros, shifted_tensor))
    else:
        shifted_tensor = tensor[:, -steps:, :]
        delayed_tensor = tf.concat(1, (shifted_tensor, zeros))
    delayed_tensor.set_shape(static_shape)
    return delayed_tensor


def log_std_act(log_std):
    return tf.clip_by_value(log_std, -4.0, 4.0)

def id_act(z):
    return z

def double_relu(z):
    return [tf.nn.relu(z), tf.nn.relu(-z)]

default_act = tf.nn.relu  # double_relu
do_bn = dict(bn=True)

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

    lm = LayerManager(auto_summaries=False)

    with tf.name_scope('input'):
        all_train_data_initializer = tf.placeholder(tf.float32, [TRAIN_SIZE, SIG_LEN])
        all_train_data = tf.Variable(all_train_data_initializer, trainable=False, collections=[])
        random_training_example = tf.train.slice_input_producer([all_train_data])
        training_batch = tf.train.batch([random_training_example], batch_size=BATCH_SIZE, enqueue_many=True)

        all_test_data_initializer = tf.placeholder(tf.float32, [TEST_SIZE, SIG_LEN])
        all_test_data = tf.Variable(all_test_data_initializer, trainable=False, collections=[])
        test_batch = tf.train.batch([all_test_data], batch_size=BATCH_SIZE, enqueue_many=True)
        num_runs = tf.Variable(0.0, trainable=False, collections=[])
        running_error = tf.Variable(0.0, trainable=False, collections=[])
        fed_input_data = tf.placeholder(tf.float32, [None, SIG_LEN])


    def sub_predictor(last):
        all_res = []
        # Causal convolution -- may be better implemented as a convolution

        FILTER_SIZE = 16
        zeros = tf.zeros(modified_dynamic_shape(last, [None, FILTER_SIZE-1, None]), dtype=last.dtype)
        last = tf.expand_dims(tf.concat(1, (zeros, last)), 1)
        last = lm.conv_layer(last, 1, FILTER_SIZE, HIDDEN_LAYER_SIZE, 'predictor/causalconv', act=id_act, padding='VALID', **do_bn)
        last = tf.reshape(last, [-1, SIG_LEN, HIDDEN_LAYER_SIZE])
        #last = lm.nn_layer([delay(last, i) for i in range(16)], HIDDEN_LAYER_SIZE, 'predictor/causalconv', act=id_act, **do_bn)
        res = last
        all_res.append(res)
        for res_layers in range(RES_LAYERS):
            tanh_input = last
            sigmoid_input = last
            # Dilated causal convolution
            for i in range(NUM_HIDDEN_LAYERS-1):
                tanh_input = lm.nn_layer([tanh_input, delay(tanh_input, 2**i)], HIDDEN_LAYER_SIZE, 'predictor/res{}/hiddenT{}'.format(res_layers, i), act=id_act, **do_bn)
                sigmoid_input = lm.nn_layer([sigmoid_input, delay(sigmoid_input, 2 ** i)], HIDDEN_LAYER_SIZE, 'predictor/res{}/hiddenS{}'.format(res_layers, i), act=id_act, **do_bn)
            last = tf.nn.tanh(tanh_input)*tf.nn.sigmoid(sigmoid_input)
            last = lm.nn_layer(last, HIDDEN_LAYER_SIZE, 'predictor/res{}/hidden'.format(res_layers), act=id_act, **do_bn)
            res, last = last, last + res
            all_res.append(res)
        last = lm.nn_layer(all_res, HIDDEN_LAYER_SIZE, 'output/hidden', act=tf.nn.relu, **do_bn)
        last = lm.nn_layer(last, QUANT_LEVELS, 'output/logits', act=id_act, **do_bn)
        return last


    def predictor(data):
        last = tf.expand_dims(data, 2)
        ones = tf.ones_like(last, dtype=last.dtype)
        last = tf.concat(2, (last, ones))
        return sub_predictor(last)

    def full_model(data):
        output_logits = predictor(data)
        output_logits = output_logits[:, :SIG_LEN-1, :]
        output_mean = tf.argmax(output_logits, dimension=2)

        targets = data[:, 1:]
        quantized_targets = quantizer(targets, QUANT_LOWER, QUANT_UPPER, QUANT_LEVELS)
        with tf.name_scope('error'):
            batch_error = tf.reduce_mean(tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(output_logits, quantized_targets), reduction_indices=[1]))

            lm.summaries.scalar_summary('error', (running_error + batch_error)/(num_runs + 1.0))
        num_copies = 85
        image = tf.reshape(
            tf.tile(tf.expand_dims(tf.transpose(tf.pack([tf.cast(data, tf.float32) for data in [quantized_targets - QUANT_LEVELS//2, output_mean - QUANT_LEVELS//2, quantized_targets - output_mean]]), perm=[1, 0, 2]), 2),
                    [1, 1, num_copies, 1]), [-1, 3 * num_copies, SIG_LEN-1])
        lm.summaries.image_summary('posterior_sample', tf.expand_dims(image, -1), 5)
        return output_mean, batch_error

    def prior_model(init):
        def fn(acc, _):
            next_logit = sub_predictor(acc)
            # The logit multiplier is a hack which reduces randomness by artificially inflating the logits
            gumbeled = 1.0*next_logit[:, SIG_LEN-1, :] - tf.log(-tf.log(tf.random_uniform((tf.shape(acc)[0], QUANT_LEVELS))))
            sample_disc = tf.arg_max(gumbeled, 1)
            sample_cont = QUANT_LOWER + (QUANT_UPPER - QUANT_LOWER)*tf.cast(sample_disc, tf.float32)/tf.cast(QUANT_LEVELS-1, tf.float32)
            sample_cont = tf.expand_dims(sample_cont, 1)
            sample_cont = tf.expand_dims(sample_cont, 1) # sic
            sample_cont = tf.concat(2, (sample_cont, tf.ones_like(sample_cont)))
            return tf.concat(1, (acc[:, 1:, :], sample_cont))

        return tf.foldl(fn, numpy.arange(SIG_LEN), initializer=init.astype(numpy.float32), back_prop=False, swap_memory=True)[:, :, 0]

    def prior_model_with_summary():
        init = numpy.zeros((PRIOR_BATCH_SIZE, SIG_LEN, 2), dtype=numpy.float32)
        output_mean = prior_model(init)

        num_copies = 255
        image = tf.tile(tf.expand_dims(output_mean, 1), [1, num_copies, 1])
        sample_image = lm.summaries.image_summary('prior_sample', tf.expand_dims(image, -1), PRIOR_BATCH_SIZE)
        return output_mean, sample_image

    with tf.name_scope('posterior'):
        posterior_mean, training_error = full_model(training_batch)
    training_merged = lm.summaries.merge_all_summaries()
    tf.get_variable_scope().reuse_variables()
    with tf.name_scope('prior'):
        prior_sample, prior_sample_summary = prior_model_with_summary()
    lm.summaries.reset()
    with tf.name_scope('test'):
        _, test_error = full_model(test_batch)
        accum_test_error = [num_runs.assign(num_runs+1.0), running_error.assign(running_error+test_error)]
    test_merged = lm.summaries.merge_all_summaries()

    saver = tf.train.Saver(tf.trainable_variables())

    batch = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, batch, 5000, 0.8, staircase=True)

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(training_error, global_step=batch, var_list=lm.filter_factory.variables + lm.weight_factory.variables + lm.bias_factory.variables)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        sess.run(tf.initialize_variables(tf.get_collection('BatchNormInternal')))
        sess.run(all_train_data.initializer, feed_dict={all_train_data_initializer: train_data})
        sess.run(all_test_data.initializer, feed_dict={all_test_data_initializer: test_data})
        sess.run([num_runs.initializer, running_error.initializer])

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        if TRAIN:
            train_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/train', sess.graph)
            test_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/test')
            if RESTORE_BEFORE_TRAIN:
                log('restoring')
                restore_latest(saver, sess, 'data/wavenet')
            try:
                log('starting training')
                for i in range(FLAGS.max_steps):
                    if i % 1000 == 999:
                        # Track training error
                        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata = tf.RunMetadata()
                        summary, _ = sess.run([training_merged, train_step],
                                              options=run_options,
                                              run_metadata=run_metadata)
                        train_writer.add_summary(summary, i)
                        train_writer.add_run_metadata(run_metadata, 'batch%d' % i)

                        # Plot prior samples
                        prior_sample_summary_val, = sess.run([prior_sample_summary])
                        train_writer.add_summary(prior_sample_summary_val, i)

                        # Track test error
                        for _ in range(TEST_SIZE//BATCH_SIZE - 1):
                            sess.run(accum_test_error)
                        summary, _, _ = sess.run([test_merged] + accum_test_error)
                        acc, = sess.run([running_error/num_runs])
                        sess.run([num_runs.initializer, running_error.initializer])
                        test_writer.add_summary(summary, i)
                        log('batch %s: Test error = %s' % (i, acc))
                    else:
                        sess.run([train_step])
            finally:
                log('saving')
                saver.save(sess, FLAGS.train_dir, global_step=batch)
                log('done')
        else:
            log('restoring')
            restore_latest(saver, sess, 'data/wavenet')
            import matplotlib.pyplot as plt
            plt.ioff()
            fig = plt.figure()
            ax = fig.add_subplot(111)
            init = numpy.zeros((1, SIG_LEN, 2), dtype=numpy.float32)
            init[0, :, 0] = train_data[0, :]
            init[0, :, 1] = 1.0
            prior = prior_model(init)
            def plot_prior(_):
                prior_samples, = sess.run([prior])
                plt.cla()
                ax.plot(prior_samples[0, :])
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