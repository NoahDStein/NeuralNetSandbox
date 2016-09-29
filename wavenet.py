from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy
import tensorflow as tf
import tensorflow.contrib.slim as slim

import time

from tfutil import LayerManager, restore_latest, modified_dynamic_shape, quantizer, dequantizer, crappy_plot, draw_on, \
    queue_append_and_update, modified_static_shape

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('max_steps', 100000, 'Number of steps to run trainer.')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
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

HIDDEN_LAYER_SIZE = 32
DELAYS = [1, 2, 4, 8, 16] * 4

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
    train_data = rand_periodic(NUM_COMPONENTS, TRAIN_SIZE, SIG_LEN)
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

    def sub_predictor(input_val, queue_contents=None):
        queue_updates = []

        def next_queue(model_tensor, depth):
            if queue_contents is None:
                new_shape = [None] * model_tensor.get_shape().ndims
                new_shape[1] = depth
                this_queue_contents = tf.zeros(shape=modified_static_shape(model_tensor, new_shape))
            else:
                this_queue_contents = queue_contents[len(queue_updates)]
            concatenated_contents, updated_contents = queue_append_and_update(1, this_queue_contents, model_tensor)
            queue_updates.append(updated_contents)
            return concatenated_contents

        all_res = []
        last = input_val
        # Causal convolution
        FILTER_SIZE = 16
        bn_params = dict(decay=0.95, scope='bn', updates_collections=None)
        with slim.arg_scope([slim.conv2d, slim.fully_connected], normalizer_fn=slim.batch_norm, normalizer_params=bn_params, num_outputs=HIDDEN_LAYER_SIZE):
            last = next_queue(last, FILTER_SIZE-1)
            last = tf.expand_dims(last, 1)
            last = slim.conv2d(last, kernel_size=(1, FILTER_SIZE), padding='VALID', activation_fn=None, scope='predictor/causalconv')
            last = tf.reshape(last, modified_static_shape(input_val, [None, None, HIDDEN_LAYER_SIZE]))
            res = last
            all_res.append(res)
            for res_layer, cur_delay in enumerate(DELAYS):
                total = next_queue(last, cur_delay)
                last = tf.concat(2, (total[:, cur_delay:, :], total[:, :-cur_delay, :]))
                # Dilated causal convolution
                tanh = slim.fully_connected(last, activation_fn=tf.nn.tanh, scope='predictor/res{}T'.format(res_layer))
                sigmoid = slim.fully_connected(last, activation_fn=tf.nn.sigmoid, scope='predictor/res{}S'.format(res_layer))
                last = slim.fully_connected(tanh*sigmoid, activation_fn=None, scope='predictor/res{}/hidden'.format(res_layer))
                res, last = last, last + res
                all_res.append(res)


            # last = tf.concat(3, [tf.expand_dims(r, 3) for r in all_res])
            # num_layers = len(all_res)

            # Need to keep these convolutions as not running over time or else add queues
            # last = lm.conv_transpose_layer(last, 1, 5, num_layers//2, 'output/conv0', act=tf.nn.relu, strides=[1, 1, 2, 1], padding='SAME', bias_dim=2, **do_bn)
            # last = lm.conv_transpose_layer(last, 1, 5, num_layers//4, 'output/conv1', act=tf.nn.relu, strides=[1, 1, 2, 1], padding='SAME', bias_dim=2, **do_bn)
            # last = lm.conv_transpose_layer(last, 1, 5, num_layers//8, 'output/conv2', act=tf.nn.relu, strides=[1, 1, 2, 1], padding='SAME', bias_dim=2, **do_bn)
            # last = lm.conv_layer(last, 1, 5, 1, 'output/conv3', act=id_act, padding='SAME', bias_dim=2, **do_bn)
            # last = last[:, :, :, 0]


            last = slim.fully_connected(tf.concat(2, all_res), activation_fn=tf.nn.relu, scope='output/hidden')
            last = slim.fully_connected(last, num_outputs=QUANT_LEVELS, activation_fn=None, normalizer_params=dict(bn_params, scale=True), scope='output/logits')
        return last, queue_updates


    def predictor(data):
        last = tf.expand_dims(data, 2)
        ones = tf.ones_like(last, dtype=last.dtype)
        noise = tf.random_normal(tf.shape(last))
        last = tf.concat(2, (last + 0.1*noise, ones))
        return sub_predictor(last)

    def full_model(data):
        output_logits, queue_updates = predictor(data)
        output_logits = output_logits[:, :SIG_LEN-1, :]
        output_mean = tf.argmax(output_logits, dimension=2)

        targets = data[:, 1:]
        quantized_targets = quantizer(targets, QUANT_LOWER, QUANT_UPPER, QUANT_LEVELS)
        with tf.name_scope('error'):
            batch_error = tf.reduce_mean(tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(output_logits, quantized_targets), reduction_indices=[1]))

            lm.summaries.scalar_summary('training error', (running_error + batch_error)/(num_runs + 1.0))
        output_plot = crappy_plot(output_mean, QUANT_LEVELS)
        target_plot = crappy_plot(quantized_targets, QUANT_LEVELS)

        M = tf.reduce_max(output_logits)
        m = tf.reduce_min(output_logits)
        scaled_logits = (output_logits-m)/(M-m)
        # image = draw_on(tf.transpose(scaled_logits, perm=[0, 2, 1])[:, :, :, None], target_plot, [1.0, 0.0, 0.0])
        # Casting is to work around some stupid tf bug; shouldn't be necessary
        output_probs = tf.reshape(tf.cast(tf.nn.softmax(tf.reshape(tf.cast(output_logits, tf.float64), [-1, QUANT_LEVELS])), tf.float32), [-1, SIG_LEN-1, QUANT_LEVELS])
        image = draw_on(tf.transpose(output_probs, perm=[0, 2, 1])[:, :, :, None], target_plot, [1.0, 0.0, 0.0])


        # image = draw_on(1.0, target_plot, [1.0, 0.0, 0.0])    # The first 1.0 starts with a white canvas
        # image = draw_on(image, output_plot, [0.0, 0.0, 1.0])

        lm.summaries.image_summary('posterior_sample', image, 5)
        return output_mean, queue_updates, batch_error, batch_error #+ 0.1*weight_decay

    def prior_model(prior_queue_init, length=SIG_LEN):
        def cond(loop_counter, *_):
            return tf.less(loop_counter, length)

        def body(loop_counter, accumulated_output, accumulated_logits, next_input, *queue_contents):
            next_logit, queue_updates = sub_predictor(next_input, queue_contents)
            gumbeled = next_logit[:, 0, :] - tf.log(-tf.log(tf.random_uniform((tf.shape(accumulated_output)[0], QUANT_LEVELS))))
            sample_disc = tf.arg_max(gumbeled, 1)
            sample_cont = dequantizer(sample_disc, QUANT_LOWER, QUANT_UPPER, QUANT_LEVELS)
            sample_cont = tf.expand_dims(sample_cont, 1)
            accumulated_output = tf.concat(1, (accumulated_output, sample_cont))
            accumulated_logits = tf.concat(1, (accumulated_logits, next_logit))
            sample_cont = tf.expand_dims(sample_cont, 1) # sic
            next_input = tf.concat(2, (sample_cont, tf.ones_like(sample_cont)))
            return [loop_counter+1, accumulated_output, accumulated_logits, next_input] + queue_updates

        loop_var_init = [tf.constant(0, dtype=tf.int32), tf.zeros((PRIOR_BATCH_SIZE, 0)), tf.zeros((PRIOR_BATCH_SIZE, 0, QUANT_LEVELS)), tf.zeros((PRIOR_BATCH_SIZE, 1, 2))] + prior_queue_init
        output, logits = tf.while_loop(cond, body, loop_var_init, back_prop=False)[1:3]
        output.set_shape((PRIOR_BATCH_SIZE, length))
        logits.set_shape((PRIOR_BATCH_SIZE, length, QUANT_LEVELS))
        return output, logits

    def prior_model_with_summary(queue_model):
        prior_queue_init = []
        for tensor in queue_model:
            new_shape = tensor.get_shape().as_list()
            new_shape[0] = PRIOR_BATCH_SIZE
            prior_queue_init.append(tf.zeros(new_shape, dtype=tf.float32))
        output_sample, output_logits = prior_model(prior_queue_init)

        M = tf.reduce_max(output_logits)
        m = tf.reduce_min(output_logits)
        scaled_logits = (output_logits-m)/(M-m)
        # Casting is to work around some stupid tf bug; shouldn't be necessary
        output_probs = tf.reshape(tf.cast(tf.nn.softmax(tf.reshape(tf.cast(output_logits, tf.float64), [-1, QUANT_LEVELS])), tf.float32), [-1, SIG_LEN, QUANT_LEVELS])
        image = draw_on(tf.transpose(output_probs, perm=[0, 2, 1])[:, :, :, None], crappy_plot(quantizer(output_sample, QUANT_LOWER, QUANT_UPPER, QUANT_LEVELS), QUANT_LEVELS), [0.0, 0.0, 1.0])

        sample_image = lm.summaries.image_summary('prior_sample', image, PRIOR_BATCH_SIZE)
        return output_sample, sample_image

    with tf.name_scope('posterior'):
        posterior_mean, queue_updates, _, training_error = full_model(training_batch)
    training_merged = lm.summaries.merge_all_summaries()
    with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=False):
        tf.get_variable_scope().reuse_variables()
        with tf.name_scope('prior'):
            prior_sample, prior_sample_summary = prior_model_with_summary(queue_updates)
        lm.summaries.reset()
        with tf.name_scope('test'):
            _, _, test_error, _ = full_model(test_batch)
            accum_test_error = [num_runs.assign(num_runs+1.0), running_error.assign(running_error+test_error)]
    test_merged = lm.summaries.merge_all_summaries()

    saver = tf.train.Saver(tf.trainable_variables() + tf.get_collection('BatchNormInternal'))

    batch = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, batch, 5000, 0.8, staircase=True)

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(training_error, global_step=batch)

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
            logit, = sess.run([predictor(fed_input_data)[0]], feed_dict={fed_input_data: train_data[10:20, :]})

            def softmax(x, axis=None):
                x = x - x.max(axis=axis, keepdims=True)
                x = numpy.exp(x)
                return x/numpy.sum(x, axis=axis, keepdims=True)

            import IPython
            IPython.embed()

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
