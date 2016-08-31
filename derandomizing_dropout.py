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

import tensorflow.models.image.cifar10.cifar10 as cifar10

from tensorflow.python.framework import ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

from tfutil import LayerManager, log, restore_latest, multi_class_hinge_loss

flags = tf.app.flags
FLAGS = flags.FLAGS

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
IMAGE_SIZE = cifar10.IMAGE_SIZE
IMAGE_AREA = IMAGE_SIZE*IMAGE_SIZE
NUM_CLASSES = 10
BATCH_SIZE = FLAGS.batch_size
BATCHES_PER_DERANDOMIZE_STEP = 40
BINOMIAL_TEST_CUTOFF = 3
CHI2_TEST_CUTOFF = 25
SIGNIFICANCE_LEVEL = 0.5

TRAIN = True
DERANDOMIZE_DROPOUT = True
DEFAULT_KEEP_PROB = 0.5

DROPOUT_LAYER_SIZES = [96, 192]
NUM_DROPOUT_LAYERS = len(DROPOUT_LAYER_SIZES)

def id_act(z):
    return z

def double_relu(z):
    return [tf.nn.relu(z), tf.nn.relu(-z)]



# Modified version of dropout from TF codebase (Apache license)
# Removing error checking on keep_prob so it can be a tensor broadcastable to the shape of x, just like noise_shape
# pylint: disable=invalid-name
def dropout(x, keep_prob, noise_shape=None, seed=None, name=None):
    with ops.op_scope([x], name, "dropout") as name:
        x = ops.convert_to_tensor(x, name="x")

        noise_shape = noise_shape if noise_shape is not None else array_ops.shape(x)
        # uniform [keep_prob, 1.0 + keep_prob)
        random_tensor = keep_prob
        random_tensor += random_ops.random_uniform(noise_shape,
                                                   seed=seed,
                                                   dtype=x.dtype)
        # 0. if [keep_prob, 1.0) and 1. if [1.0, 1.0 + keep_prob)
        binary_tensor = math_ops.floor(random_tensor)
        ret = x * math_ops.inv(tf.reduce_mean(keep_prob, reduction_indices=[1], keep_dims=True)) * binary_tensor
        ret.set_shape(x.get_shape())
        return ret

# Modified version of dropout from TF codebase (Apache license)
# Removing error checking on keep_prob so it can be a tensor broadcastable to the shape of x, just like noise_shape
# pylint: disable=invalid-name
def double_relu_dropout(x, keep_prob, noise_shape=None, seed=None, name=None):
    with ops.op_scope(x, name, "dropout") as name:
        noise_shape = noise_shape if noise_shape is not None else array_ops.shape(x[0])
        # uniform [keep_prob, 1.0 + keep_prob)
        random_tensor = keep_prob
        random_tensor += random_ops.random_uniform(noise_shape,
                                                   seed=seed,
                                                   dtype=x[0].dtype)
        # 0. if [keep_prob, 1.0) and 1. if [1.0, 1.0 + keep_prob)
        binary_tensor = math_ops.floor(random_tensor)
        ret = [x[0] * binary_tensor, x[1] * (1-binary_tensor)]
        ret[0].set_shape(x[0].get_shape())
        ret[1].set_shape(x[1].get_shape())
        return ret


default_act = tf.nn.relu  # double_relu
do_bn = dict(bn=True)

def classifier(lm, last, drop_probs):
    logits = []
    last = lm.conv_layer(last, 5, 5, 192, 'classifier/hidden/conv0', act=id_act, **do_bn)

    logits.append(lm.nn_layer(tf.reshape(last, [-1, IMAGE_AREA*192]), 10, 'classifier/output/logits0', act=id_act, **do_bn))

    last = tf.nn.relu(last)
    last = lm.conv_layer(last, 1, 1, 160, 'classifier/hidden/conv0mlp0', act=tf.nn.relu, **do_bn)
    last = lm.conv_layer(last, 1, 1, 96, 'classifier/hidden/conv0mlp1', act=tf.nn.relu, **do_bn)
    last = lm.max_pool(last, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    last = dropout(last, drop_probs[0])

    last = lm.conv_layer(last, 5, 5, 192, 'classifier/hidden/conv1', act=id_act, **do_bn)

    logits.append(lm.nn_layer(tf.reshape(last, [-1, IMAGE_AREA*192//4]), 10, 'classifier/output/logits1', act=id_act, **do_bn))

    last = tf.nn.relu(last)
    last = lm.conv_layer(last, 1, 1, 192, 'classifier/hidden/conv1mlp0', act=tf.nn.relu, **do_bn)
    last = lm.conv_layer(last, 1, 1, 192, 'classifier/hidden/conv1mlp1', act=tf.nn.relu, **do_bn)
    last = lm.max_pool(last, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    last = dropout(last, drop_probs[1])

    last = lm.conv_layer(last, 5, 5, 192, 'classifier/hidden/conv2', act=id_act, **do_bn)

    logits.append(lm.nn_layer(tf.reshape(last, [-1, IMAGE_AREA*192//16]), 10, 'classifier/output/logits2', act=id_act, **do_bn))

    last = tf.nn.relu(last)
    last = lm.conv_layer(last, 1, 1, 192, 'classifier/hidden/conv2mlp0', act=tf.nn.relu, **do_bn)
    last = lm.conv_layer(last, 1, 1, 10, 'classifier/hidden/conv2mlp1', act=id_act, **do_bn)
    # Pool of size 6 rather than 8 as in DSN paper because tensorflow gives 24x24 images rather than 32x32
    last = lm.avg_pool(last, ksize=[1, 6, 6, 1], strides=[1, 1, 1, 1], padding='VALID')

    logits.append(tf.squeeze(last, name='classifier/output/logits3'))
    return logits

def full_model(lm, drop_probs, data, labels):
    output_logits = classifier(lm, data, drop_probs)

    with tf.name_scope('error'):
        per_example_error = [multi_class_hinge_loss(out, labels, power=2) for out in output_logits]
        batch_error = [tf.reduce_mean(ce) for ce in per_example_error]
        for i, ce in enumerate(batch_error):
            lm.summaries.scalar_summary('batch_error{}'.format(i), ce)
        weight_decay = sum([tf.reduce_sum(t ** 2) for t in lm.weight_factory.variables])
        lm.summaries.scalar_summary('weight_decay', weight_decay)
        total_error = sum(batch_error)+0.001*weight_decay
        lm.summaries.scalar_summary('total_error', total_error)
        incorrect_examples = tf.not_equal(tf.cast(tf.arg_max(output_logits[-1], dimension=1), tf.int32), labels)
        percent_error = 100.0 * tf.reduce_mean(
            tf.cast(incorrect_examples, tf.float32))
        lm.summaries.scalar_summary('percent_error', percent_error)

    # Perhaps weights should be different on different layers
    # Need to add alphas and gammas
    return total_error, percent_error, per_example_error, incorrect_examples

def train():
    print("\nSource code of training file {}:\n\n{}".format(__file__, open(__file__).read()))

    log('loading CIFAR')
    # Import data
    training_batch = cifar10.distorted_inputs()

    lm = LayerManager(forward_biased_estimate=False)
    batch = tf.Variable(0)

    with tf.name_scope('input'):
        fed_input_data = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, 3])
        fed_input_labels = tf.placeholder(tf.int32, [None])
        drop_probs = [tf.Variable(tf.constant(DEFAULT_KEEP_PROB, shape=[1, 1, 1, ], dtype=tf.float32), trainable=False, collections=['Dropout']) for _ in range(NUM_DROPOUT_LAYERS)]

    with tf.name_scope('posterior'):
        training_batch_error, _, _, _ = full_model(lm, drop_probs, *training_batch)
    training_merged = lm.summaries.merge_all_summaries()
    lm.is_training = False
    tf.get_variable_scope().reuse_variables()
    lm.summaries.reset()
    with tf.name_scope('test'):
        _, test_percent_error, _, _ = full_model(lm, drop_probs, *cifar10.inputs(eval_data=True))
    with tf.name_scope('forward'):
        _, _, forward_per_example_error, forward_incorrect_examples = full_model(lm, drop_probs, fed_input_data, fed_input_labels)

    def compute_test_percent_error():
        return numpy.mean([sess.run([test_percent_error]) for _ in range(int(numpy.ceil(FLAGS.num_test_examples / FLAGS.batch_size)))])

    saver = tf.train.Saver(tf.trainable_variables() + tf.get_collection('BatchNormInternal'))

    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, batch, 5000, 0.8, staircase=True)

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(training_batch_error, global_step=batch, var_list=lm.filter_factory.variables + lm.weight_factory.variables + lm.bias_factory.variables + lm.scale_factory.variables)

    fed_drop_probs = tf.placeholder(tf.float32, [None, None, None, None])
    update_drop_probs = [tf.assign(drop_prob, fed_drop_probs, validate_shape=False) for drop_prob in drop_probs]

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        sess.run(tf.initialize_variables(tf.get_collection('BatchNormInternal')))
        sess.run(tf.initialize_variables(tf.get_collection('Dropout')))

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        if TRAIN:
            train_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/train', sess.graph)
            # test_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/test')
            try:
                log('starting training')
                for i in range(FLAGS.max_steps):
                    if i % 1000 == 999: # Do test set
                        err = compute_test_percent_error()
                        for j in range(NUM_DROPOUT_LAYERS):
                            sess.run([update_drop_probs[j]], feed_dict={fed_drop_probs: [[[[1.0]]]]})
                        det_err = compute_test_percent_error()
                        for j in range(NUM_DROPOUT_LAYERS):
                            sess.run([update_drop_probs[j]], feed_dict={fed_drop_probs: [[[[DEFAULT_KEEP_PROB]]]]})
                        log('batch %s: Random test classification error = %s%%, deterministic test classification error = %s%%' % (i, err, det_err))
                    if i % 100 == 99: # Record a summary
                        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata = tf.RunMetadata()
                        summary, _ = sess.run([training_merged, train_step],
                                              options=run_options,
                                              run_metadata=run_metadata)
                        train_writer.add_summary(summary, i)
                        train_writer.add_run_metadata(run_metadata, 'batch%d' % i)
                    else:
                        sess.run([train_step])
            finally:
                log('saving')
                saver.save(sess, FLAGS.train_dir, global_step=batch)
                log('done')
        else:
            restore_latest(saver, sess, '/tmp/derandomizing_dropout', suffix='-100000')

        if DERANDOMIZE_DROPOUT:
            # NUM_RUNS = 10
            # runs = []
            # for _ in range(NUM_RUNS):
            #     new_output_probs, = sess.run([forward_output], feed_dict={fed_input_data: mnist.train.images, fed_input_labels: mnist.train.labels})
            #     new_output = numpy.argmax(new_output_probs, 1)
            #     runs.append(new_output)
            #
            # all_runs = numpy.vstack(runs).T
            # entropy = numpy.array([scipy.stats.entropy(numpy.bincount(row), base=2.0) for row in all_runs])


            derandomized_drop_probs = [DEFAULT_KEEP_PROB * numpy.ones((1, HIDDEN_LAYER_SIZE)) for _ in range(NUM_DROPOUT_LAYERS)]

            num_tests_performed = 0

            for pass_count in range(1):
                for j in range(HIDDEN_LAYER_SIZE):
                    for i in range(NUM_DROPOUT_LAYERS):  # range(NUM_DROPOUT_LAYERS-1,-1,-1):
                        if derandomized_drop_probs[i][0, j] == 0.0 or derandomized_drop_probs[i][0, j] == 1.0:
                            continue
                        num_tests_performed += 1
                        for k in range(NUM_DROPOUT_LAYERS):
                            if k == i:
                                # curr_drop_probs = numpy.tile(derandomized_drop_probs[i], (BATCHES_PER_DERANDOMIZE_STEP*BATCH_SIZE, 1))
                                # to_randomize = HIDDEN_LAYER_SIZE - j - 1
                                # randperms = numpy.argsort(numpy.random.rand(BATCHES_PER_DERANDOMIZE_STEP*BATCH_SIZE, to_randomize), axis=1)
                                #
                                # to_keep = max(int(HIDDEN_LAYER_SIZE*DEFAULT_KEEP_PROB-derandomized_drop_probs[i][:j].sum()), 1)
                                # curr_drop_probs[:, j+1:] = (randperms < to_keep)


                                curr_drop_probs = (numpy.random.rand(BATCHES_PER_DERANDOMIZE_STEP*BATCH_SIZE, HIDDEN_LAYER_SIZE) < derandomized_drop_probs[i]).astype(numpy.float32)
                                curr_drop_probs[:, j] = 0.0
                                # curr_drop_probs[:, j+1:j+2] = 1.0
                                sess.run([update_drop_probs[i]], feed_dict={fed_drop_probs: curr_drop_probs})
                            else:
                                sess.run([update_drop_probs[k]], feed_dict={fed_drop_probs: numpy.random.rand(BATCHES_PER_DERANDOMIZE_STEP * BATCH_SIZE, HIDDEN_LAYER_SIZE) < derandomized_drop_probs[k]})

                        #indices = numpy.argmax(entropy[:, numpy.newaxis] + -numpy.log(-numpy.log(numpy.random.rand(entropy.shape[0], BATCHES_PER_DERANDOMIZE_STEP*BATCH_SIZE))), axis=0)

                        #  indices = [numpy.argmax(1000*entropy + -numpy.log(-numpy.log(numpy.random.rand(*entropy.shape)))) for _ in range(BATCHES_PER_DERANDOMIZE_STEP*BATCH_SIZE)]
                        # examples = mnist.train.images[indices, :]
                        # labels = mnist.train.labels[indices]
                        # Collect a bunch of 64-example batches together
                        examples, labels = [numpy.concatenate(things, axis=0) for things in zip(*[sess.run(training_batch) for _ in range(BATCHES_PER_DERANDOMIZE_STEP)])]

                        # Might want to use cross entropy, but why not not use percent error since we're not differentiating?
                        # Using "test" expressions so we can manually feed in data, but we are feeding training data (same data for obj0 and obj1)
                        err0, cross_entropies0 = sess.run([forward_incorrect_examples, forward_per_example_error], feed_dict={fed_input_data: examples, fed_input_labels: labels})
                        curr_drop_probs[:, j] = 1.0
                        # curr_drop_probs[:, j+1:] = (randperms < to_keep - 1)
                        # curr_drop_probs[:, j+1:j+2] = 0.0
                        sess.run([update_drop_probs[i]], feed_dict={fed_drop_probs: curr_drop_probs})
                        err1, cross_entropies1 = sess.run([forward_incorrect_examples, forward_per_example_error], feed_dict={fed_input_data: examples, fed_input_labels: labels})

                        # One-sided paired-sample t-test
                        cross_entropy_diff = cross_entropies0 - cross_entropies1
                        t = numpy.sqrt(BATCHES_PER_DERANDOMIZE_STEP * BATCH_SIZE)*cross_entropy_diff.mean()/cross_entropy_diff.std(ddof=1)
                        p = scipy.stats.t.sf(-t, df=BATCHES_PER_DERANDOMIZE_STEP * BATCH_SIZE - 1)

                        b = numpy.sum(err0 & ~err1)
                        c = numpy.sum(err1 & ~err0)
                        # if b + c < BINOMIAL_TEST_CUTOFF:
                        #     p = 0.5
                        #     stat_message = "too small"
                        # else:
                        #     # McNemar's test
                        #     if b + c >= CHI2_TEST_CUTOFF:
                        #         chi2 = (b-c)**2/(b+c)
                        #         p = scipy.stats.distributions.chi2.sf(chi2, df=1)  # Two-sided
                        #     else:
                        #         p = scipy.stats.binom_test([b,c]) - scipy.stats.binom.pmf(b, b+c, 0.5)  # Mid-p test
                        #     # Form one-sided p-value
                        #     if b > c:
                        #         p = 1-0.5*p
                        #     else:
                        #         p = 0.5*p
                        #     if b + c >= CHI2_TEST_CUTOFF:
                        #         stat_message = "p = %.4f, chi square test" % p
                        #     else:
                        #         stat_message = "p = %.4f, binomial mid-p test" % p

                        if p < SIGNIFICANCE_LEVEL:  # cross_entropies0.mean() <= cross_entropies1.mean():  # b <= c:
                            new_drop_prob = 0.0
                            neuron_status = "drop"
                        elif p > 1 - SIGNIFICANCE_LEVEL:
                            new_drop_prob = 1.0
                            neuron_status = "keep"
                        else:
                            new_drop_prob = DEFAULT_KEEP_PROB
                            neuron_status = "hmmm"

                        #log(neuron_status + ' L{} N{}: b + c = {}, {}'.format(i, j, b+c, stat_message))
                        log(neuron_status + ' P{} L{} N{}: b = {}, c = {}, p = {}'.format(pass_count, i, j, b, c, p))
                        derandomized_drop_probs[i][0, j] = new_drop_prob
                for i in range(NUM_DROPOUT_LAYERS):
                    num_dropped = (derandomized_drop_probs[i] == 0.0).sum()
                    num_kept = (derandomized_drop_probs[i] == 1.0).sum()
                    num_hmmm = HIDDEN_LAYER_SIZE - num_dropped - num_kept
                    sess.run([update_drop_probs[i]], feed_dict={fed_drop_probs: numpy.ceil(derandomized_drop_probs[i])})

                    log('layer {}: {} neurons dropped, {} kept, {} undecided'.format(i, num_dropped, num_kept, num_hmmm))
                log('Performed {} statistical tests'.format(num_tests_performed))
            log('saving')
            saver.save(sess, FLAGS.train_dir, global_step=batch+1)
            log('done')
        else:
            restore_latest(saver, sess, '/tmp/derandomizing_dropout', suffix='-100001')

        err, = compute_test_percent_error()
        log('Test classification error = %s%%' % err)

        coord.request_stop()
        coord.join(threads)
        sess.close()

def main(_):
    if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    tf.gfile.MakeDirs(FLAGS.summaries_dir)
    train()

if __name__ == '__main__':
    flags.DEFINE_integer('max_steps', 100000, 'Number of steps to run trainer.')
    flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
    flags.DEFINE_string('summaries_dir', '/tmp/derandomizing_dropout/logs', 'Summaries directory')
    flags.DEFINE_string('train_dir', '/tmp/derandomizing_dropout/save', 'Saves directory')
    flags.DEFINE_integer('num_test_examples', 10000, """Number of examples to run.""")
    tf.app.run()
