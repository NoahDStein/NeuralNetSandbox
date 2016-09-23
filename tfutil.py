import time
import os
import numpy
import tensorflow as tf

from moment_tracker import MomentTracker

def listify(x):
    try:
        len(x)
    except TypeError:
        return [x]
    return x


class SummaryAccumulator(object):
    def __init__(self):
        self._summaries = []

    def reset(self):
        self.__init__()

    def merge_all_summaries(self):
        return tf.merge_summary(self._summaries)

    def scalar_summary(self, *args, **kwargs):
        summary = tf.scalar_summary(*args, **kwargs)
        self._summaries.append(summary)
        return summary

    def image_summary(self, *args, **kwargs):
        summary = tf.image_summary(*args, **kwargs)
        self._summaries.append(summary)
        return summary

    def audio_summary(self, *args, **kwargs):
        summary = tf.audio_summary(*args, **kwargs)
        self._summaries.append(summary)
        return summary

    def histogram_summary(self, *args, **kwargs):
        summary = tf.histogram_summary(*args, **kwargs)
        self._summaries.append(summary)
        return summary

class ConstantInit(object):
    def __init__(self, constant):
        self.constant = constant

    def __call__(self, shape):
        return tf.constant(self.constant, shape=shape, dtype=tf.float32)


class RandomInit(object):
    def __init__(self, lim):
        self.lim = lim

    def __call__(self, shape):
        return tf.truncated_normal(shape, stddev=self.lim)

class OrthogonalInit(object):
    def __call__(self, shape):
        if len(shape) != 2:
            raise ValueError('Requested shape {}, but OrthogonalInitializer can only provide matrices.'.format(shape))
        x = numpy.random.randn(*shape)
        u, _, v = numpy.linalg.svd(x, full_matrices=False)
        if shape[0] < shape[1]:
            return tf.constant(v, dtype=tf.float32)
        else:
            return tf.constant(u, dtype=tf.float32)


class VariableFactory(object):
    def __init__(self, init, summaries):
        self.variables = []
        self.init = init
        self.summaries = summaries

    def get_variable(self, name, shape, *args, **kwargs):
        try:
            var = tf.get_variable(name, *args, initializer=self.init(shape), **kwargs)
        except TypeError:
            var = tf.get_variable(name, *args, **kwargs)
        if not tf.get_variable_scope().reuse:
            self.variables.append(var)
        if self.summaries is not None and not tf.get_variable_scope().reuse:
            summary_name = tf.get_variable_scope().name + '/' + name
            with tf.name_scope('summaries'):
                mean = tf.reduce_mean(var)
                self.summaries.scalar_summary(summary_name + '(mean)', mean)
                with tf.name_scope('stddev'):
                    stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
                self.summaries.scalar_summary(summary_name + '(stddev)', stddev)
                self.summaries.scalar_summary(summary_name + '(max)', tf.reduce_max(var))
                self.summaries.scalar_summary(summary_name + '(min)', tf.reduce_min(var))
                self.summaries.histogram_summary(summary_name, var)
        return var



class LayerManager(object):
    def __init__(self, forward_biased_estimate=False, is_training=True, auto_summaries=True):
        self.summaries = SummaryAccumulator()
        self.auto_summaries = auto_summaries
        if self.auto_summaries:
            var_summaries = self.summaries
        else:
            var_summaries = None
        self.weight_factory = VariableFactory(init=OrthogonalInit(), summaries=var_summaries)
        self.filter_factory = VariableFactory(init=RandomInit(0.1), summaries=var_summaries)
        self.bias_factory = VariableFactory(init=ConstantInit(0.0), summaries=var_summaries)
        self.scale_factory = VariableFactory(init=ConstantInit(numpy.log(numpy.exp(1)-1)), summaries=var_summaries)
        self.is_training = is_training
        self.forward_biased_estimate = forward_biased_estimate

    def variables(self):
        return self.weight_factory.variables + self.filter_factory.variables + self.bias_factory.variables + self.scale_factory.variables

    def nn_layer(self, input_tensor, output_dim, scope, act, scale=None, bias=True, bn=False):
        if scale is None:
            scale = bn  # scale should default to True for bn and False otherwise
        with tf.variable_scope(scope):
            layer_name = tf.get_variable_scope().name
            preactivate = 0.0
            input_tensor = listify(input_tensor)
            extra_dims = len(input_tensor[0].get_shape()) - 2

            for i in range(len(input_tensor)):
                input_dim = input_tensor[i].get_shape().as_list()[-1]
                weights = self.weight_factory.get_variable('weight{}'.format(i), [input_dim, output_dim])
                if extra_dims == 0:
                    preactivate = preactivate + tf.matmul(input_tensor[i], weights)
                else:
                    shape = input_tensor[i].get_shape()
                    reshaped_input = tf.reshape(input_tensor[i], [-1, input_dim])
                    product = tf.matmul(reshaped_input, weights)
                    new_shape = shape.as_list()
                    new_shape[0] = -1
                    new_shape[-1] = output_dim
                    preactivate = preactivate + tf.reshape(product, new_shape)
            if bn:
                preactivate = self.batch_normalization(preactivate, layer_name + '/bn', normalization_indices=list(range(extra_dims + 1)))
            if scale:
                scale_var = self.scale_factory.get_variable('scale{}'.format(i), [1])
                preactivate = tf.nn.softplus(scale_var)*preactivate
            if bias:
                bias_var = self.bias_factory.get_variable('bias', [1] * extra_dims + [output_dim])
                preactivate = preactivate+bias_var
            self.summaries.histogram_summary(layer_name + '/pre_activations', preactivate)
            activations = act(preactivate)
            if self.auto_summaries:
                try:
                    for i in range(len(activations)):
                        self.summaries.histogram_summary(layer_name + '/activations{}'.format(i), activations[i])
                except TypeError:
                    self.summaries.histogram_summary(layer_name + '/activations', activations)
            return activations


    def conv_layer(self, input_tensor, filter_height, filter_width, num_filters, scope, act, padding='SAME', strides=None, scale=None, bias=True, bn=False):
        if scale is None:
            scale = bn  # scale should default to True for bn and False otherwise
        with tf.variable_scope(scope):
            layer_name = tf.get_variable_scope().name
            preactivate = 0.0
            input_tensor = listify(input_tensor)
            for i in range(len(input_tensor)):
                num_channels_in = input_tensor[i].get_shape().as_list()[3]
                filters = self.filter_factory.get_variable('filter{}'.format(i), [filter_height, filter_width, num_channels_in, num_filters])
                preactivate = preactivate + tf.nn.conv2d(input_tensor[i], filters, strides=strides or [1, 1, 1, 1], padding=padding)
            if bn:
                preactivate = self.batch_normalization(preactivate, layer_name + '/bn', normalization_indices=[0, 1, 2])
            if scale:
                scale_var = self.scale_factory.get_variable('scale{}'.format(i), [1])
                preactivate = tf.nn.softplus(scale_var)*preactivate
            if bias:
                bias_var = self.bias_factory.get_variable('bias', [num_filters])
                preactivate = preactivate+bias_var
            self.summaries.histogram_summary(layer_name + '/pre_activations', preactivate)
            activations = act(preactivate)
            try:
                for i in range(len(activations)):
                    self.summaries.histogram_summary(layer_name + '/activations{}'.format(i), activations[i])
            except TypeError:
                self.summaries.histogram_summary(layer_name + '/activations', activations)
            return activations


    def conv_transpose_layer(self, input_tensor, filter_height, filter_width, num_filters, scope, act, padding='SAME', strides=None, scale=None, bias=True, bn=False):
        if scale is None:
            scale = bn  # scale should default to True for bn and False otherwise
        with tf.variable_scope(scope):
            layer_name = tf.get_variable_scope().name
            preactivate = 0.0
            input_tensor = listify(input_tensor)
            for i in range(len(input_tensor)):
                input_shape = input_tensor[i].get_shape().as_list()
                num_channels_in = input_shape[3]
                filters = self.filter_factory.get_variable('filter{}'.format(i), [filter_height, filter_width, num_filters, num_channels_in])
                output_shape = [tf.shape(input_tensor[i])[0], input_shape[1]*strides[1], input_shape[2]*strides[2], num_filters]
                preactivate = preactivate + tf.nn.conv2d_transpose(input_tensor[i], filters, output_shape=output_shape, strides=strides or [1, 1, 1, 1], padding=padding)
            output_shape[0] = None
            preactivate.set_shape(output_shape)
            if bn:
                preactivate = self.batch_normalization(preactivate, layer_name + '/bn', normalization_indices=[0, 1, 2])
            if scale:
                scale_var = self.scale_factory.get_variable('scale{}'.format(i), [1])
                preactivate = tf.nn.softplus(scale_var)*preactivate
            if bias:
                bias_var = self.bias_factory.get_variable('bias', [num_filters])
                preactivate = preactivate+bias_var
            self.summaries.histogram_summary(layer_name + '/pre_activations', preactivate)
            activations = act(preactivate)
            try:
                for i in range(len(activations)):
                    self.summaries.histogram_summary(layer_name + '/activations{}'.format(i), activations[i])
            except TypeError:
                self.summaries.histogram_summary(layer_name + '/activations', activations)
            return activations


    def max_pool(self, value, *args, **kwargs):
        return tf.reduce_max(
            [tf.nn.max_pool(t, *args, **kwargs) for t in listify(value)],
            reduction_indices=[0])

    def avg_pool(self, value, *args, **kwargs):
        return tf.reduce_mean(
            [tf.nn.avg_pool(t, *args, **kwargs) for t in listify(value)],
            reduction_indices=[0])

    def reparam_normal_sample(self, mean, log_std, layer_name):
        with tf.name_scope(layer_name):
            prior_sample = tf.random_normal(tf.shape(mean))
            posterior_sample = mean + tf.exp(log_std) * prior_sample
        self.summaries.histogram_summary(layer_name + '/prior_sample', prior_sample)
        self.summaries.histogram_summary(layer_name + '/posterior_sample', posterior_sample)
        return posterior_sample

    def parametrized_sinusoid(self, sig_len, norm_freq, sin_weight, cos_weight):
        arg = tf.linspace(0.0, 2*numpy.pi*(sig_len-1), sig_len)*norm_freq
        return sin_weight*tf.sin(arg) + cos_weight*tf.cos(arg)

    def batch_normalization(self, input_tensor, name, decay=0.95, normalization_indices=None):
        mt = MomentTracker(input_tensor, decay=decay, collections=['BatchNormInternal'], reduction_indices=normalization_indices or [0])
        if self.is_training:
            with tf.control_dependencies([mt.update_mean]):
                mean = tf.identity(mt.batch_mean)
            with tf.control_dependencies([mt.update_variance]):
                variance = tf.identity(mt.batch_variance)

            if self.forward_biased_estimate:
                mean = mean + tf.stop_gradient(mt.update_mean - mean)
                #variance = variance + tf.stop_gradient(mt.update_variance - variance)
                #variance = variance*tf.stop_gradient(mt.tracked_variance / (variance + 1e-3))

            self.summaries.histogram_summary(name + '/mean (running)', mt.tracked_mean)
            self.summaries.histogram_summary(name + '/mean (batch)', mt.batch_mean)
            self.summaries.histogram_summary(name + '/mean (diff)', mt.batch_mean - mt.tracked_mean)

            self.summaries.histogram_summary(name + '/variance (running)', mt.tracked_variance)
            self.summaries.histogram_summary(name + '/variance (batch)', mt.batch_variance)
            self.summaries.histogram_summary(name + '/variance (diff)', mt.batch_variance - mt.tracked_variance)
        else:
            mean = mt.tracked_mean
            variance = mt.tracked_variance
        return tf.nn.batch_normalization(input_tensor, mean, variance, None, None, 1e-3)

# Not averaged over examples
def multi_class_hinge_loss(inner_products, labels, power=1):
    membership_indicator = tf.one_hot(labels, tf.shape(inner_products)[1], on_value=1.0, off_value=-1.0)
    hinge_loss = tf.nn.relu(1.0 - membership_indicator * inner_products)
    if power !=1:
        if power == 2:
            hinge_loss = tf.square(hinge_loss)
        else:
            hinge_loss = tf.pow(hinge_loss, power)
    return tf.reduce_sum(hinge_loss, reduction_indices=[1])

def log(s):
    print('[%s] ' % time.asctime() + s)


def restore_latest(saver, sess, path, suffix=''):
    dated_files = [(os.path.getmtime(path + '/' + fn), os.path.basename(fn)) for fn in os.listdir(path) if
                   fn.startswith('save') and fn.endswith(suffix) and os.path.splitext(fn)[1] == '']
    dated_files.sort()
    dated_files.reverse()
    newest = dated_files[0][1]
    log('restoring %s updated at %s' % (dated_files[0][1], time.ctime(dated_files[0][0])))
    saver.restore(sess, path + '/' + newest)


def modified_dynamic_shape(tensor, new_shape):
    return tuple(new_dim or tf.shape(tensor)[i] for i, new_dim in enumerate(new_shape))


def modified_static_shape(tensor, new_shape):
    return tuple(new_dim or tensor.get_shape().as_list()[i] for i, new_dim in enumerate(new_shape))


def roll0d(tensor, shift):  # roll tensor along 0th dimension
    n = tensor.get_shape()[0]
    if shift >= 0:
        z = tf.concat(concat_dim=0, values=[tf.gather(tensor, indices=tf.range(n-shift, n)), \
                                           tf.gather(tensor, indices=tf.range(n-shift))])
    else:
        z = tf.concat(concat_dim=0, values=[tf.gather(tensor, indices=tf.range(-shift, n)), \
                                           tf.gather(tensor, indices=tf.range(-shift))])
    return z


def tensor_roll_scalar(tensor, shift, axis, ndim):  # roll a tensor by a scalar argument
    dims = numpy.arange(ndim)
    if axis != 0:
        dims[axis], dims[0] = dims[0], dims[axis]
        permuted_tensor = tf.transpose(tensor, perm=dims)
    else:
        permuted_tensor = tensor
    return tf.transpose(roll0d(permuted_tensor, shift), perm=dims)


def quantizer(val, lower, upper, levels):
    normalized = (tf.clip_by_value(val, lower, upper) - lower)/(upper - lower + 1e-6)
    return tf.cast(tf.floor(normalized*levels), tf.int64)


def dequantizer(val, lower, upper, levels):
    return lower + (upper - lower) * tf.cast(val, tf.float32) / tf.cast(levels - 1, tf.float32)


def draw_on(canvas, to_draw, color):
    to_draw = tf.expand_dims(to_draw, 3)
    color = tf.constant(color, tf.float32)[None, None, None, :]
    return canvas*(1.0 - to_draw) + color*to_draw


def crappy_plot(val, levels):
    x_len = val.get_shape().as_list()[1]
    left_val = tf.concat(1, (val[:, 0:1], val[:, 0:x_len - 1]))
    right_val = tf.concat(1, (val[:, 1:], val[:, x_len - 1:]))

    left_mean = (val + left_val) // 2
    right_mean = (val + right_val) // 2
    low_val = tf.minimum(tf.minimum(left_mean, right_mean), val)
    high_val = tf.maximum(tf.maximum(left_mean, right_mean), val + 1)
    return tf.cumsum(tf.one_hot(low_val, levels, axis=1) - tf.one_hot(high_val, levels, axis=1), axis=1)


def queue_append_and_update(axis, old_contents, contents_to_append):
    ndims = old_contents.get_shape().ndims
    slice_begin = numpy.zeros(shape=(ndims,), dtype=numpy.int32)
    slice_begin[axis] = contents_to_append.get_shape().as_list()[axis]
    slice_size = -numpy.ones(shape=(ndims,), dtype=numpy.int32)
    concatenated_contents = tf.concat(axis, (old_contents[:contents_to_append.get_shape().as_list()[0], ...], contents_to_append))
    paddings = [[0, 0]] * ndims
    paddings[0] = [0, old_contents.get_shape().as_list()[0] - contents_to_append.get_shape().as_list()[0]]
    updated_contents = tf.pad(tf.slice(concatenated_contents, slice_begin, slice_size), paddings)
    return concatenated_contents, updated_contents
