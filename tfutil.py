import numpy
import tensorflow as tf

from better_weighted_moving_average import WeightedMovingAverage

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
        var = tf.get_variable(name, *args, initializer=self.init(shape), **kwargs)
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
    def __init__(self, bn_unbias_forward=False):
        self.summaries = SummaryAccumulator()
        self.weight_factory = VariableFactory(init=OrthogonalInit(), summaries=self.summaries)
        self.bias_factory = VariableFactory(init=ConstantInit(0.1), summaries=self.summaries)
        self.scale_factory = VariableFactory(init=ConstantInit(numpy.log(numpy.exp(1)-1)), summaries=self.summaries)
        self.is_training = True
        self.bn_unbias_forward = bn_unbias_forward


    def nn_layer(self, input_tensor, output_dim, scope, act=tf.nn.relu, scale=None, bias=True, bn=False):
        if scale is None:
            scale = bn # scale should default to True for bn and False otherwise
        with tf.variable_scope(scope):
            layer_name = tf.get_variable_scope().name
            preactivate = 0.0
            try:
                num_inputs = len(input_tensor)
            except TypeError:
                num_inputs = 1
                input_tensor = [input_tensor]
            for i in xrange(num_inputs):
                input_dim = input_tensor[i].get_shape().as_list()[1]
                weights = self.weight_factory.get_variable('weight{}'.format(i), [input_dim, output_dim])
                preactivate = preactivate + tf.matmul(input_tensor[i], weights)
            if bn:
                preactivate = self.batch_normalization(preactivate, layer_name + '/bn')
            if scale:
                scale_var = self.scale_factory.get_variable('scale{}'.format(i), [1])
                preactivate = tf.nn.softplus(scale_var)*preactivate
            if bias:
                bias_var = self.bias_factory.get_variable('bias', [output_dim])
                preactivate = preactivate+bias_var
            self.summaries.histogram_summary(layer_name + '/pre_activations', preactivate)
            activations = act(preactivate)
            try:
                for i in xrange(len(activations)):
                    self.summaries.histogram_summary(layer_name + '/activations{}'.format(i), activations[i])
            except TypeError:
                self.summaries.histogram_summary(layer_name + '/activations', activations)
            return activations

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

    def batch_normalization(self, input_tensor, name, decay=0.95):
        mean_val, variance_val = tf.nn.moments(input_tensor, axes=[0])
        # Using weighted moving average with constant weight amounts to the bias correction that Adam uses for averaging
        mean_container = WeightedMovingAverage(mean_val, decay, tf.ones((1,)), collections='BatchNormInternal')
        variance_container = WeightedMovingAverage(variance_val, decay, tf.ones((1,)), collections='BatchNormInternal')
        if self.is_training:
            with tf.control_dependencies([mean_container.average_with_update, variance_container.average_with_update]):
                mean = tf.identity(mean_val)
                variance = tf.identity(variance_val)

            if self.bn_unbias_forward:
                mean = mean_val + tf.stop_gradient(mean_container.average_with_update - mean_val)
                # variance = variance_val + tf.stop_gradient(variance_container.average_with_update - variance_val)
                # variance = variance_val*tf.stop_gradient(variance_container.average_with_update / (variance_val + 1e-3))

            self.summaries.histogram_summary(name + '/mean (running)', mean_container.average)
            self.summaries.histogram_summary(name + '/mean (batch)', mean_val)
            self.summaries.histogram_summary(name + '/mean (diff)', mean_val - mean_container.average)

            self.summaries.histogram_summary(name + '/variance (running)', variance_container.average)
            self.summaries.histogram_summary(name + '/variance (batch)', variance_val)
            self.summaries.histogram_summary(name + '/variance (diff)', variance_val - variance_container.average)
        else:
            mean = mean_container.average
            variance = variance_container.average
        return tf.nn.batch_normalization(input_tensor, mean, variance, None, None, 1e-3)
