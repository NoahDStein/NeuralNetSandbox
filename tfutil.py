import numpy
import tensorflow as tf

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
    def __init__(self):
        self.summaries = SummaryAccumulator()
        self.weight_factory = VariableFactory(init=OrthogonalInit(), summaries=self.summaries)
        self.bias_factory = VariableFactory(init=ConstantInit(0.1), summaries=self.summaries)
        self.scale_factory = VariableFactory(init=ConstantInit(1.0), summaries=self.summaries)


    def nn_layer(self, input_tensor, output_dim, scope, act=tf.nn.relu):
        """Reusable code for making a simple neural net layer.
        It does a matrix multiply, bias add, and then uses relu to nonlinearize.
        It also sets up name scoping so that the resultant graph is easy to read,
        and adds a number of summary ops.
        """
        with tf.variable_scope(scope):
            layer_name = tf.get_variable_scope().name
            biases = self.bias_factory.get_variable('bias', [output_dim])
            preactivate = biases
            try:
                num_inputs = len(input_tensor)
            except TypeError:
                num_inputs = 1
                input_tensor = [input_tensor]
            for i in xrange(num_inputs):
                input_dim = input_tensor[i].get_shape().as_list()[1]
                scale = self.scale_factory.get_variable('scale{}'.format(i), [1])
                weights = self.weight_factory.get_variable('weight{}'.format(i), [input_dim, output_dim])
                preactivate = preactivate + scale*tf.matmul(input_tensor[i], weights)
    
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
