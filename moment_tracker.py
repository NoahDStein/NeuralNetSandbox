# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import tensorflow as tf

from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.framework import ops
from tensorflow.python.training.moving_averages import assign_moving_average

# Adapted from tensorflow.python.training.moving_averages.weighted_moving_average
# Using weighted moving average with constant weight amounts to the bias correction that Adam uses for averaging
class MomentTracker(object):
    def __init__(self, value, decay,
                 truediv=True,
                 collections=None,
                 reduction_indices=None,
                 name=None):
        self.value = value
        self.reduction_indices = reduction_indices or [0]

        eps = 1e-8
        if truediv:
            div = math_ops.truediv
        else:
            div = math_ops.div
        if collections is None:
            collections = [ops.GraphKeys.VARIABLES]

        value_shape = value.get_shape().as_list()
        shape = []
        for dim in xrange(len(value_shape)):
            if dim in self.reduction_indices:
                shape.append(1)
            else:
                shape.append(value_shape[dim])

        with variable_scope.variable_op_scope(
                [value, decay], name, "MomentTracker") as scope:

            mean_x_weight_var = variable_scope.get_variable("mean_x_weight", trainable=False, collections=collections,
                initializer=init_ops.zeros_initializer(shape, dtype=value.dtype))

            variance_x_weight_var = variable_scope.get_variable("variance_x_weight", trainable=False,
                collections=collections, initializer=init_ops.zeros_initializer(shape, dtype=value.dtype))

            weight_var = variable_scope.get_variable("weight", trainable=False, collections=collections,
                initializer=init_ops.zeros_initializer([1], dtype=tf.float32))

            self.tracked_mean = div(mean_x_weight_var, weight_var + eps)
            self.tracked_variance = div(variance_x_weight_var, weight_var + eps)

            self.batch_mean, self.batch_variance = tf.nn.moments(self.value, axes=self.reduction_indices,
                                                                 shift=self.tracked_mean, keep_dims=True)

            mean_numerator = assign_moving_average(mean_x_weight_var, self.batch_mean, decay)
            variance_numerator = assign_moving_average(variance_x_weight_var, self.batch_variance, decay)
            denominator = assign_moving_average(weight_var, 1.0, decay)

            self.update_mean = div(mean_numerator, denominator + eps, name=scope.name)
            self.update_variance = div(variance_numerator, denominator + eps, name=scope.name)
