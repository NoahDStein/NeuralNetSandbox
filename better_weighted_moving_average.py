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

from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.framework import ops
from tensorflow.python.training.moving_averages import assign_moving_average

# Same as tensorflow.python.training.moving_averages.weighted_moving_average but allows access to the average without updating
class WeightedMovingAverage(object):
    def __init__(self, value,
                            decay,
                            weight,
                            truediv=True,
                            collections=None,
                            name=None):

        """Compute the weighted moving average of `value`.
        Conceptually, the weighted moving average is:
          `moving_average(value * weight) / moving_average(weight)`,
        where a moving average updates by the rule
          `new_value = decay * old_value + (1 - decay) * update`
        Internally, this Op keeps moving average variables of both `value * weight`
        and `weight`.
        Args:
          value: A numeric `Tensor`.
          decay: A float `Tensor` or float value.  The moving average decay.
          weight:  `Tensor` that keeps the current value of a weight.
            Shape should be able to multiply `value`.
          truediv:  Boolean, if `True`, dividing by `moving_average(weight)` is
            floating point division.  If `False`, use division implied by dtypes.
          collections:  List of graph collections keys to add the internal variables
            `value * weight` and `weight` to.  Defaults to `[GraphKeys.VARIABLES]`.
          name: Optional name of the returned operation.
            Defaults to "WeightedMovingAvg".
        Returns:
          An Operation that updates and returns the weighted moving average.
        """
        # Unlike assign_moving_average, the weighted moving average doesn't modify
        # user-visible variables. It is the ratio of two internal variables, which are
        # moving averages of the updates.  Thus, the signature of this function is
        # quite different than assign_moving_average.
        if collections is None:
            collections = [ops.GraphKeys.VARIABLES]
        with variable_scope.variable_op_scope(
                [value, weight, decay], name, "WeightedMovingAvg") as scope:
            value_x_weight_var = variable_scope.get_variable(
                "value_x_weight",
                initializer=init_ops.zeros_initializer(value.get_shape(),
                                                       dtype=value.dtype),
                trainable=False,
                collections=collections)
            weight_var = variable_scope.get_variable(
                "weight",
                initializer=init_ops.zeros_initializer(weight.get_shape(),
                                                       dtype=weight.dtype),
                trainable=False,
                collections=collections)
            numerator = assign_moving_average(value_x_weight_var, value * weight, decay)
            denominator = assign_moving_average(weight_var, weight, decay)

            if truediv:
                div = math_ops.truediv
            else:
                div = math_ops.div
            self.average_with_update = div(numerator, denominator, name=scope.name)
            self.average = div(value_x_weight_var, weight_var)
