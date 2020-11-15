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
# pylint: disable=unused-import
"""Built-in metrics.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import types
import numpy as np
import six

from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.losses import binary_crossentropy
from tensorflow.python.keras.losses import categorical_crossentropy
from tensorflow.python.keras.losses import categorical_hinge
from tensorflow.python.keras.losses import cosine_similarity
from tensorflow.python.keras.losses import hinge
from tensorflow.python.keras.losses import kullback_leibler_divergence
from tensorflow.python.keras.losses import logcosh
from tensorflow.python.keras.losses import mean_absolute_error
from tensorflow.python.keras.losses import mean_absolute_percentage_error
from tensorflow.python.keras.losses import mean_squared_error
from tensorflow.python.keras.losses import mean_squared_logarithmic_error
from tensorflow.python.keras.losses import poisson
from tensorflow.python.keras.losses import sparse_categorical_crossentropy
from tensorflow.python.keras.losses import squared_hinge
from tensorflow.python.keras.utils import metrics_utils
from tensorflow.python.keras.utils.generic_utils import deserialize_keras_object
from tensorflow.python.keras.utils.generic_utils import serialize_keras_object
from tensorflow.python.keras.utils.generic_utils import to_list
from tensorflow.python.keras.utils.tf_utils import is_tensor_or_variable
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import confusion_matrix
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.ops import weights_broadcast_ops
from tensorflow.python.ops.losses import util as tf_losses_utils
from tensorflow.python.util.tf_export import keras_export
from tensorflow.tools.docs import doc_controls

import tensorflow as tf
from keras.backend import tensorflow_backend as K


@keras_export('keras.metrics.Metric')
@six.add_metaclass(abc.ABCMeta)
class Metric(Layer):
    """Encapsulates metric logic and state.

    Usage:

    ```python
    m = SomeMetric(...)
    for input in ...:
      m.update_state(input)
    print('Final result: ', m.result().numpy())
    ```

    Usage with tf.keras API:

    ```python
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    model.compile(optimizer=tf.compat.v1.train.RMSPropOptimizer(0.01),
                  loss=tf.keras.losses.categorical_crossentropy,
                  metrics=[tf.keras.metrics.CategoricalAccuracy()])

    data = np.random.random((1000, 32))
    labels = np.random.random((1000, 10))

    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.batch(32)
    dataset = dataset.repeat()

    model.fit(dataset, epochs=10, steps_per_epoch=30)
    ```

    To be implemented by subclasses:
    * `__init__()`: All state variables should be created in this method by
      calling `self.add_weight()` like: `self.var = self.add_weight(...)`
    * `update_state()`: Has all updates to the state variables like:
      self.var.assign_add(...).
    * `result()`: Computes and returns a value for the metric
      from the state variables.

    Example subclass implementation:

    ```
    class BinaryTruePositives(tf.keras.metrics.Metric):

      def __init__(self, name='binary_true_positives', **kwargs):
        super(BinaryTruePositives, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')

      def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(y_pred, tf.bool)

        values = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
        values = tf.cast(values, self.dtype)
        if sample_weight is not None:
          sample_weight = tf.cast(sample_weight, self.dtype)
          sample_weight = tf.broadcast_weights(sample_weight, values)
          values = tf.multiply(values, sample_weight)
        self.true_positives.assign_add(tf.reduce_sum(values))

      def result(self):
        return self.true_positives
    ```
    """

    def __init__(self, name=None, dtype=None, **kwargs):
        super(Metric, self).__init__(name=name, dtype=dtype, **kwargs)
        self.stateful = True  # All metric layers are stateful.
        self.built = True
        if not base_layer_utils.v2_dtype_behavior_enabled():
            # We only do this when the V2 behavior is not enabled, as when it is
            # enabled, the dtype already defaults to floatx.
            self._dtype = K.floatx() if dtype is None else dtypes.as_dtype(dtype).name

    def __new__(cls, *args, **kwargs):
        obj = super(Metric, cls).__new__(cls)

        # TODO(psv): We are excluding wrapping `update_state` of built-in metrics
        # with function here because of b/121302287. With this, built-in metrics
        # will continue to work with TPUs and custom metrics will not, however
        # users writing custom metrics need not worry about control dependencies
        # and returning ops.
        if cls.__module__ == Metric.__module__:
            update_state_fn = obj.update_state
        else:
            update_state_fn = def_function.function(obj.update_state)

        obj.update_state = types.MethodType(
            metrics_utils.update_state_wrapper(update_state_fn), obj)
        obj.result = types.MethodType(
            metrics_utils.result_wrapper(obj.result), obj)
        return obj

    def __call__(self, *args, **kwargs):
        """Accumulates statistics and then computes metric result value.

        Args:
          *args:
          **kwargs: A mini-batch of inputs to the Metric,
            passed on to `update_state()`.

        Returns:
          The metric value tensor.
        """

        def replica_local_fn(*args, **kwargs):
            """Updates the state of the metric in a replica-local context."""
            update_op = self.update_state(
                *args, **kwargs)  # pylint: disable=not-callable
            with ops.control_dependencies([update_op]):
                result_t = self.result()  # pylint: disable=not-callable

                # We are adding the metric object as metadata on the result tensor.
                # This is required when we want to use a metric with `add_metric` API on
                # a Model/Layer in graph mode. This metric instance will later be used
                # to reset variable state after each epoch of training.
                # Example:
                #   model = Model()
                #   mean = Mean()
                #   model.add_metric(mean(values), name='mean')
                result_t._metric_obj = self  # pylint: disable=protected-access
                return result_t

        from tensorflow.python.keras.distribute import distributed_training_utils  # pylint:disable=g-import-not-at-top
        return distributed_training_utils.call_replica_local_fn(
            replica_local_fn, *args, **kwargs)

    @property
    def dtype(self):
        return self._dtype

    def get_config(self):
        """Returns the serializable config of the metric."""
        return {'name': self.name, 'dtype': self.dtype}

    def reset_states(self):
        """Resets all of the metric state variables.

        This function is called between epochs/steps,
        when a metric is evaluated during training.
        """
        K.batch_set_value([(v, 0) for v in self.variables])

    @abc.abstractmethod
    def update_state(self, *args, **kwargs):
        """Accumulates statistics for the metric.

        Note: This function is executed as a graph function in graph mode.
        This means:
          a) Operations on the same resource are executed in textual order.
             This should make it easier to do things like add the updated
             value of a variable to another, for example.
          b) You don't need to worry about collecting the update ops to execute.
             All update ops added to the graph by this function will be executed.
          As a result, code should generally work the same way with graph or
          eager execution.

        Please use `tf.config.experimental_run_functions_eagerly(True)` to execute
        this function eagerly for debugging or profiling.

        Args:
          *args:
          **kwargs: A mini-batch of inputs to the Metric.
        """
        raise NotImplementedError('Must be implemented in subclasses.')

    @abc.abstractmethod
    def result(self):
        """Computes and returns the metric value tensor.

        Result computation is an idempotent operation that simply calculates the
        metric value using the state variables.
        """
        raise NotImplementedError('Must be implemented in subclasses.')

    ### For use by subclasses ###
    @doc_controls.for_subclass_implementers
    def add_weight(self,
                   name,
                   shape=(),
                   aggregation=tf_variables.VariableAggregation.SUM,
                   synchronization=tf_variables.VariableSynchronization.ON_READ,
                   initializer=None,
                   dtype=None):
        """Adds state variable. Only for use by subclasses."""
        from tensorflow.python.distribute import distribution_strategy_context as ds_context  # pylint:disable=g-import-not-at-top
        from tensorflow.python.keras.distribute import distributed_training_utils  # pylint:disable=g-import-not-at-top

        if ds_context.has_strategy():
            strategy = ds_context.get_strategy()
        else:
            strategy = None

        # TODO(b/120571621): Make `ON_READ` work with Keras metrics on TPU.
        if distributed_training_utils.is_tpu_strategy(strategy):
            synchronization = tf_variables.VariableSynchronization.ON_WRITE

        return super(Metric, self).add_weight(
            name=name,
            shape=shape,
            dtype=self._dtype if dtype is None else dtype,
            trainable=False,
            initializer=initializer,
            collections=[],
            synchronization=synchronization,
            aggregation=aggregation)

    ### End: For use by subclasses ###


@keras_export('keras.metrics.AUC0')
class AUC0(Metric):
    """AUC0 based on that 0 class is the "positive" class
    """
    """Computes the approximate AUC (Area under the curve) via a Riemann sum.

  This metric creates four local variables, `true_positives`, `true_negatives`,
  `false_positives` and `false_negatives` that are used to compute the AUC.
  To discretize the AUC curve, a linearly spaced set of thresholds is used to
  compute pairs of recall and precision values. The area under the ROC-curve is
  therefore computed using the height of the recall values by the false positive
  rate, while the area under the PR-curve is the computed using the height of
  the precision values by the recall.

  This value is ultimately returned as `auc`, an idempotent operation that
  computes the area under a discretized curve of precision versus recall values
  (computed using the aforementioned variables). The `num_thresholds` variable
  controls the degree of discretization with larger numbers of thresholds more
  closely approximating the true AUC. The quality of the approximation may vary
  dramatically depending on `num_thresholds`. The `thresholds` parameter can be
  used to manually specify thresholds which split the predictions more evenly.

  For best results, `predictions` should be distributed approximately uniformly
  in the range [0, 1] and not peaked around 0 or 1. The quality of the AUC
  approximation may be poor if this is not the case. Setting `summation_method`
  to 'minoring' or 'majoring' can help quantify the error in the approximation
  by providing lower or upper bound estimate of the AUC.

  If `sample_weight` is `None`, weights default to 1.
  Use `sample_weight` of 0 to mask values.

  Usage:

  ```python
  m = tf.keras.metrics.AUC(num_thresholds=3)
  m.update_state([0, 0, 1, 1], [0, 0.5, 0.3, 0.9])

  # threshold values are [0 - 1e-7, 0.5, 1 + 1e-7]
  # tp = [2, 1, 0], fp = [2, 0, 0], fn = [0, 1, 2], tn = [0, 2, 2]
  # recall = [1, 0.5, 0], fp_rate = [1, 0, 0]
  # auc = ((((1+0.5)/2)*(1-0))+ (((0.5+0)/2)*(0-0))) = 0.75

  print('Final result: ', m.result().numpy())  # Final result: 0.75
  ```

  Usage with tf.keras API:

  ```python
  model = tf.keras.Model(inputs, outputs)
  model.compile('sgd', loss='mse', metrics=[tf.keras.metrics.AUC()])
  ```
  """

    def __init__(self,
                 num_thresholds=200,
                 curve='ROC',
                 summation_method='interpolation',
                 name=None,
                 dtype=None,
                 thresholds=None):
        """Creates an `AUC` instance.

        Args:
          num_thresholds: (Optional) Defaults to 200. The number of thresholds to
            use when discretizing the roc curve. Values must be > 1.
          curve: (Optional) Specifies the name of the curve to be computed, 'ROC'
            [default] or 'PR' for the Precision-Recall-curve.
          summation_method: (Optional) Specifies the Riemann summation method used
            (https://en.wikipedia.org/wiki/Riemann_sum): 'interpolation' [default],
              applies mid-point summation scheme for `ROC`. For PR-AUC, interpolates
              (true/false) positives but not the ratio that is precision (see Davis
              & Goadrich 2006 for details); 'minoring' that applies left summation
              for increasing intervals and right summation for decreasing intervals;
              'majoring' that does the opposite.
          name: (Optional) string name of the metric instance.
          dtype: (Optional) data type of the metric result.
          thresholds: (Optional) A list of floating point values to use as the
            thresholds for discretizing the curve. If set, the `num_thresholds`
            parameter is ignored. Values should be in [0, 1]. Endpoint thresholds
            equal to {-epsilon, 1+epsilon} for a small positive epsilon value will
            be automatically included with these to correctly handle predictions
            equal to exactly 0 or 1.
        """
        # Validate configurations.
        if isinstance(curve, metrics_utils.AUCCurve) and curve not in list(
                metrics_utils.AUCCurve):
            raise ValueError('Invalid curve: "{}". Valid options are: "{}"'.format(
                curve, list(metrics_utils.AUCCurve)))
        if isinstance(
            summation_method,
            metrics_utils.AUCSummationMethod) and summation_method not in list(
                metrics_utils.AUCSummationMethod):
            raise ValueError(
                'Invalid summation method: "{}". Valid options are: "{}"'.format(
                    summation_method, list(metrics_utils.AUCSummationMethod)))

        # Update properties.
        if thresholds is not None:
            # If specified, use the supplied thresholds.
            self.num_thresholds = len(thresholds) + 2
            thresholds = sorted(thresholds)
        else:
            if num_thresholds <= 1:
                raise ValueError('`num_thresholds` must be > 1.')

            # Otherwise, linearly interpolate (num_thresholds - 2) thresholds in
            # (0, 1).
            self.num_thresholds = num_thresholds
            thresholds = [(i + 1) * 1.0 / (num_thresholds - 1)
                          for i in range(num_thresholds - 2)]

        # Add an endpoint "threshold" below zero and above one for either
        # threshold method to account for floating point imprecisions.
        self.thresholds = [0.0 - K.epsilon()] + thresholds + \
            [1.0 + K.epsilon()]

        if isinstance(curve, metrics_utils.AUCCurve):
            self.curve = curve
        else:
            self.curve = metrics_utils.AUCCurve.from_str(curve)
        if isinstance(summation_method, metrics_utils.AUCSummationMethod):
            self.summation_method = summation_method
        else:
            self.summation_method = metrics_utils.AUCSummationMethod.from_str(
                summation_method)
        super(AUC0, self).__init__(name=name, dtype=dtype)

        # Create metric variables
        self.true_positives = self.add_weight(
            'true_positives',
            shape=(self.num_thresholds,),
            initializer=init_ops.zeros_initializer)
        self.true_negatives = self.add_weight(
            'true_negatives',
            shape=(self.num_thresholds,),
            initializer=init_ops.zeros_initializer)
        self.false_positives = self.add_weight(
            'false_positives',
            shape=(self.num_thresholds,),
            initializer=init_ops.zeros_initializer)
        self.false_negatives = self.add_weight(
            'false_negatives',
            shape=(self.num_thresholds,),
            initializer=init_ops.zeros_initializer)

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates confusion matrix statistics.

        Args:
          y_true: The ground truth values.
          y_pred: The predicted values.
          sample_weight: Optional weighting of each example. Defaults to 1. Can be a
            `Tensor` whose rank is either 0, or the same rank as `y_true`, and must
            be broadcastable to `y_true`.
          y_true_0: The ground truth values based on that 0 class is the "positive" class.
          y_pred_0: The predicted values based on that 0 class is the "positive".
        Returns:
          Update op.
        """
        y_true = K.cast(y_true, self.dtype)
        y_pred = K.cast(y_pred, self.dtype)
        y_true = tf.map_fn(lambda v: 1 - v, y_true)
        y_pred = tf.map_fn(lambda v: 1 - v, y_pred)
        return metrics_utils.update_confusion_matrix_variables({
            metrics_utils.ConfusionMatrix.TRUE_POSITIVES: self.true_positives,
            metrics_utils.ConfusionMatrix.TRUE_NEGATIVES: self.true_negatives,
            metrics_utils.ConfusionMatrix.FALSE_POSITIVES: self.false_positives,
            metrics_utils.ConfusionMatrix.FALSE_NEGATIVES: self.false_negatives,
        }, y_true, y_pred, self.thresholds, sample_weight=sample_weight)

    def interpolate_pr_auc(self):
        """Interpolation formula inspired by section 4 of Davis & Goadrich 2006.

        https://www.biostat.wisc.edu/~page/rocpr.pdf

        Note here we derive & use a closed formula not present in the paper
        as follows:

          Precision = TP / (TP + FP) = TP / P

        Modeling all of TP (true positive), FP (false positive) and their sum
        P = TP + FP (predicted positive) as varying linearly within each interval
        [A, B] between successive thresholds, we get

          Precision slope = dTP / dP
                          = (TP_B - TP_A) / (P_B - P_A)
                          = (TP - TP_A) / (P - P_A)
          Precision = (TP_A + slope * (P - P_A)) / P

        The area within the interval is (slope / total_pos_weight) times

          int_A^B{Precision.dP} = int_A^B{(TP_A + slope * (P - P_A)) * dP / P}
          int_A^B{Precision.dP} = int_A^B{slope * dP + intercept * dP / P}

        where intercept = TP_A - slope * P_A = TP_B - slope * P_B, resulting in

          int_A^B{Precision.dP} = TP_B - TP_A + intercept * log(P_B / P_A)

        Bringing back the factor (slope / total_pos_weight) we'd put aside, we get

          slope * [dTP + intercept *  log(P_B / P_A)] / total_pos_weight

        where dTP == TP_B - TP_A.

        Note that when P_A == 0 the above calculation simplifies into

          int_A^B{Precision.dTP} = int_A^B{slope * dTP} = slope * (TP_B - TP_A)

        which is really equivalent to imputing constant precision throughout the
        first bucket having >0 true positives.

        Returns:
          pr_auc: an approximation of the area under the P-R curve.
        """
        dtp = self.true_positives[:self.num_thresholds -
                                  1] - self.true_positives[1:]
        p = self.true_positives + self.false_positives
        dp = p[:self.num_thresholds - 1] - p[1:]

        prec_slope = math_ops.div_no_nan(
            dtp, math_ops.maximum(dp, 0), name='prec_slope')
        intercept = self.true_positives[1:] - \
            math_ops.multiply(prec_slope, p[1:])

        safe_p_ratio = array_ops.where(
            math_ops.logical_and(p[:self.num_thresholds - 1] > 0, p[1:] > 0),
            math_ops.div_no_nan(
                p[:self.num_thresholds - 1],
                math_ops.maximum(p[1:], 0),
                name='recall_relative_ratio'),
            array_ops.ones_like(p[1:]))

        return math_ops.reduce_sum(
            math_ops.div_no_nan(
                prec_slope * (dtp + intercept * math_ops.log(safe_p_ratio)),
                math_ops.maximum(self.true_positives[1:] + self.false_negatives[1:],
                                 0),
                name='pr_auc_increment'),
            name='interpolate_pr_auc')

    def result(self):
        if (self.curve == metrics_utils.AUCCurve.PR and
                    self.summation_method == metrics_utils.AUCSummationMethod.INTERPOLATION
                ):
            # This use case is different and is handled separately.
            return self.interpolate_pr_auc()

        # Set `x` and `y` values for the curves based on `curve` config.
        recall = math_ops.div_no_nan(self.true_positives,
                                     self.true_positives + self.false_negatives)
        if self.curve == metrics_utils.AUCCurve.ROC:
            fp_rate = math_ops.div_no_nan(self.false_positives,
                                          self.false_positives + self.true_negatives)
            x = fp_rate
            y = recall
        else:  # curve == 'PR'.
            precision = math_ops.div_no_nan(
                self.true_positives, self.true_positives + self.false_positives)
            x = recall
            y = precision

        # Find the rectangle heights based on `summation_method`.
        if self.summation_method == metrics_utils.AUCSummationMethod.INTERPOLATION:
            # Note: the case ('PR', 'interpolation') has been handled above.
            heights = (y[:self.num_thresholds - 1] + y[1:]) / 2.
        elif self.summation_method == metrics_utils.AUCSummationMethod.MINORING:
            heights = math_ops.minimum(y[:self.num_thresholds - 1], y[1:])
        else:  # self.summation_method = metrics_utils.AUCSummationMethod.MAJORING:
            heights = math_ops.maximum(y[:self.num_thresholds - 1], y[1:])

        # Sum up the areas of all the rectangles.
        return math_ops.reduce_sum(
            math_ops.multiply(x[:self.num_thresholds - 1] - x[1:], heights),
            name=self.name)

    def reset_states(self):
        K.batch_set_value(
            [(v, np.zeros((self.num_thresholds,))) for v in self.variables])

    def get_config(self):
        config = {
            'num_thresholds': self.num_thresholds,
            'curve': self.curve.value,
            'summation_method': self.summation_method.value,
            # We remove the endpoint thresholds as an inverse of how the thresholds
            # were initialized. This ensures that a metric initialized from this
            # config has the same thresholds.
            'thresholds': self.thresholds[1:-1],
        }
        base_config = super(AUC0, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
