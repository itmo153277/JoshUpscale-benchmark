# -*- encoding: utf-8 -*-

"""Tensorflow types for static checks."""

from typing import Union

import numpy as np
import tensorflow as tf
try:
    from keras.engine import keras_tensor
except ImportError:
    from tensorflow.python.keras.engine import keras_tensor

KerasTensor = keras_tensor.KerasTensor
TensorLike = Union[
    np.ndarray,
    tf.Tensor,
    tf.SparseTensor,
    tf.Variable,
    KerasTensor,
]
