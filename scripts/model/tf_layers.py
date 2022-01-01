# -*- encoding: utf-8 -*-

"""Custom tensorflow layers."""

from typing import Any, Dict, List
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers
from tf_types import KerasTensor


class UpscaleLayer(layers.Layer):
    """Upscale layer (XLA-complaint)."""

    # pylint: disable=arguments-differ

    def __init__(self, scale: int = 2, **kwargs) -> None:
        """Construct object."""
        super().__init__(**kwargs)
        self.scale = scale

    def call(self, inputs: KerasTensor) -> tf.Tensor:
        """Execute forward pass."""
        return tf.compat.v1.image.resize_bilinear(
            images=inputs,
            size=tf.shape(inputs)[1:3] * self.scale,
            align_corners=False,
            half_pixel_centers=False
        )

    def get_config(self) -> Dict[str, Any]:
        """Serialize config."""
        config = super().get_config()
        return {
            **config,
            "scale": self.scale
        }


class DenseWarpLayer(layers.Layer):
    """Layer for dense-warping (tensorflow_addons)."""

    # pylint: disable=arguments-differ

    def call(self, inputs: List[KerasTensor]) -> KerasTensor:
        """Execute forward pass."""
        return tfa.image.dense_image_warp(inputs[0], inputs[1])


class SpaceToDepth(layers.Layer):
    """Space to depth layer."""

    # pylint: disable=arguments-differ

    def __init__(self, block_size: int = 2, **kwargs) -> None:
        """Construct object."""
        super().__init__(**kwargs)
        self.block_size = block_size

    def call(self, inputs: KerasTensor) -> tf.Tensor:
        """Execute forward pass."""
        return tf.nn.space_to_depth(inputs, self.block_size)

    def get_config(self) -> Dict[str, Any]:
        """Serialize config."""
        config = super().get_config()
        return {
            **config,
            "block_size": self.block_size
        }
