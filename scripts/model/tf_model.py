#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

"""Script for loading tensorflow models."""

import sys
from typing import Dict
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tf_layers import UpscaleLayer, DenseWarpLayer, SpaceToDepth


def load_model(config_path: str, weight_path: str) -> keras.Model:
    """Load tensorflow model."""
    with open(config_path, "rt", encoding="utf-8") as f:
        config = f.read()
    model = keras.models.model_from_json(config, custom_objects={
        "UpscaleLayer": UpscaleLayer,
        "DenseWarpLayer": DenseWarpLayer,
        "SpaceToDepth": SpaceToDepth,
    })
    model.load_weights(weight_path)
    return model


def create_inference_model(model: keras.Model) -> tf.compat.v1.GraphDef:
    """Create GraphDef from keras model."""
    # pylint: disable=import-outside-toplevel

    from tensorflow.python.framework.convert_to_constants \
        import convert_variables_to_constants_v2

    assert len(model.input) == 3

    @tf.function
    def model_inference(cur_frame, last_frame, pre_gen):
        """Run model for inference."""
        return model(
            [cur_frame, last_frame, pre_gen], training=False)

    model_inference = model_inference.get_concrete_function(
        cur_frame=tf.TensorSpec(
            model.input[0].shape,
            model.input[0].dtype
        ),
        last_frame=tf.TensorSpec(
            model.input[1].shape,
            model.input[1].dtype
        ),
        pre_gen=tf.TensorSpec(
            model.input[2].shape,
            model.input[2].dtype
        ))
    model_inference = convert_variables_to_constants_v2(model_inference)
    graph_def = model_inference.graph.as_graph_def()
    for node in graph_def.node:
        if node.name == model_inference.structured_outputs[0].op.name:
            node.name = "output"
            inputs_to_delete = [
                inp for inp in node.input if inp[0] == "^"
            ]
            for inp in inputs_to_delete:
                node.input.remove(inp)
    return graph_def


def convert_model_to_nchw(model: keras.Model,
                          name: str = "final") -> keras.Model:
    """Convert model from NHWC to NCHW."""
    assert len(model.input) == 3
    cur_frame = keras.Input(shape=np.array(
        model.input[0].shape)[[3, 1, 2]], name="cur_frame")
    last_frame = keras.Input(shape=np.array(
        model.input[1].shape)[[3, 1, 2]], name="last_frame")
    pre_gen = keras.Input(shape=np.array(
        model.input[2].shape)[[3, 1, 2]], name="pre_gen")
    cur_frame_pr = layers.Lambda(lambda x: K.permute_dimensions(
        x, pattern=[0, 2, 3, 1]))(cur_frame)
    last_frame_pr = layers.Lambda(lambda x: K.permute_dimensions(
        x, pattern=[0, 2, 3, 1]))(last_frame)
    pre_gen_pr = layers.Lambda(lambda x: K.permute_dimensions(
        x, pattern=[0, 2, 3, 1]))(pre_gen)
    output = model([cur_frame_pr, last_frame_pr, pre_gen_pr])
    output = layers.Lambda(lambda x: K.permute_dimensions(
        x, pattern=[0, 3, 1, 2]))(output)
    output = layers.Layer(name="output")(output)
    model = keras.Model(
        inputs=[cur_frame, last_frame, pre_gen], outputs=output, name=name)
    return model


def optimize_for_inference(graph_def: tf.compat.v1.GraphDef,
                           input_info: Dict[str, tf.TensorSpec],
                           output_info: Dict[str, tf.TensorSpec]) \
        -> tf.compat.v1.GraphDef:
    """Optimize GraphDef for inference (fusing, constant folding, etc.)."""
    # pylint: disable=import-outside-toplevel
    # pylint: disable=not-context-manager

    from tensorflow.python.grappler import tf_optimizer
    from tensorflow.python.training.saver import export_meta_graph
    from tensorflow.python.tools.optimize_for_inference_lib import \
        optimize_for_inference as tf_optimize_for_inference

    config_proto = tf.compat.v1.ConfigProto()
    graph = tf.Graph()
    with graph.as_default():
        tf.import_graph_def(graph_def, name="")
    for i in list(input_info.keys()) + list(output_info.keys()):
        graph.add_to_collection(
            tf.compat.v1.GraphKeys.TRAIN_OP,
            graph.get_operation_by_name(i)
        )
    metagraph = export_meta_graph(
        graph_def=graph_def, graph=graph, clear_devices=True)
    signature = tf.compat.v1.saved_model.build_signature_def(
        inputs={
            k: tf.compat.v1.TensorInfo(
                dtype=v.dtype.as_datatype_enum,
                tensor_shape=v.shape.as_proto(),
                name=k + ":0"
            )
            for k, v in input_info.items()
        },
        outputs={
            k: tf.compat.v1.TensorInfo(
                dtype=v.dtype.as_datatype_enum,
                tensor_shape=v.shape.as_proto(),
                name=k + ":0"
            )
            for k, v in output_info.items()
        },
        method_name=tf.saved_model.PREDICT_METHOD_NAME
    )
    metagraph.signature_def[tf.saved_model.SERVING].CopyFrom(signature)
    optimized_graph_def = tf_optimizer.OptimizeGraph(
        config_proto,
        metagraph,
        verbose=True,
        strip_default_attributes=True
    )
    optimized_graph_def = tf_optimize_for_inference(
        optimized_graph_def,
        list(input_info.keys()),
        list(output_info.keys()),
        [x.dtype.as_datatype_enum for x in input_info.values()]
    )
    return optimized_graph_def


def main() -> int:
    """Run CLI."""
    return 0


if __name__ == "__main__":
    sys.exit(main())
