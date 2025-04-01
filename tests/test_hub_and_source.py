import unittest
from typing import Tuple, Dict
import tensorflow_hub as hub
import tensorflow as tf
from six.moves import urllib
from io import BytesIO
from PIL import Image
from official.projects.movinet.modeling import movinet
from official.projects.movinet.modeling import movinet_model
import numpy as np
import re

model_id = 'a0'
num_classes = 600
H = W = 172
C = 3
T = 1
bs = 1


def create_hub_model(model_id) -> Tuple[tf.keras.Model, Dict]:
    hub_url = f"https://tfhub.dev/tensorflow/movinet/{model_id}/stream/kinetics-600/classification/"
    model_hub = hub.KerasLayer(hub_url)
    init_states_fn = model_hub.resolved_object.signatures['init_states']
    dummy_input = tf.random.normal(shape=[bs, T, H, W, 3])
    init_states = init_states_fn(tf.shape(dummy_input))
    return model_hub, init_states


def create_local(model_id) -> Tuple[movinet.Movinet, Dict]:
    backbone = movinet.Movinet(
        model_id=model_id,
        causal=True,
        conv_type='2plus1d',
        se_type='2plus3d',
        activation='hard_swish',
        gating_activation='hard_sigmoid',
        use_positional_encoding=False,
        use_external_states=True,
    )
    backbone.trainable = False
    model = movinet_model.MovinetClassifier(
        backbone,
        num_classes=600,
        output_states=True
    )
    checkpoint_dir = f'movinet_{model_id}_stream'
    checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
    checkpoint = tf.train.Checkpoint(model=model)
    status = checkpoint.restore(checkpoint_path).expect_partial()
    status.assert_existing_objects_matched()
    dummy_input = tf.random.normal(shape=[bs, T, H, W, 3])
    init_states_local = model.init_states(tf.shape(dummy_input))
    return model, init_states_local


def hub_name_to_local(hub_name: str) -> str:
    # hub -> local
    # b{i}/l{i} -> block{i}_layer{i}
    # expansion/bn -> expansion/conv2d/bn
    # feature/bn -> feature/conv2d/bn
    # projection/bn -> projection/conv2d/bn
    # bneck/feature/conv2d/bn_temporal -> bneck/feature/conv2d_temporal/bn_temporal
    # bneck/feature/conv2d/bn_temporal -> bneck/feature/conv2d_temporal/bn_temporal
    # head/project/bn -> head/project/conv2d/bn
    # stem/stem/bn -> stem/stem/conv2d/bn
    pattern = r"b(\d+)/l(\d+)"
    replacement = r"block\1_layer\2"
    res = re.sub(pattern, replacement, hub_name)
    res = res.replace('expansion/bn', 'expansion/conv2d/bn')
    res = res.replace('feature/bn', 'feature/conv2d/bn')
    res = res.replace('projection/bn', 'projection/conv2d/bn')
    res = res.replace('bneck/feature/conv2d/bn_temporal', 'bneck/feature/conv2d_temporal/bn_temporal')
    res = res.replace('head/project/bn', 'head/project/conv2d/bn')
    res = res.replace('stem/stem/bn', 'stem/stem/conv2d/bn')
    res = res.replace('bneck/skip/skip_project/bn', 'bneck/skip/skip_project/conv2d/bn')
    return res


def hub_to_local(hub_var) -> tf.Variable:
    dtype = hub_var.dtype
    initial_value = hub_var.read_value()  # Corrected line

    resource_variable = tf.compat.v1.get_variable(
        name=hub_name_to_local(hub_var.name)[:-2],
        initializer=initial_value,
        dtype=dtype,
        use_resource=True
    )

    return resource_variable


class MyTestCase(unittest.TestCase):

    def test_hub_equal_source(self):
        tf.keras.backend.clear_session()
        model_hub, states_hub = create_hub_model(model_id)
        image_url = 'https://upload.wikimedia.org/wikipedia/commons/8/84/Ski_Famille_-_Family_Ski_Holidays.jpg'
        with urllib.request.urlopen(image_url) as f:
            image = Image.open(BytesIO(f.read())).resize((H, W))
        X = tf.reshape(np.array(image), [1, 1, H, W, 3])
        X = tf.cast(X, tf.float32) / 255
        y_hub, _ = model_hub({**states_hub, 'image': X})
        print(y_hub[0][0:5])
        model_local, states_local = create_local(model_id)
        y_local, _ = model_local({**states_local, 'image': X})
        print(y_local[0][0:5])
        tf.debugging.assert_near(y_local, y_hub, atol=1e-3)
        del model_hub
        del model_local


    def test_hub_weights_equal_source(self):
        tf.keras.backend.clear_session()
        model_hub, _ = create_hub_model(model_id)
        model_source, states_source = create_local(model_id)
        hub_weights = model_hub.weights
        source_weights = model_source.weights
        self.assertEqual(len(source_weights), len(hub_weights))
        hub_weights_to_source = []
        for var in hub_weights:
            hub_weights_to_source.append(hub_to_local(var))
        source_weights = sorted(source_weights, key=lambda v: v.name)
        hub_weights_to_source = sorted(hub_weights_to_source, key=lambda v: v.name)
        for i in range(len(source_weights)):
            self.assertEqual(source_weights[i].name, hub_weights_to_source[
                i].name, f'source_weights[i].name={source_weights[i].name}, hub_weights_to_source[i].name={hub_weights_to_source[i].name}')
            tf.debugging.assert_near(source_weights[i], hub_weights_to_source[i], atol=1e-4)


if __name__ == '__main__':
    unittest.main()
