from typing import Tuple, Dict

import tensorflow_hub as hub
import tensorflow as tf
from official.projects.movinet.modeling import movinet
from official.projects.movinet.modeling import movinet_model
import re


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
    if hub_name.startswith('stem/stem'):
        print(hub_name)
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
    model.build(dummy_input.shape)
    checkpoint_dir = f'movinet_{model_id}_stream'
    checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
    checkpoint = tf.train.Checkpoint(model=model)
    status = checkpoint.restore(checkpoint_path).expect_partial()
    status.assert_existing_objects_matched()
    init_states_local = model.init_states(tf.shape(dummy_input))
    return model, init_states_local


class MoViNetKaras(tf.keras.Model):
    def __init__(self, hub_url):
        super(MoViNetKaras, self).__init__()
        self.encoder = hub.KerasLayer(hub_url)

    def call(self, X, states=None):
        if states:
            return self.encoder({**states, 'image': X})
        else:
            return self.encoder({'image': X})


model_id = 'a0'
num_classes = 600
H = W = 172
C = 3
T = 3
bs = 1

dummy_input = tf.random.normal(shape=[bs, T, H, W, 3])

hub_url = f"https://tfhub.dev/tensorflow/movinet/{model_id}/stream/kinetics-600/classification/"
model_hub = MoViNetKaras(hub_url)
init_states_fn = model_hub.encoder.resolved_object.signatures['init_states']
init_states = init_states_fn(tf.shape(dummy_input))

model_local, init_local = create_local(model_id)

local_weights = model_local.weights
hub_weights = model_hub.encoder.weights
print(local_weights[0])

local_weights_ = []
for var in hub_weights:
    local_weights_.append(hub_to_local(var))
local_weights = sorted(local_weights, key=lambda v: v.name)
local_weights_ = sorted(local_weights_, key=lambda v: v.name)
for i in range(len(local_weights)):
    assert local_weights[i].name == local_weights_[i].name, f'local_weights[i].name={local_weights[i].name}, local_weights_[i].name={local_weights_[i].name}'
for i in range(len(local_weights)):
    tf.debugging.assert_near(local_weights[i], local_weights_[i], atol=1e-4)