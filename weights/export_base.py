from typing import List, Tuple, Dict

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import torch
from einops import rearrange


def _key_translate_base(k, model_name):
    k = (k
         .replace("classifier_head/head/conv3d/", "classifier.0.conv_1.conv3d.")
         .replace("classifier_head/classifier/conv3d/", "classifier.3.conv_1.conv3d.")
         .replace("se/se_reduce/conv3d/", "se.fc1.conv_1.conv3d.")
         .replace("se/se_expand/conv3d/", "se.fc2.conv_1.conv3d.")
         .replace("stem/stem/", "conv1.conv_1.")
         .replace("conv3d/", "conv3d.")
         .replace("kernel:0", "weight")
         .replace("bias:0", "bias")
         .replace("bn/gamma:0", "norm.weight")
         .replace("bn/beta:0", "norm.bias")
         .replace("bn/moving_mean:0", "norm.running_mean")
         .replace("bn/moving_variance:0", "norm.running_var")
         .replace("skip/skip_project/", "res.1.conv_1.")
         .replace("expansion/", "expand.conv_1.")
         .replace("feature/", "deep.conv_1.")
         .replace("projection/", "project.conv_1.")
         .replace("scale:0", "alpha")
         .replace("head/project/", "conv7.conv_1."))
    for i in range(5):
        for j in range(20):
            k = k.replace(f"b{i}/l{j}/bneck/", f"blocks.b{i}_l{j}.").replace(f"b{i}/l{j}/", f"blocks.b{i}_l{j}.")
    if (model_name == "a3" or model_name == "a5") and "b3_l0" in k:
        k = k.replace("res.1.", "res.0.")
    return k


def _weight_translate(w, name):
    if len(w.shape) == 5:
        w = rearrange(w, "d h w c_in c_out -> c_out c_in d h w")
    if len(w.shape) == 4:
        #w = rearrange(w, "h w c_in c_out -> c_out c_in h w")
        if "feature" in name:
            w = rearrange(w, "h w c_out c_in-> c_out c_in h w")
        else:
            w = rearrange(w, "h w c_in c_out -> c_out c_in h w")
    return torch.tensor(w)


def _get_tf_model_params(model_name: str) -> List[Tuple[str, np.ndarray]]:
    encoder = hub.KerasLayer(
        f'https://tfhub.dev/tensorflow/movinet/{model_name}/base/kinetics-600/classification/3')
    encoder.call = tf.function(encoder.call, experimental_compile=True)
    tf_params = []
    for item in encoder.variables:
        tf_params.append((item.name, item.numpy()))
    return tf_params


def get_movinet_base_state_dict(model_name: str) -> Dict:
    tf_params = _get_tf_model_params(model_name)
    torch_params = {_key_translate_base(tf_key_name, model_name): _weight_translate(tf_value, tf_key_name) for i, (tf_key_name, tf_value)
                    in enumerate(tf_params)}
    return torch_params
