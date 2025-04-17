import unittest

import torch

from weights import export_base
from movinets import MoViNet
from movinets.config import _C


class TestTorchHubAndExport(unittest.TestCase):

    def test_torch_hub_weights_equal_direct_export_a0(self):
        model_name = 'a0'
        params_from_tf = export_base.get_movinet_base_state_dict(model_name)
        model_from_tf = MoViNet(_C.MODEL.MoViNetA0, causal=False, num_classes=600, tf_like=True)
        model_from_tf.load_state_dict(params_from_tf)
        params_from_tf = model_from_tf.state_dict()
        model_from_torch_hub = MoViNet(_C.MODEL.MoViNetA0, causal=False, tf_like=True, pretrained=True)
        params_from_torch = model_from_torch_hub.state_dict()
        keys_tf = set(params_from_tf.keys())
        keys_torch = set(params_from_torch.keys())
        self.assertEqual(keys_tf, keys_torch)
        for key_tf, value_tf in params_from_tf.items():
            value_torch = params_from_torch[key_tf]
            torch.testing.assert_close(value_tf, value_torch)

    def test_torch_hub_weights_equal_direct_export_a3(self):
        model_name = 'a3'
        params_from_tf = export_base.get_movinet_base_state_dict(model_name)
        model_from_tf = MoViNet(_C.MODEL.MoViNetA3, causal=False, num_classes=600, tf_like=True)
        model_from_tf.load_state_dict(params_from_tf)
        params_from_tf = model_from_tf.state_dict()
        model_from_torch_hub = MoViNet(_C.MODEL.MoViNetA3, causal=False, tf_like=True, pretrained=True)
        params_from_torch = model_from_torch_hub.state_dict()
        keys_tf = set(params_from_tf.keys())
        keys_torch = set(params_from_torch.keys())
        self.assertEqual(keys_tf, keys_torch)
        for key_tf, value_tf in params_from_tf.items():
            value_torch = params_from_torch[key_tf]
            torch.testing.assert_close(value_tf, value_torch)


if __name__ == '__main__':
    unittest.main()
