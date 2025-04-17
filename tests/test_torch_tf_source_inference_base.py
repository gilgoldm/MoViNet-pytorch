import unittest
from io import BytesIO

import numpy as np
import tensorflow as tf
import torch
from PIL import Image
from einops import rearrange
from official.projects.movinet.modeling import movinet
from official.projects.movinet.modeling import movinet_model
from six.moves import urllib

from movinets import MoViNet
from movinets.config import _C

movinets = [_C.MODEL.MoViNetA0,
            _C.MODEL.MoViNetA1,
            _C.MODEL.MoViNetA2,
            _C.MODEL.MoViNetA3,
            _C.MODEL.MoViNetA4,
            _C.MODEL.MoViNetA5]


def create_base_source_model(model_id, res) -> movinet.Movinet:
    num_classes = 600
    C = 3
    T = 1
    bs = 1
    backbone = movinet.Movinet(
        model_id=model_id,
        causal=False,
        use_external_states=False,
    )
    backbone.trainable = False
    model = movinet_model.MovinetClassifier(
        backbone,
        num_classes=num_classes,
        output_states=False
    )
    # Create your example input here.
    # Refer to the paper for recommended input shapes.
    dummy_input = tf.ones([bs, T, res, res, C])

    # [Optional] Build the model and load a pretrained checkpoint
    model.build(dummy_input.shape)
    checkpoint_dir = f'movinet_{model_id}_base'
    checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
    checkpoint = tf.train.Checkpoint(model=model)
    status = checkpoint.restore(checkpoint_path)
    status.assert_existing_objects_matched()
    return model


class TestTorchTfInference(unittest.TestCase):

    def testBasePretrainedModels(self):
        image_url = 'https://upload.wikimedia.org/wikipedia/commons/8/84/Ski_Famille_-_Family_Ski_Holidays.jpg'
        res_list = [172, 172, 224, 256, 290, 320]

        for i in range(6):
            res = res_list[i]
            with urllib.request.urlopen(image_url) as f:
                image = Image.open(BytesIO(f.read())).resize((res, res))
            video = tf.reshape(np.array(image), [1, 1, res, res, 3])
            video = tf.cast(video, tf.float32) / 255.
            video = tf.concat([video, video / 2], axis=1)
            video_2 = rearrange(torch.from_numpy(video.numpy()), "b t h w c-> b c t h w")

            model_tf = create_base_source_model(model_id=f'a{i}', res=res)
            output_tf = model_tf(video)
            del model_tf

            model = MoViNet(movinets[i], causal=False, pretrained=True)
            model.eval()
            with torch.no_grad():
                model.clean_activation_buffers()
                output = model(video_2)
            del model
            self.assertTrue(np.allclose(output.detach().numpy(), output_tf.numpy(), atol=1e-2))


if __name__ == '__main__':
    unittest.main()
