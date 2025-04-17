import unittest
from io import BytesIO

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import torch
from PIL import Image
from einops import rearrange
from six.moves import urllib

from movinets import MoViNet
from movinets.config import _C

movinets = [_C.MODEL.MoViNetA0,
            _C.MODEL.MoViNetA1,
            _C.MODEL.MoViNetA2,
            _C.MODEL.MoViNetA3,
            _C.MODEL.MoViNetA4,
            _C.MODEL.MoViNetA5]

tf.config.optimizer.set_experimental_options({"layout_optimizer": False})


class MoViNetKaras(tf.keras.Model):
    def __init__(self, hub_url):
        super(MoViNetKaras, self).__init__()
        encoder = hub.KerasLayer(hub_url, trainable=False)
        encoder.call = tf.function(encoder.call, experimental_compile=True)
        self.encoder = encoder

    def call(self, X):
        return self.encoder({'image': X})


def create_base_hub_model(model_id) -> tf.keras.Model:
    hub_url = f"https://tfhub.dev/tensorflow/movinet/{model_id}/base/kinetics-600/classification/"
    model = MoViNetKaras(hub_url)
    model.build([1, 1, 1, 1, 3])
    return model


class TestTorchTfInference(unittest.TestCase):

    def testBasePretrainedModels(self):
        image_url = 'https://upload.wikimedia.org/wikipedia/commons/8/84/Ski_Famille_-_Family_Ski_Holidays.jpg'
        image_height_l = [172, 172, 224, 256, 290, 320]
        image_width_l = [172, 172, 224, 256, 290, 320]

        for i in range(6):
            image_width = image_width_l[i]
            image_height = image_height_l[i]
            with urllib.request.urlopen(image_url) as f:
                image = Image.open(BytesIO(f.read())).resize((image_height, image_width))
            video = tf.reshape(np.array(image), [1, 1, image_height, image_width, 3])
            video = tf.cast(video, tf.float32) / 255.
            video = tf.concat([video, video / 2], axis=1)
            video_2 = rearrange(torch.from_numpy(video.numpy()), "b t h w c-> b c t h w")

            model_tf = create_base_hub_model(f'a{i}')
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
