import sys
import unittest
from movinets import MoViNet
from movinets.config import _C
from io import BytesIO
import tensorflow as tf
import numpy as np
from six.moves import urllib
from PIL import Image
from einops import rearrange
import torch
import tensorflow as tf
import tensorflow_hub as hub

movinets = [_C.MODEL.MoViNetA0,
            _C.MODEL.MoViNetA1,
            _C.MODEL.MoViNetA2,
            _C.MODEL.MoViNetA3,
            _C.MODEL.MoViNetA4,
            _C.MODEL.MoViNetA5, ]

tf.config.optimizer.set_experimental_options({"layout_optimizer": False})


class MoViNetKaras(tf.keras.Model):
    def __init__(self, hub_url):
        super(MoViNetKaras, self).__init__()
        self.encoder = hub.KerasLayer(hub_url)

    def call(self, X, states=None):
        if states:
            return self.encoder({**states, 'image': X})
        else:
            return self.encoder({'image': X})


class TestPretrainedModels(unittest.TestCase):
    def testBasePretrainedModels(self):
        image_url = 'https://upload.wikimedia.org/wikipedia/commons/8/84/Ski_Famille_-_Family_Ski_Holidays.jpg'
        image_height_l = [172, 172, 224, 256, 290, 320]
        image_width_l = [172, 172, 224, 256, 290, 320]

        # f = open('/dev/null', 'w')
        # sys.stderr = f

        for i in range(6):
            image_width = image_width_l[i]
            image_height = image_height_l[i]
            with urllib.request.urlopen(image_url) as f:
                image = Image.open(BytesIO(f.read())).resize((image_height, image_width))
            video = tf.reshape(np.array(image), [1, 1, image_height, image_width, 3])
            video = tf.cast(video, tf.float32) / 255.
            video = tf.concat([video, video / 2], axis=1)
            video_2 = rearrange(torch.from_numpy(video.numpy()), "b t h w c-> b c t h w")
            hub_url = f"https://tfhub.dev/tensorflow/movinet/a{i}/base/kinetics-600/classification/3"
            model_tf = MoViNetKaras(hub_url)
            output_tf = model_tf(video)
            del model_tf

            model = MoViNet(movinets[i], causal=False, pretrained=True)
            model.eval()
            with torch.no_grad():
                model.clean_activation_buffers()
                output = model(video_2)
            del model
            torch.testing.assert_close(output.detach(), torch.tensor(output_tf.numpy()), atol=1e-2, rtol=1e-2)
            # self.assertTrue(np.allclose(output.detach().numpy(), output_tf.numpy(), rtol=1e-06, atol=1e-4))

    def testStreamPretrainedModels(self):
        image_url = 'https://upload.wikimedia.org/wikipedia/commons/8/84/Ski_Famille_-_Family_Ski_Holidays.jpg'
        image_height_l = [172, 172, 224, 256, 290, 320]
        image_width_l = [172, 172, 224, 256, 290, 320]

        # f = open('/dev/null', 'w')
        # sys.stderr = f

        for i in range(3):
            image_width = image_width_l[i]
            image_height = image_height_l[i]
            with urllib.request.urlopen(image_url) as f:
                image = Image.open(BytesIO(f.read())).resize((image_height, image_width))
            video = tf.reshape(np.array(image), [1, 1, image_height, image_width, 3])
            video = tf.cast(video, tf.float32) / 255.
            video = tf.concat([video, video / 2, video / 3], axis=1)
            video_2 = rearrange(torch.from_numpy(video.numpy()), "b t h w c-> b c t h w")
            hub_url = f"https://tfhub.dev/tensorflow/movinet/a{i}/stream/kinetics-600/classification/3"
            model_tf = MoViNetKaras(hub_url)
            init_states_fn = model_tf.encoder.resolved_object.signatures['init_states']
            init_states = init_states_fn(tf.shape(video))
            # Run the model prediction by looping over each frame.
            frames = tf.split(video, video.shape[1], axis=1)
            states = init_states
            predictions = []
            for frame in frames:
                output, states = model_tf(frame, states)
                predictions.append(output)
            # The video classification will simply be the last output of the model.
            output_tf = predictions[-1]
            del model_tf

            model = MoViNet(movinets[i], causal=True, pretrained=True)
            model.eval()
            with torch.no_grad():
                model.clean_activation_buffers()
                output = model(video_2)
                model.clean_activation_buffers()
                _ = model(video_2[:, :, :1])
                _ = model(video_2[:, :, 1:2])
                output_2 = model(video_2[:, :, 2:3])
            del model
            torch.testing.assert_close(output.detach(), output_2.detach(), rtol=1e-6, atol=1e-4)
            torch.testing.assert_close(torch.tensor(output_tf.numpy()), output.detach(), rtol=1e-1, atol=1e-1)


if __name__ == '__main__':
    unittest.main()
