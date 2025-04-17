import unittest
from io import BytesIO

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
from official.projects.movinet.modeling import movinet
from official.projects.movinet.modeling import movinet_model
from six.moves import urllib

tf.config.optimizer.set_experimental_options({"layout_optimizer": False})


class MoViNetKaras(tf.keras.Model):
    def __init__(self, hub_url):
        super(MoViNetKaras, self).__init__()
        self.encoder = hub.KerasLayer(hub_url, trainable=False)

    def call(self, X):
        return self.encoder({'image': X})


def create_base_hub_model(model_id, H, W) -> tf.keras.Model:
    hub_url = f"https://tfhub.dev/tensorflow/movinet/{model_id}/base/kinetics-600/classification/"
    model = MoViNetKaras(hub_url)
    model.build([1, 1, 1, 1, 3])
    return model


def create_base_source_model(model_id, H, W) -> movinet.Movinet:
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
    dummy_input = tf.ones([bs, T, H, W, C])

    # [Optional] Build the model and load a pretrained checkpoint
    model.build(dummy_input.shape)
    checkpoint_dir = f'movinet_{model_id}_base'
    checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
    checkpoint = tf.train.Checkpoint(model=model)
    status = checkpoint.restore(checkpoint_path)
    status.assert_existing_objects_matched()
    return model


class MyTestCase(unittest.TestCase):

    def test_base_hub_equal_source(self):
        try:
            model_id = 'a0'
            H = W = 172
            tf.keras.backend.clear_session()
            model_hub = create_base_hub_model(model_id, H=H, W=W)
            image_url = 'https://upload.wikimedia.org/wikipedia/commons/8/84/Ski_Famille_-_Family_Ski_Holidays.jpg'
            with urllib.request.urlopen(image_url) as f:
                image = Image.open(BytesIO(f.read())).resize((H, W))
            X = tf.reshape(np.array(image), [1, 1, H, W, 3])
            X = tf.cast(X, tf.float32) / 255
            y_hub = model_hub(X)
            print(y_hub[0][0:5])
            model_source = create_base_source_model(model_id, H=H, W=W)
            y_source = model_source({'image': X})
            print(y_source[0][0:5])
            tf.debugging.assert_near(y_source, y_hub, atol=1e-1, rtol=1e-3)
        finally:
            del model_hub
            del model_source


if __name__ == '__main__':
    unittest.main()
