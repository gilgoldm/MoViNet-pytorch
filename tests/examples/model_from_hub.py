import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
from six.moves import urllib
from io import BytesIO
from PIL import Image


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
T = 1
bs = 1

hub_url = f"https://tfhub.dev/tensorflow/movinet/{model_id}/stream/kinetics-600/classification/"
model_tf = MoViNetKaras(hub_url)
init_states_fn = model_tf.encoder.resolved_object.signatures['init_states']
dummy_input = tf.random.normal(shape=[bs, T, H, W, 3])
init_states = init_states_fn(tf.shape(dummy_input))
image_url = 'https://upload.wikimedia.org/wikipedia/commons/8/84/Ski_Famille_-_Family_Ski_Holidays.jpg'
with urllib.request.urlopen(image_url) as f:
    image = Image.open(BytesIO(f.read())).resize((H, W))
video = tf.reshape(np.array(image), [1, 1, H, W, 3])
video = tf.cast(video, tf.float32) / 255.
video = tf.concat([video, video], axis=1)

frames = tf.split(video, video.shape[1], axis=1)
states = init_states
predictions = []
for frame in frames:
    output, states = model_tf(frame, states)
    predictions.append(output)
# The video classification will simply be the last output of the model.
output_tf = predictions[-1]


