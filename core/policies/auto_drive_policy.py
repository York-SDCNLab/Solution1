import cv2
import numpy as np
from keras import backend as K
from keras.optimizers import Adam

from core.models.tf.cnn_auto_drive import get_model

IMG_HT, IMG_WIDTH, IMG_CH = 66, 200, 3

class AutoDrivePolicy:
    def __init__(self, weights_dir):
        opt = Adam(1e-4, decay=0.0)
        self.model = get_model(opt)
        self.model.load_weights(weights_dir)

        self.vel = 0.1

    def __call__(self, obs):
        metrics = {}
        image = self.preprocess(obs["image"])
        model_input = image[None, ...].astype(np.float16)
        #model_input = K.cast_to_floatx(model_input)

        steering = self.model.predict(model_input)[0, 0]
        vel = self.vel * np.cos(steering)

        return np.array([vel, steering]), metrics

    def preprocess(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image[40:-20, :, :]
        image = cv2.resize(image, (IMG_WIDTH, IMG_HT), cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

        return image