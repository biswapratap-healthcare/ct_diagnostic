import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from tensorflow.keras import Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input


class FeatureExtractor:

    def __init__(self):
        self.__model = VGG16(include_top=False)
        self.__model.layers.pop()
        self.__model = Model(inputs=self.__model.inputs, outputs=self.__model.layers[-1].output)

    def get_features(self, image):
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        if self.__model:
            features = self.__model.predict(image)
            features = features.flatten().tolist()
        else:
            features = list()
        return features
