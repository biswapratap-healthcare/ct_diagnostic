import numpy as np
from collections import OrderedDict
from tensorflow.python.keras.models import model_from_json

from common import LABEL_GROUND_GLASS_OPACITY, LABEL_CONSOLIDATION, LABEL_SUB_PLEURAL_BAND, LABEL_FIBROSIS, \
    LABEL_PLEURAL_EFFUSION, LABEL_PNEUMOTHORAX, LABEL_NORMAL, LABEL_UNKNOWN

label_map = OrderedDict()
label_map[LABEL_GROUND_GLASS_OPACITY] = [1, 0, 0, 0, 0, 0, 0, 0]
label_map[LABEL_CONSOLIDATION]        = [0, 1, 0, 0, 0, 0, 0, 0]
label_map[LABEL_SUB_PLEURAL_BAND]     = [0, 0, 1, 0, 0, 0, 0, 0]
label_map[LABEL_FIBROSIS]             = [0, 0, 0, 1, 0, 0, 0, 0]
label_map[LABEL_PLEURAL_EFFUSION]     = [0, 0, 0, 0, 1, 0, 0, 0]
label_map[LABEL_PNEUMOTHORAX]         = [0, 0, 0, 0, 0, 1, 0, 0]
label_map[LABEL_NORMAL]               = [0, 0, 0, 0, 0, 0, 1, 0]
label_map[LABEL_UNKNOWN]              = [0, 0, 0, 0, 0, 0, 0, 1]

label_keys = list(label_map.keys())


class DeepClassification:

    def __init__(self):
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.loaded_model = model_from_json(loaded_model_json)
        self.loaded_model.load_weights("model.h5")

    @staticmethod
    def get_predicted_labels(r):
        result = list()
        r = r[0]
        for idx, p in enumerate(r):
            p = p * 100
            p = np.round(p, 2)
            result.append((label_keys[idx], p))
        result = sorted(result, key=lambda x: x[1], reverse=True)[0:2]
        return result

    def predict(self, x):
        r = self.loaded_model.predict(x)
        result = self.get_predicted_labels(r)
        return result
