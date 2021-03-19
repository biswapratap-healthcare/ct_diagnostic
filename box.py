import numpy as np
from common import BOX_SIZE, LABEL_NORMAL
from feature import FeatureExtractor
from classifier import DeepClassification

dc = DeepClassification()
fe = FeatureExtractor()


class Box:
    def __init__(self,
                 top_left_coord,
                 top_right_coord,
                 bottom_right_coord,
                 bottom_left_coord,
                 pixel_array):
        self.__top_left_coord = top_left_coord
        self.__top_right_coord = top_right_coord
        self.__bottom_right_coord = bottom_right_coord
        self.__bottom_left_coord = bottom_left_coord
        self.__pixel_array = pixel_array
        self.__feature_array = None
        self.__labels = list()
        self.__confidences = list()

    def get_pixel_array(self):
        return self.__pixel_array

    def set_pixel_array_value_at(self, x, y, ix, iy, v):
        self.__pixel_array[ix, iy, 0] = v
        self.__pixel_array[ix, iy, 1] = x + ix
        self.__pixel_array[ix, iy, 2] = y + iy

    def set_feature_array(self):
        fa = fe.get_features(self.__pixel_array)
        fa = [np.array(fa)]
        fa = np.array(fa)
        self.__feature_array = fa

    def predict_box(self):
        x = self.__feature_array
        r = dc.predict(x)
        return r

    def set_label(self):
        r = self.predict_box()
        for ri in r:
            self.__labels.append(ri[0])
            self.__confidences.append(ri[1])

    def is_affected(self):
        if self.__labels[0] != LABEL_NORMAL and self.__confidences[0] == 100:
            return True
        else:
            return False

    def get_center(self):
        x = self.__top_left_coord[0] + BOX_SIZE / 2
        y = self.__top_left_coord[1] + BOX_SIZE / 2
        return x, y

    def get_four_corners(self):
        return [list(self.__top_left_coord),
                list(self.__top_right_coord),
                list(self.__bottom_right_coord),
                list(self.__bottom_left_coord)]

    def get_label(self):
        return self.__labels, self.__confidences
