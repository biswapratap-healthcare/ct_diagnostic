import time

import cv2
import numpy as np

from box import Box
from utils import voxels_to_pixels, get_pixel_lobe_classification
from common import MAX_HEIGHT, BOX_SIZE, MAX_WIDTH, LABEL_NORMAL, LABEL_GROUND_GLASS_OPACITY, \
    LABEL_CONSOLIDATION, LABEL_SUB_PLEURAL_BAND, LABEL_FIBROSIS, LABEL_PLEURAL_EFFUSION, LABEL_PNEUMOTHORAX


class CTSlice:

    def __init__(self, meta_data_dicom):

        self.lsl_normal_count = 0.0
        self.lsl_abnormal_count = 0.0
        self.lml_normal_count = 0.0
        self.lml_abnormal_count = 0.0
        self.lil_normal_count = 0.0
        self.lil_abnormal_count = 0.0
        self.rsl_normal_count = 0.0
        self.rsl_abnormal_count = 0.0
        self.ril_normal_count = 0.0
        self.ril_abnormal_count = 0.0

        self.affected_pixels = list()

        self.__sop_instance_uid = meta_data_dicom.SOPInstanceUID
        self.__series_instance_uid = meta_data_dicom.SeriesInstanceUID
        self.__study_instance_uid = meta_data_dicom.StudyInstanceUID
        self.__image_position_patient = meta_data_dicom.ImagePositionPatient

        self.__pos_x = float(self.__image_position_patient[0])
        self.__pos_y = float(self.__image_position_patient[1])
        self.__pos_z = float(self.__image_position_patient[2])

        self.__voxel_array = meta_data_dicom.pixel_array
        self.__pixel_array, self.__segmented_pixel_array = self.__voxels_to_pixels__()

        x = list()
        y = list()
        z = list()

        for i in range(0, MAX_HEIGHT, 1):
            for j in range(0, MAX_WIDTH, 1):
                if self.__segmented_pixel_array[i][j] != 0:
                    x.append(self.__pos_x + i)
                    y.append(self.__pos_y + j)
                    z.append(self.__pos_z)

        if len(x) > 0:
            self.__min_x = min(x)
            self.__max_x = max(x)
        if len(y) > 0:
            self.__min_y = min(y)
            self.__max_y = max(y)
        if len(z) > 0:
            self.__min_z = min(z)
            self.__max_z = max(z)

        self.all_zeros = not self.__segmented_pixel_array.any()
        if not self.all_zeros:
            self.__boxes = self.__pixels_to_boxes__()
            if len(self.__boxes) == 0:
                self.__is_valid = False
            else:
                self.__is_valid = True
                for box in self.__boxes:
                    labels, confidences = box.get_label()
                    if labels[0] != LABEL_NORMAL and int(confidences[0]) == int(100.00):
                        lung_pixels = box.get_lung_pixels()
                        for lung_pixel in lung_pixels:
                            x_idx = lung_pixel[0]
                            y_idx = lung_pixel[1]
                            z_idx = self.__pos_z
                            self.affected_pixels.append((labels[0],
                                                         confidences[0],
                                                         [x_idx, y_idx, z_idx]))
        else:
            self.__is_valid = False

    def is_valid(self):
        return self.__is_valid

    def get_image_position_patient(self):
        return self.__image_position_patient

    def __voxels_to_pixels__(self):
        pix_arr = np.array(self.__voxel_array)
        return voxels_to_pixels(pix_arr)

    def __pixels_to_boxes__(self):
        boxes = list()

        for x in range(0, MAX_HEIGHT, BOX_SIZE):
            for y in range(0, MAX_WIDTH, BOX_SIZE):

                count = 0.0
                total_count = 0.0
                box_zero = np.zeros((BOX_SIZE, BOX_SIZE, 1), dtype=np.uint8)
                box = Box(top_left_coord=(x, y),
                          top_right_coord=(x, y + BOX_SIZE),
                          bottom_right_coord=(x + BOX_SIZE, y + BOX_SIZE),
                          bottom_left_coord=(x + BOX_SIZE, y),
                          pixel_array=box_zero)
                lobes = list()

                for ix in range(0, BOX_SIZE, 1):
                    for iy in range(0, BOX_SIZE, 1):

                        xx = self.__pos_x + x + ix
                        yy = self.__pos_y + y + iy

                        if self.__segmented_pixel_array[x + ix, y + iy] == 255:
                            box.set_a_lung_pixel(x + ix, y + iy)
                            x_mod = xx - self.__min_x
                            y_mod = yy - self.__min_y
                            min_x_mod = 0
                            max_x_mod = self.__max_x - self.__min_x
                            min_y_mod = 0
                            max_y_mod = self.__max_y - self.__min_y
                            lobe = get_pixel_lobe_classification(x_mod,
                                                                 y_mod,
                                                                 min_x_mod,
                                                                 max_x_mod,
                                                                 min_y_mod,
                                                                 max_y_mod)
                            lobes.append(lobe)
                            count += 1.0

                        total_count += 1.0
                        box.set_pixel_array_value_at(x, y, ix, iy, self.__pixel_array[x + ix, y + iy])

                per = (count * 100.0)/total_count

                if per > 70:
                    box.set_feature_array()
                    box.set_label()
                    boxes.append(box)

                    if box.is_affected():
                        # file_path = 'abnormal_images/' + str(int(round(time.time() * 1000))) + '.png'
                        # cv2.imwrite(file_path, box.get_pixel_array())
                        for lobe in lobes:
                            if lobe == 'lsl':
                                self.lsl_abnormal_count += 0.00001
                            elif lobe == 'lil':
                                self.lil_abnormal_count += 0.00001
                            elif lobe == 'lml':
                                self.lml_abnormal_count += 0.00001
                            elif lobe == 'rsl':
                                self.rsl_abnormal_count += 0.00001
                            else:
                                self.ril_abnormal_count += 0.00001
                    else:
                        # file_path = 'normal_images/' + str(int(round(time.time() * 1000))) + '.png'
                        # cv2.imwrite(file_path, box.get_pixel_array())
                        for lobe in lobes:
                            if lobe == 'lsl':
                                self.lsl_normal_count += 0.00001
                            elif lobe == 'lil':
                                self.lil_normal_count += 0.00001
                            elif lobe == 'lml':
                                self.lml_normal_count += 0.00001
                            elif lobe == 'rsl':
                                self.rsl_normal_count += 0.00001
                            else:
                                self.ril_normal_count += 0.00001
                else:
                    for lobe in lobes:
                        if lobe == 'lsl':
                            self.lsl_normal_count += 0.00001
                        elif lobe == 'lil':
                            self.lil_normal_count += 0.00001
                        elif lobe == 'lml':
                            self.lml_normal_count += 0.00001
                        elif lobe == 'rsl':
                            self.rsl_normal_count += 0.00001
                        else:
                            self.ril_normal_count += 0.00001
        return boxes

    def get_box_label_distribution(self):
        num_of_ggo = 0
        num_of_con = 0
        num_of_subs = 0
        num_of_fib = 0
        num_of_ple = 0
        num_of_pne = 0
        num_of_nor = 0

        affected_points = list()

        for affected_pixel in self.affected_pixels:
            label = affected_pixel[0]
            confidence = affected_pixel[1]
            if int(confidence) == int(100.0):
                affected_points.append((affected_pixel[2], label))
            if label == LABEL_GROUND_GLASS_OPACITY and confidence > 90.0:
                num_of_ggo += 1
            elif label == LABEL_CONSOLIDATION and confidence > 90.0:
                num_of_con += 1
            elif label == LABEL_SUB_PLEURAL_BAND and confidence > 90.0:
                num_of_subs += 1
            elif label == LABEL_FIBROSIS and confidence > 90.0:
                num_of_fib += 1
            elif label == LABEL_PLEURAL_EFFUSION and confidence > 90.0:
                num_of_ple += 1
            elif label == LABEL_PNEUMOTHORAX and confidence > 90.0:
                num_of_pne += 1
            else:
                num_of_nor += 1

        return (num_of_ggo,
                num_of_con,
                num_of_subs,
                num_of_fib,
                num_of_ple,
                num_of_pne,
                num_of_nor,
                affected_points)
