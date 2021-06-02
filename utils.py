import os

import pydicom
import numpy as np
from tqdm import tqdm
from threading import Lock
from numpy import ones, vstack
from numpy.linalg import lstsq

from common import MAX_HEIGHT, MAX_WIDTH
from rest_client import update_in_progress_state, update_progress_percent
from segmentation import get_segmented_lungs


def to_dictionary(ds):
    output = dict()
    for elem in ds:
        if elem.VR != 'SQ':
            output[elem.keyword] = elem.value
        else:
            output[elem.keyword] = [to_dictionary(item) for item in elem]
    return output


def dictify(ds):
    output = to_dictionary(ds)
    return output


def get_instance_files(instance_files):
    ct_series = dict()
    for instance_file in tqdm(instance_files):
        try:
            meta_data_dicom = pydicom.dcmread(instance_file)
            ds = dictify(meta_data_dicom)
            modality = ds['Modality']
            if modality == 'CT':
                series_instance_id = str(ds['SeriesInstanceUID'])
                series_desc = str(ds['SeriesDescription'])
                entry = ct_series.get(series_instance_id + '_' + series_desc)
                if not entry:
                    ct_series[series_instance_id + '_' + series_desc] = []
                    ct_series.get(series_instance_id + '_' + series_desc).append(instance_file)
                else:
                    ct_series.get(series_instance_id + '_' + series_desc).append(instance_file)
        except:
            continue
    insts = None
    max_len = 0
    for k, v in ct_series.items():
        if 'HRCT THIN' in k:
            insts = v
            break
        if len(v) > max_len:
            max_len = len(v)
            insts = v
    return insts


def voxels_to_pixels(voxel_array):
    pix_arr = np.array(voxel_array)
    pix_arr = pix_arr.astype(np.uint32)
    segmented_pixel_array = get_segmented_lungs(pix_arr.copy(), plot=False)

    xmax, xmin = segmented_pixel_array.max(), segmented_pixel_array.min()
    d = xmax - xmin
    if d == 0:
        segmented_pixel_array = np.zeros((MAX_HEIGHT, MAX_WIDTH), dtype=np.uint8)
    else:
        segmented_pixel_array = (segmented_pixel_array - xmin) / d
    segmented_pixel_array = segmented_pixel_array * 255.0
    segmented_pixel_array = segmented_pixel_array.astype(np.uint8)

    xmax, xmin = pix_arr.max(), pix_arr.min()
    d = xmax - xmin
    if d == 0:
        pix_arr = np.zeros((MAX_HEIGHT, MAX_WIDTH), dtype=np.uint8)
    else:
        pix_arr = (pix_arr - xmin) / d
    pix_arr = pix_arr * 255.0
    pix_arr = pix_arr.astype(np.uint8)

    return pix_arr, segmented_pixel_array


def is_left_lung(y, min_y, max_y):
    y_mid = ((max_y - min_y) * 45.0) / 100.0
    if y < y_mid:
        return True
    else:
        return False


def is_left_superior_lobe(x, y, min_x, max_x, min_y, max_y):
    points = list()
    y_mid = ((max_y - min_y) * 45.0) / 100.0
    x_1 = ((max_x - min_x) * 40.0) / 100.0
    y_1 = min_y
    x_2 = ((max_x - min_x) * 55.0) / 100.0
    y_2 = y_mid
    points.append((x_1, y_1))
    points.append((x_2, y_2))
    x_coords, y_coords = zip(*points)
    A = vstack([x_coords, ones(len(x_coords))]).T
    m, c = lstsq(A, y_coords, rcond=None)[0]
    y_pred = m * x + c
    if y > y_pred:
        return True
    else:
        return False


def is_left_inferior_lobe(x, y, min_x, max_x, min_y, max_y):
    points = list()
    y_mid = ((max_y - min_y) * 45.0) / 100.0
    x_1 = ((max_x - min_x) * 50.0) / 100.0
    y_1 = min_y
    x_2 = ((max_x - min_x) * 90.0) / 100.0
    y_2 = y_mid
    points.append((x_1, y_1))
    points.append((x_2, y_2))
    x_coords, y_coords = zip(*points)
    A = vstack([x_coords, ones(len(x_coords))]).T
    m, c = lstsq(A, y_coords, rcond=None)[0]
    y_pred = m * x + c
    if y < y_pred:
        return True
    else:
        return False


def is_right_superior_lobe(x, y, min_x, max_x, min_y, max_y):
    points = list()
    y_mid = ((max_y - min_y) * 45.0) / 100.0
    x_1 = ((max_x - min_x) * 40.0) / 100.0
    y_1 = max_y
    x_2 = ((max_x - min_x) * 90.0) / 100.0
    y_2 = y_mid
    points.append((x_1, y_1))
    points.append((x_2, y_2))
    x_coords, y_coords = zip(*points)
    A = vstack([x_coords, ones(len(x_coords))]).T
    m, c = lstsq(A, y_coords, rcond=None)[0]
    y_pred = m * x + c
    if y > y_pred:
        return True
    else:
        return False


def get_pixel_lobe_classification(x, y, min_x, max_x, min_y, max_y):
    if is_left_lung(y, min_y, max_y):
        if is_left_superior_lobe(x, y, min_x, max_x, min_y, max_y):
            return 'lsl'
        else:
            if is_left_inferior_lobe(x, y, min_x, max_x, min_y, max_y):
                return 'lil'
            else:
                return 'lml'
    else:
        if is_right_superior_lobe(x, y, min_x, max_x, min_y, max_y):
            return 'rsl'
        else:
            return 'ril'


def create_json(study_instance_id,
                scores,
                ggo_count,
                con_count,
                sub_count,
                fib_count,
                ple_count,
                abnormal_slice_count,
                total_slice_count):
    final_json = dict()
    final_json['StudyInstanceUID'] = study_instance_id
    final_json['num_of_positive_slices'] = str(abnormal_slice_count)
    final_json['num_of_lung_slices'] = str(total_slice_count)
    final_json['ratio_of_positive_slices'] = str(round(float(float(abnormal_slice_count) * 100.0) / float(total_slice_count), 2))
    if ggo_count > 0:
        final_json['focal_ggo_detections'] = "True"
    else:
        final_json['focal_ggo_detections'] = "False"
    total_count = ggo_count + con_count + sub_count + fib_count + ple_count
    final_json['ggo_ratio'] = str(round(float(float(ggo_count) * 100.0) / float(total_count), 2))
    final_json['consolidation_ratio'] = str(round(float(float(con_count) * 100.0) / float(total_count), 2))

    if scores[5] > 5:
        result_type = 'Abnormal'
    else:
        result_type = 'Normal'

    if scores[5] <= 5:
        diagnosis = 'NA'
    elif 5 < scores[5] <= 10:
        diagnosis = 'Mild'
    elif 10 < scores[5] <= 15:
        diagnosis = 'Moderate'
    else:
        diagnosis = 'High'

    diagnosis_details = str(scores[5]) + ', ' + str(scores[0]) + \
                        ', ' + str(scores[1]) + ', ' + str(scores[2]) + \
                        ', ' + str(scores[3]) + ', ' + str(scores[4])

    final_json['ResultType'] = result_type
    final_json['GlobalDiagnosis'] = diagnosis
    final_json['GlobalDiagnosisDetails'] = diagnosis_details

    final_json['co_rads_score'] = str(scores[5])
    final_json['right_superior_lobe_percentage_affected'] = str(scores[0])
    final_json['right_middle_lobe_percentage_affected'] = str(scores[1])
    final_json['right_inferior_lobe_percentage_affected'] = str(scores[2])
    final_json['left_superior_lobe_percentage_affected'] = str(scores[3])
    final_json['left_inferior_lobe_percentage_affected'] = str(scores[4])
    final_json['3DDiagnosisImage'] = ''
    return final_json


def get_score(percent):
    if percent < 5.0:
        return 1
    elif 5.0 <= percent < 25.0:
        return 2
    elif 25.0 <= percent < 50.0:
        return 3
    elif 50.0 <= percent < 75.0:
        return 4
    elif 75.0 <= percent <= 100.0:
        return 5
    else:
        return 0


def get_25_score(ct_slice_objects, sid):
    lsl_normal_count = 0.0
    lsl_abnormal_count = 0.0
    lml_normal_count = 0.0
    lml_abnormal_count = 0.0
    lil_normal_count = 0.0
    lil_abnormal_count = 0.0
    rsl_normal_count = 0.0
    rsl_abnormal_count = 0.0
    ril_normal_count = 0.0
    ril_abnormal_count = 0.0

    for ct_slice_object in ct_slice_objects:
        lsl_normal_count += ct_slice_object.lsl_normal_count
        lsl_abnormal_count += ct_slice_object.lsl_abnormal_count
        lml_normal_count += ct_slice_object.lml_normal_count
        lml_abnormal_count += ct_slice_object.lml_abnormal_count
        lil_normal_count += ct_slice_object.lil_normal_count
        lil_abnormal_count += ct_slice_object.lil_abnormal_count
        rsl_normal_count += ct_slice_object.rsl_normal_count
        rsl_abnormal_count += ct_slice_object.rsl_abnormal_count
        ril_normal_count += ct_slice_object.ril_normal_count
        ril_abnormal_count += ct_slice_object.ril_abnormal_count

    total_score = 0.0

    try:
        lsl_percent = round((float(lsl_abnormal_count) * 100.0) /
                            (float(lsl_normal_count) + float(lsl_abnormal_count)), 2)
    except ZeroDivisionError as e:
        lsl_percent = 0.00
    total_score += get_score(lsl_percent)
    try:
        lml_percent = round((float(lml_abnormal_count) * 100.0) /
                            (float(lml_normal_count) + float(lml_abnormal_count)), 2)
    except ZeroDivisionError as e:
        lml_percent = 0.00
    total_score += get_score(lml_percent)
    try:
        lil_percent = round((float(lil_abnormal_count) * 100.0) /
                            (float(lil_normal_count) + float(lil_abnormal_count)), 2)
    except ZeroDivisionError as e:
        lil_percent = 0.00
    total_score += get_score(lil_percent)
    try:
        rsl_percent = round((float(rsl_abnormal_count) * 100.0) /
                            (float(rsl_normal_count) + float(rsl_abnormal_count)), 2)
    except ZeroDivisionError as e:
        rsl_percent = 0.00
    total_score += get_score(rsl_percent)
    try:
        ril_percent = round((float(ril_abnormal_count) * 100.0) /
                            (float(ril_normal_count) + float(ril_abnormal_count)), 2)
    except ZeroDivisionError as e:
        ril_percent = 0.00
    total_score += get_score(ril_percent)

    return lsl_percent, lml_percent, lil_percent, rsl_percent, ril_percent, total_score


mutex = Lock()
last_sent_mutex = Lock()


def read_last_sent_progress(job_id):
    last_sent_mutex.acquire()
    try:
        if os.path.exists(job_id + '/last_sent_progress.txt') is False:
            return "Invalid Job ID"
        with open(job_id + '/last_sent_progress.txt', "r") as f:
            percent = f.read()
        return percent
    finally:
        last_sent_mutex.release()


def read_progress(job_id):
    mutex.acquire()
    try:
        if os.path.exists(job_id + '/progress.txt') is False:
            return "Invalid Job ID"
        with open(job_id + '/progress.txt', "r") as f:
            percent = f.read()
        return percent
    finally:
        mutex.release()


def write_last_sent_progress(job_id, percent):
    last_sent_mutex.acquire()
    try:
        with open(job_id + '/last_sent_progress.txt', "w") as f:
            f.write(percent)
    finally:
        last_sent_mutex.release()


def write_progress(job_id, percent):
    mutex.acquire()
    try:
        with open(job_id + '/progress.txt', "w") as f:
            f.write(percent)
    finally:
        if percent == "5":
            update_in_progress_state(job_id, 'TRUE')
        if percent == "100":
            update_in_progress_state(job_id, 'FALSE')
        mutex.release()
