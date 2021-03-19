import pydicom
import numpy as np
from tqdm import tqdm
from numpy import ones, vstack
from numpy.linalg import lstsq

from common import MAX_HEIGHT, MAX_WIDTH
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
