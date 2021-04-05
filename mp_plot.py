import copy
import math
import os

import pydicom
import numpy as np
import concurrent.futures

from scipy.spatial import distance
from scipy.spatial.qhull import Delaunay

from common import BOX_SIZE, LABEL_GROUND_GLASS_OPACITY, LABEL_CONSOLIDATION, LABEL_FIBROSIS
from datetime import timezone
import datetime


def get_type(type_map, point):
    ret = 0
    for key, value in type_map.items():
        dst = distance.euclidean(key, point)
        if dst < BOX_SIZE / 2:
            if value.lower() == LABEL_GROUND_GLASS_OPACITY.lower():
                ret = 3500
                break
            elif value.lower() == LABEL_CONSOLIDATION.lower():
                ret = 3400
                break
            elif value.lower() == LABEL_FIBROSIS.lower():
                ret = 3300
                break
            else:
                ret = 0
                break
    return ret


def worker_plot(inp):
    r = inp[0]
    clusters = inp[1]
    type_map = inp[2]
    ct_ggo_dir = inp[3]
    ct_con_dir = inp[4]
    ct_fib_dir = inp[5]
    meta_data_dicom = r[9]
    meta_data_dicom.pixel_array.fill(0)

    meta_data_dicom_ggo = copy.deepcopy(meta_data_dicom)
    meta_data_dicom_ggo.pixel_array.fill(0)

    meta_data_dicom_con = copy.deepcopy(meta_data_dicom)
    meta_data_dicom_con.pixel_array.fill(0)

    meta_data_dicom_fib = copy.deepcopy(meta_data_dicom)
    meta_data_dicom_fib.pixel_array.fill(0)

    z = int(math.floor(float(meta_data_dicom.ImagePositionPatient[2])))

    for key, new_value in clusters.items():
        try:
            c_x = [c[0] for c in new_value]
            c_y = [c[1] for c in new_value]
            min_x = min(c_x) - int((BOX_SIZE / 2))
            max_x = max(c_x) + int((BOX_SIZE / 2))
            min_y = min(c_y) - int((BOX_SIZE / 2))
            max_y = max(c_y) + int((BOX_SIZE / 2))
            hull = Delaunay(np.array(new_value))
        except Exception as e:
            continue

        for x in range(min_x, max_x, 1):
            for y in range(min_y, max_y, 1):
                point = [x, y, z]
                if hull.find_simplex(np.array(point)) >= 0:
                    t = get_type(type_map, point)
                    if t == 3500:
                        meta_data_dicom_ggo.pixel_array[x][y] = t
                        meta_data_dicom_con.pixel_array[x][y] = 0
                        meta_data_dicom_fib.pixel_array[x][y] = 0
                    elif t == 3400:
                        print("Found Consolidation")
                        meta_data_dicom_ggo.pixel_array[x][y] = 0
                        meta_data_dicom_con.pixel_array[x][y] = t
                        meta_data_dicom_fib.pixel_array[x][y] = 0
                    elif t == 3300:
                        print("Found Fibrosis")
                        meta_data_dicom_ggo.pixel_array[x][y] = 0
                        meta_data_dicom_con.pixel_array[x][y] = 0
                        meta_data_dicom_fib.pixel_array[x][y] = t
                    else:
                        meta_data_dicom_ggo.pixel_array[x][y] = t
                        meta_data_dicom_con.pixel_array[x][y] = t
                        meta_data_dicom_fib.pixel_array[x][y] = t

    meta_data_dicom_ggo.PixelData = meta_data_dicom_ggo.pixel_array.tobytes()
    dt = datetime.datetime.now(timezone.utc)
    utc_time = dt.replace(tzinfo=timezone.utc)
    utc_timestamp = utc_time.timestamp()
    pydicom.dcmwrite(ct_ggo_dir + '/' + str(utc_timestamp) + '_' + str(os.getpid()) + '_ggo.dcm', meta_data_dicom_ggo)

    meta_data_dicom_con.PixelData = meta_data_dicom_con.pixel_array.tobytes()
    dt = datetime.datetime.now(timezone.utc)
    utc_time = dt.replace(tzinfo=timezone.utc)
    utc_timestamp = utc_time.timestamp()
    pydicom.dcmwrite(ct_con_dir + '/' + str(utc_timestamp) + '_' + str(os.getpid()) + '_con.dcm', meta_data_dicom_con)

    meta_data_dicom_fib.PixelData = meta_data_dicom_fib.pixel_array.tobytes()
    dt = datetime.datetime.now(timezone.utc)
    utc_time = dt.replace(tzinfo=timezone.utc)
    utc_timestamp = utc_time.timestamp()
    pydicom.dcmwrite(ct_fib_dir + '/' + str(utc_timestamp) + '_' + str(os.getpid()) + '_fib.dcm', meta_data_dicom_fib)


def mp_plot(rs, clusters, type_map, ct_ggo_dir, ct_con_dir, ct_fib_dir):
    inps = list()
    for r in rs:
        inps.append((r, clusters, type_map, ct_ggo_dir, ct_con_dir, ct_fib_dir))
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(worker_plot, inps)


def mp_plot_2(rs, clusters, type_map, ct_ggo_dir, ct_con_dir, ct_fib_dir):
    for r in rs:
        worker_plot((r, clusters, type_map, ct_ggo_dir, ct_con_dir, ct_fib_dir))
