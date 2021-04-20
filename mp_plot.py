import copy
import math
import os
import numpy as np

import pydicom
import concurrent.futures

from scipy.spatial import distance, Delaunay

from common import BOX_SIZE, LABEL_GROUND_GLASS_OPACITY, LABEL_CONSOLIDATION, LABEL_FIBROSIS
from datetime import timezone
import datetime

from utils import read_progress, write_progress


def get_type(type_map, point):
    dists = list()
    for key, value in type_map.items():
        if key[2] != point[2]:
            continue
        dst = distance.euclidean(key, point)
        dists.append((value, dst))
    dists.sort(key=lambda x: x[1])
    v = dists[0][0].lower()
    if v == LABEL_GROUND_GLASS_OPACITY.lower():
        ret = 3500
    elif v == LABEL_CONSOLIDATION.lower():
        ret = 3400
    elif v == LABEL_FIBROSIS.lower():
        ret = 3300
    else:
        ret = 0
    return ret


def worker_plot(inp):
    r = inp[0]
    type_map = inp[1]
    ct_ggo_dir = inp[2]
    ct_con_dir = inp[3]
    ct_fib_dir = inp[4]
    total_number_of_instances = inp[5]
    study_instance_id = inp[6]
    vtk_dir = inp[7]
    clusters = inp[8]

    meta_data_dicom = r[9]
    meta_data_dicom_vtk = copy.deepcopy(meta_data_dicom)
    meta_data_dicom.pixel_array.fill(0)

    meta_data_dicom_ggo = copy.deepcopy(meta_data_dicom)
    meta_data_dicom_ggo.pixel_array.fill(0)

    meta_data_dicom_con = copy.deepcopy(meta_data_dicom)
    meta_data_dicom_con.pixel_array.fill(0)

    meta_data_dicom_fib = copy.deepcopy(meta_data_dicom)
    meta_data_dicom_fib.pixel_array.fill(0)

    z = int(math.floor(float(meta_data_dicom.ImagePositionPatient[2])))

    # percent = read_progress(study_instance_id)
    # new_percent = str(round(float(float(percent) + 19.0 / float(total_number_of_instances)), 2))
    # write_progress(study_instance_id, new_percent)

    for key, new_value in clusters.items():
        c_x = [c[0] for c in new_value if c[2] == z]
        c_y = [c[1] for c in new_value if c[2] == z]
        if len(c_x) == 0 and len(c_y) == 0:
            continue
        min_x = min(c_x)
        max_x = max(c_x)
        min_y = min(c_y)
        max_y = max(c_y)
        hull = Delaunay(np.array(new_value))

        for x in range(min_x, max_x, 1):
            for y in range(min_y, max_y, 1):
                point = [x, y, z]
                if hull.find_simplex(np.array(point)) >= 0:
                    t = get_type(type_map, point)
                    if t == 3500:
                        meta_data_dicom_vtk.pixel_array[x][y] = 3000
                        meta_data_dicom_ggo.pixel_array[x][y] = t
                        meta_data_dicom_con.pixel_array[x][y] = 0
                        meta_data_dicom_fib.pixel_array[x][y] = 0
                    elif t == 3400:
                        print("Found Consolidation")
                        meta_data_dicom_ggo.pixel_array[x][y] = 0
                        meta_data_dicom_con.pixel_array[x][y] = t
                        meta_data_dicom_vtk.pixel_array[x][y] = 2000
                        meta_data_dicom_fib.pixel_array[x][y] = 0
                    elif t == 3300:
                        print("Found Fibrosis")
                        meta_data_dicom_ggo.pixel_array[x][y] = 0
                        meta_data_dicom_con.pixel_array[x][y] = 0
                        meta_data_dicom_fib.pixel_array[x][y] = t
                        meta_data_dicom_vtk.pixel_array[x][y] = 2000
                    else:
                        meta_data_dicom_ggo.pixel_array[x][y] = t
                        meta_data_dicom_con.pixel_array[x][y] = t
                        meta_data_dicom_fib.pixel_array[x][y] = t
                        meta_data_dicom_vtk.pixel_array[x][y] = t

    meta_data_dicom_vtk.PixelData = meta_data_dicom_vtk.pixel_array.tobytes()
    dt = datetime.datetime.now(timezone.utc)
    utc_time = dt.replace(tzinfo=timezone.utc)
    utc_timestamp = utc_time.timestamp()
    pydicom.dcmwrite(vtk_dir + '/' + str(utc_timestamp) + '_' + str(os.getpid()) + '_vtk.dcm', meta_data_dicom_vtk)

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

    # print("MP Plot Done : " + str(z))


def mp_plot(study_instance_id,
            rs,
            type_map,
            ct_ggo_dir,
            ct_con_dir,
            ct_fib_dir,
            vtk_dir,
            clusters):
    inps = list()
    for r in rs:
        inps.append((r,
                     type_map,
                     ct_ggo_dir,
                     ct_con_dir,
                     ct_fib_dir,
                     len(rs),
                     study_instance_id,
                     vtk_dir,
                     clusters))
    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        executor.map(worker_plot, inps)


def mp_plot_2(study_instance_id,
              rs,
              type_map,
              ct_ggo_dir,
              ct_con_dir,
              ct_fib_dir,
              vtk_dir,
              clusters):
    for r in rs:
        worker_plot((r,
                     type_map,
                     ct_ggo_dir,
                     ct_con_dir,
                     ct_fib_dir,
                     len(rs),
                     study_instance_id,
                     vtk_dir,
                     clusters))
