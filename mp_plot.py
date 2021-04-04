import math
import os
import random

import miniball
import pydicom
import numpy as np
import concurrent.futures

from scipy.spatial import distance
from scipy.spatial.qhull import Delaunay

from common import BOX_SIZE, LABEL_GROUND_GLASS_OPACITY, LABEL_CONSOLIDATION, LABEL_SUB_PLEURAL_BAND, LABEL_FIBROSIS, \
    LABEL_PLEURAL_EFFUSION, LABEL_PNEUMOTHORAX
from datetime import timezone
import datetime


def get_color_random(color_map, point):
    if point[1] > 256:
        return 4096
    else:
        return 2048

def get_color(color_map, point):
    return 3025
'''
    ret = 2000
    for key, value in color_map.items():
        dst = distance.euclidean(key, point)
        if dst < BOX_SIZE / 2:
            if value.lower() == LABEL_GROUND_GLASS_OPACITY.lower():
                ret = 2000
                break
            elif value.lower() == LABEL_CONSOLIDATION.lower():
                ret = 1000
                break
            elif value.lower() == LABEL_SUB_PLEURAL_BAND.lower():
                ret = 500
                break
            elif value.lower() == LABEL_FIBROSIS.lower():
                ret = -500
                break
            elif value.lower() == LABEL_PLEURAL_EFFUSION.lower():
                ret = -1000
                break
            elif value.lower() == LABEL_PNEUMOTHORAX.lower():
                ret = -2000
                break
            else:
                ret = 1000
                break
    return ret
'''


def checkpoint(x, y, z, center, radius):
    d = math.sqrt((x - center[0]) * (x - center[0]) +
                  (y - center[1]) * (y - center[1]) +
                  (z - center[2]) * (z - center[2]))
    if d < radius:
        return True
    else:
        return False

def worker_plot_2(inp):
    r = inp[0]
    ct_mod_dir = inp[3]
    meta_data_dicom = r[9]
    dt = datetime.datetime.now(timezone.utc)
    utc_time = dt.replace(tzinfo=timezone.utc)
    utc_timestamp = utc_time.timestamp()
    pydicom.dcmwrite(ct_mod_dir + '/' + str(utc_timestamp) + '_' + str(os.getpid()) + '.dcm', meta_data_dicom)


def worker_plot(inp):
    r = inp[0]
    clusters = inp[1]
    color_map = inp[2]
    ct_mod_dir = inp[3]
    meta_data_dicom = r[9]
    meta_data_dicom.pixel_array.fill(0)
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
                    color = get_color(color_map, point)
                    meta_data_dicom.pixel_array[x][y] = color
    meta_data_dicom.PixelData = meta_data_dicom.pixel_array.tobytes()
    dt = datetime.datetime.now(timezone.utc)
    utc_time = dt.replace(tzinfo=timezone.utc)
    utc_timestamp = utc_time.timestamp()
    pydicom.dcmwrite(ct_mod_dir + '/' + str(utc_timestamp) + '_' + str(os.getpid()) + '.dcm', meta_data_dicom)


def mp_plot(rs, clusters, color_map, ct_mod_dir):
    inps = list()
    for r in rs:
        inps.append((r, clusters, color_map, ct_mod_dir))
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(worker_plot, inps)


def mp_plot_2(rs, clusters, color_map, ct_mod_dir):
    for r in rs:
        worker_plot((r, clusters, color_map, ct_mod_dir))
