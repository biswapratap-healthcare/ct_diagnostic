import math
import os

import pydicom
import numpy as np
import concurrent.futures
from scipy.spatial.qhull import Delaunay

from common import BOX_SIZE
from datetime import timezone
import datetime


def worker_plot(inp):
    r = inp[0]
    clusters = inp[1]
    ct_mod_dir = inp[2]
    meta_data_dicom = r[9]
    z = int(math.floor(float(meta_data_dicom.ImagePositionPatient[2])))
    for key, value in clusters.items():
        try:
            new_value = list()
            for c in value:
                new_value.append([int(math.floor(c[0])), int(math.floor(c[1])), int(math.floor(c[2]))])
            c_x = [c[0] for c in new_value]
            c_y = [c[1] for c in new_value]
            c_z = [c[2] for c in new_value]
            min_x = int(math.floor(min(c_x))) - int((BOX_SIZE / 2))
            max_x = int(math.floor(max(c_x))) + int((BOX_SIZE / 2))
            min_y = int(math.floor(min(c_y))) - int((BOX_SIZE / 2))
            max_y = int(math.floor(max(c_y))) + int((BOX_SIZE / 2))
            if 0 < len(new_value) < 100:
                new_value.clear()
                new_value.append(
                    [int(math.floor(min(c_x))) - 5, int(math.floor(min(c_y))) - 5, int(math.floor(min(c_z))) - 5])
                new_value.append(
                    [int(math.floor(min(c_x))) + 5, int(math.floor(min(c_y))) - 5, int(math.floor(min(c_z))) - 5])
                new_value.append(
                    [int(math.floor(min(c_x))) + 5, int(math.floor(min(c_y))) + 5, int(math.floor(min(c_z))) - 5])
                new_value.append(
                    [int(math.floor(min(c_x))) + 5, int(math.floor(min(c_y))) + 5, int(math.floor(min(c_z))) + 5])
                new_value.append(
                    [int(math.floor(min(c_x))) - 5, int(math.floor(min(c_y))) - 5, int(math.floor(min(c_z))) + 5])
                new_value.append(
                    [int(math.floor(min(c_x))) - 5, int(math.floor(min(c_y))) + 5, int(math.floor(min(c_z))) + 5])
                new_value.append(
                    [int(math.floor(min(c_x))) - 5, int(math.floor(min(c_y))) + 5, int(math.floor(min(c_z))) - 5])
                new_value.append(
                    [int(math.floor(min(c_x))) + 5, int(math.floor(min(c_y))) - 5, int(math.floor(min(c_z))) + 5])
            hull = Delaunay(np.array(new_value))
        except Exception as e:
            continue
        for x in range(min_x, max_x, 1):
            for y in range(min_y, max_y, 1):
                point = [x, y, z]
                if hull.find_simplex(np.array(point)) >= 0:
                    meta_data_dicom.pixel_array[x][y] = 2000
    meta_data_dicom.PixelData = meta_data_dicom.pixel_array.tobytes()
    dt = datetime.datetime.now(timezone.utc)
    utc_time = dt.replace(tzinfo=timezone.utc)
    utc_timestamp = utc_time.timestamp()
    pydicom.dcmwrite(ct_mod_dir + '/' + str(utc_timestamp) + '_' + str(os.getpid()) + '.dcm', meta_data_dicom)


def mp_plot(rs, clusters, ct_mod_dir):
    inps = list()
    for r in rs:
        inps.append((r, clusters, ct_mod_dir))
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(worker_plot, inps)


def mp_plot_2(rs, clusters, ct_mod_dir):
    for r in rs:
        worker_plot((r, clusters, ct_mod_dir))
