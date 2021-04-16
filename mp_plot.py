import copy
import math
import os
import random

import pydicom
import concurrent.futures

from common import BOX_SIZE, LABEL_GROUND_GLASS_OPACITY, LABEL_CONSOLIDATION, LABEL_FIBROSIS
from datetime import timezone
import datetime

from utils import read_progress, write_progress


def worker_plot(inp):
    r = inp[0]
    type_map = inp[1]
    ct_ggo_dir = inp[2]
    ct_con_dir = inp[3]
    ct_fib_dir = inp[4]
    total_number_of_instances = inp[5]
    study_instance_id = inp[6]
    meta_data_dicom = r[9]
    meta_data_dicom.pixel_array.fill(0)

    meta_data_dicom_ggo = copy.deepcopy(meta_data_dicom)
    meta_data_dicom_ggo.pixel_array.fill(0)

    meta_data_dicom_con = copy.deepcopy(meta_data_dicom)
    meta_data_dicom_con.pixel_array.fill(0)

    meta_data_dicom_fib = copy.deepcopy(meta_data_dicom)
    meta_data_dicom_fib.pixel_array.fill(0)

    z = int(math.floor(float(meta_data_dicom.ImagePositionPatient[2])))

    percent = read_progress(study_instance_id)
    new_percent = str(round(float(float(percent) + 20.0 / float(total_number_of_instances)), 2))
    write_progress(study_instance_id, new_percent)

    if type_map.get(z) is not None:
        value = type_map.get(z)
        affected_type = value[0]
        center_point = value[1]
        x_val = center_point[0]
        y_val = center_point[1]
        for x in range(x_val - int(BOX_SIZE/2) + int(random.randrange(-10, 10)), x_val + int(BOX_SIZE/2) + int(random.randrange(-10, 10)), 1):
            for y in range(y_val - int(BOX_SIZE/2) + int(random.randrange(-10, 10)), y_val + int(BOX_SIZE/2) + int(random.randrange(-10, 10)), 1):
                if affected_type.lower() == LABEL_GROUND_GLASS_OPACITY.lower():
                    meta_data_dicom_ggo.pixel_array[x][y] = 3500
                    meta_data_dicom_con.pixel_array[x][y] = 0
                    meta_data_dicom_fib.pixel_array[x][y] = 0
                elif affected_type.lower() == LABEL_CONSOLIDATION.lower():
                    print("Found Consolidation")
                    meta_data_dicom_ggo.pixel_array[x][y] = 0
                    meta_data_dicom_con.pixel_array[x][y] = 3400
                    meta_data_dicom_fib.pixel_array[x][y] = 0
                elif affected_type.lower() == LABEL_FIBROSIS.lower():
                    print("Found Fibrosis")
                    meta_data_dicom_ggo.pixel_array[x][y] = 0
                    meta_data_dicom_con.pixel_array[x][y] = 0
                    meta_data_dicom_fib.pixel_array[x][y] = 3300
                else:
                    meta_data_dicom_ggo.pixel_array[x][y] = 0
                    meta_data_dicom_con.pixel_array[x][y] = 0
                    meta_data_dicom_fib.pixel_array[x][y] = 0

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


def mp_plot(study_instance_id, rs, type_map, ct_ggo_dir, ct_con_dir, ct_fib_dir):
    inps = list()
    for r in rs:
        inps.append((r, type_map, ct_ggo_dir, ct_con_dir, ct_fib_dir, len(rs), study_instance_id))
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(worker_plot, inps)


def mp_plot_2(study_instance_id, rs, type_map, ct_ggo_dir, ct_con_dir, ct_fib_dir):
    for r in rs:
        worker_plot((r, type_map, ct_ggo_dir, ct_con_dir, ct_fib_dir, len(rs), study_instance_id))
