import os

import pydicom
import concurrent.futures

from ct_slice import CTSlice
from rest_client import update_progress_percent
from utils import read_progress, write_progress, read_last_sent_progress, write_last_sent_progress


def worker(inp):
    study_instance_id = inp[0]
    ct_instance = inp[1]
    total_number_of_instances = inp[2]

    percent = read_progress(study_instance_id)
    last_sent_percent = read_last_sent_progress(study_instance_id)

    if percent != '' and last_sent_percent != '':
        new_percent = str(round(float(float(percent) + 29.0/float(total_number_of_instances)), 2))
        write_progress(study_instance_id, new_percent)
        delta = float(new_percent) - float(last_sent_percent)
        if delta > 2.0:
            write_last_sent_progress(study_instance_id, new_percent)
            update_progress_percent(study_instance_id, new_percent)
    else:
        print("mp_plot: Got Empty Percentage")

    meta_data_dicom = pydicom.dcmread(ct_instance)
    ct_slice = CTSlice(meta_data_dicom)
    ggo, con, sub, fib, ple, pne, nor, affected_points = ct_slice.get_box_label_distribution()
    # print("Done : " + str(os.path.basename(ct_instance)))
    return ct_slice, ggo, con, sub, fib, ple, pne, nor, affected_points, meta_data_dicom


def process(study_instance_id, ct_instances):
    inps = list()
    for ct_instance in ct_instances:
        inps.append((study_instance_id, ct_instance, len(ct_instances)))
    print("CPU count = " + str(os.cpu_count()))
    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        rets = executor.map(worker, inps)
        rets_list = list()
        for ret in rets:
            rets_list.append(ret)
        return rets_list


def process_2(study_instance_id, ct_instances):
    rets_list = list()
    for ct_instance in ct_instances:
        rets_list.append(worker((study_instance_id, ct_instance, len(ct_instances))))
    return rets_list
