import os

import pydicom
# import multiprocessing
import concurrent.futures

from ct_slice import CTSlice
from rest_client import update_progress_percent
from utils import read_progress, write_progress, read_last_sent_progress, write_last_sent_progress, dictify


def worker(inp):
    try:
        study_instance_id = inp[0]
        ct_instance = inp[1]
        total_number_of_instances = inp[2]

        print("Processing : " + str(os.path.basename(ct_instance)))

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
            print("mp: Got Empty Percentage")

        meta_data_dicom = pydicom.dcmread(ct_instance)
        ct_slice = CTSlice(meta_data_dicom)
        ggo, con, sub, fib, ple, pne, nor, affected_points = ct_slice.get_box_label_distribution()
        print("Done : " + str(os.path.basename(ct_instance)))
    except Exception as e:
        print('mp:worker() exception = ' + str(e))
        ct_slice = None
        ggo = None
        con = None
        sub = None
        fib = None
        ple = None
        pne = None
        nor = None
        affected_points = None
        meta_data_dicom = None
    return ct_slice, ggo, con, sub, fib, ple, pne, nor, affected_points, meta_data_dicom


def process(study_instance_id, ct_instances):
    inps = list()
    for ct_instance in ct_instances:
        meta_data_dicom = pydicom.dcmread(ct_instance)
        ds = dictify(meta_data_dicom)
        modality = ds['Modality']
        if modality == 'CT':
            inps.append((study_instance_id, ct_instance, len(ct_instances)))
    if len(inps) == 0:
        print("DICOM is most probably not a CT zip.")
        return []
    max_processes = 10
    print("Processes count = " + str(max_processes))
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_processes) as executor:
        rets = executor.map(worker, inps)
        rets_list = list()
        for ret in rets:
            if ret[0] is None:
                continue
            else:
                rets_list.append(ret)
        return rets_list


def process_2(study_instance_id, ct_instances):
    rets_list = list()
    for ct_instance in ct_instances:
        meta_data_dicom = pydicom.dcmread(ct_instance)
        ds = dictify(meta_data_dicom)
        modality = ds['Modality']
        if modality == 'CT':
            ret = worker((study_instance_id, ct_instance, len(ct_instances)))
            if ret[0] is None:
                continue
            else:
                rets_list.append(ret)
    return rets_list
