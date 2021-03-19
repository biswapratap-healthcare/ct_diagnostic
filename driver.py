import os
import glob
import pickle
from os.path import isfile

import pydicom

from mp import process
from utils import get_instance_files
from vedo.io import load
from vedo.plotter import show


def process_ct_instances(ct_instances):
    rets_list = process(ct_instances)
    with open('points.pkl', 'wb') as fp:
        pickle.dump(rets_list, fp)
    return ""


def predict(args):
    study_instance_id = args['study_instance_id']
    print(f"Received request for {study_instance_id}")
    src_path = os.path.dirname(os.path.abspath(__file__))
    ct_dir = src_path + '/../ai_ct_diagnostic_trainer/studies/' + str(study_instance_id) + '/'
    files = glob.glob(ct_dir + '/**/*', recursive=True)
    instance_files = [file for file in files if isfile(file)]
    ct_instances = get_instance_files(instance_files)
    if len(ct_instances) == 0:
        return {'error': 'No valid CT instances found !!'}
    final_json = process_ct_instances(ct_instances)
    print(f"Finished request for {study_instance_id}")
    return final_json


if __name__ == "__main__":
    ct_mod_dir = 'ct_mod_dir'
    if os.path.exists('points.pkl'):
        with open('points.pkl', 'rb') as f:
            if os.path.exists(ct_mod_dir) is False:
                os.makedirs(ct_mod_dir)
                rs = pickle.load(f)
                points = list()
                counter = 1
                for r in rs:
                    affected_points = r[8]
                    meta_data_dicom = r[9]
                    for (x, y) in affected_points:
                        meta_data_dicom.pixel_array[x][y] = 2000
                    meta_data_dicom.PixelData = meta_data_dicom.pixel_array.tobytes()
                    pydicom.dcmwrite(ct_mod_dir + '/' + str(counter) + '.dcm', meta_data_dicom)
                    counter += 1
                g = load(ct_mod_dir)
                show(g)
    else:
        test_sids = ['1.2.826.0.1.3680043.8.1678.101.10637242073975371769.339989']
        for sid in test_sids:
            try:
                param = dict()
                param['study_instance_id'] = sid
                fin_json_dict = predict(param)
            except Exception as e:
                continue
