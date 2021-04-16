import math
import os
import glob
import pickle
from os.path import isfile

from mp import process, process_2
from mp_plot import mp_plot, mp_plot_2
from mp_slice_plot import mp_slice_plot_2, mp_slice_plot
from utils import get_instance_files


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
    ct_ggo_dir = 'ct_ggo_dir'
    ct_con_dir = 'ct_con_dir'
    ct_fib_dir = 'ct_fib_dir'

    study_instance_id = "1.2.826.0.1.3680043.8.1678.101.10637213214991521358.314450"

    if os.path.exists('points.pkl'):
        with open('points.pkl', 'rb') as f:
            type_map = dict()
            rs = pickle.load(f)
            # mp_slice_plot(rs)
            for r in rs:
                for af in r[8]:
                    x = int(math.floor(af[0][0]))
                    y = int(math.floor(af[0][1]))
                    z = int(math.floor(af[0][2]))
                    v = af[1]
                    type_map[z] = (v, (x, y, z))

        if os.path.exists(ct_ggo_dir) is False and \
           os.path.exists(ct_con_dir) is False and \
           os.path.exists(ct_fib_dir) is False:

            os.makedirs(ct_ggo_dir)
            os.makedirs(ct_con_dir)
            os.makedirs(ct_fib_dir)

            mp_plot(study_instance_id, rs, type_map, ct_ggo_dir, ct_con_dir, ct_fib_dir)

            # g = load(ct_ggo_dir)
            # show(g)
    else:
        test_sids = ['1.2.826.0.1.3680043.8.1678.101.10637216566590543417.794673']
        for sid in test_sids:
            try:
                param = dict()
                param['study_instance_id'] = sid
                fin_json_dict = predict(param)
            except Exception as e:
                print(str(e))
                continue
