import math
import os
import glob
import pickle
from os.path import isfile

from vedo import load, show, write

from mp import process, process_2
from mp_plot import mp_plot, mp_plot_2
from mp_slice_plot import mp_slice_plot_2, mp_slice_plot
from utils import get_instance_files


def process_ct_instances(study_instance_idd, ct_instances):
    rets_list = process(study_instance_idd, ct_instances)
    with open('points.pkl', 'wb') as fp:
        pickle.dump(rets_list, fp)
    return ""


def predict(args):
    study_instance_idd = args['study_instance_id']
    print(f"Received request for {study_instance_id}")
    src_path = os.path.dirname(os.path.abspath(__file__))
    ct_dir = src_path + '/../Studies/' + str(study_instance_id) + '/'
    files = glob.glob(ct_dir + '/**/*', recursive=True)
    instance_files = [file for file in files if isfile(file)]
    ct_instances = get_instance_files(instance_files)
    if len(ct_instances) == 0:
        return {'error': 'No valid CT instances found !!'}
    final_json = process_ct_instances(study_instance_idd, ct_instances)
    print(f"Finished request for {study_instance_id}")
    return final_json


if __name__ == "__main__":
    vtk_dir = 'vtk_dir'
    ct_ggo_dir = 'ct_ggo_dir'
    ct_con_dir = 'ct_con_dir'
    ct_fib_dir = 'ct_fib_dir'

    g = load(vtk_dir)
    write(objct=g, fileoutput="out.vti")
    exit(0)

    study_instance_id = "1.2.826.0.1.3680043.8.1678.101.10637203703447639663.147272"

    if os.path.exists('points.pkl'):
        with open('points.pkl', 'rb') as f:
            rs = pickle.load(f)
            type_map = dict()

            for r in rs:
                for af in r[8]:
                    x = int(math.floor(af[0][0]))
                    y = int(math.floor(af[0][1]))
                    z = int(math.floor(af[0][2]))
                    v = af[1]
                    if type_map.get(z) is None:
                        type_map[z] = [(v, (x, y, z))]
                    else:
                        type_map.get(z).append((v, (x, y, z)))
    else:
        test_sid = '1.2.826.0.1.3680043.8.1678.101.10637203703447639663.147272'
        param = dict()
        param['study_instance_id'] = test_sid
        fin_json_dict = predict(param)
        exit(0)

    if os.path.exists(ct_ggo_dir) is False and \
       os.path.exists(ct_con_dir) is False and \
       os.path.exists(ct_fib_dir) is False and \
       os.path.exists(vtk_dir) is False:
        os.makedirs(vtk_dir)
        os.makedirs(ct_ggo_dir)
        os.makedirs(ct_con_dir)
        os.makedirs(ct_fib_dir)

        mp_plot(study_instance_id,
                rs,
                type_map,
                ct_ggo_dir,
                ct_con_dir,
                ct_fib_dir,
                vtk_dir)

        g = load(vtk_dir)
        show(g)
