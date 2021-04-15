import math
import os
import glob
import pickle
import shutil
import tempfile
from zipfile import ZipFile
from werkzeug.utils import secure_filename

from driver2 import three_d_plot
from mp import process, process_2
from mp_plot import mp_plot, mp_plot_2
from mp_slice_plot import mp_slice_plot_2, mp_slice_plot

from report_assembler import assemble_report
from utils import get_25_score, create_json


def process_ct_instances(study_instance_id, ct_instances, work_dir, output_dir):
    valid_ct_slice_objects = list()
    ct_slice_objects = list()
    ggo_count = 0
    con_count = 0
    sub_count = 0
    fib_count = 0
    ple_count = 0
    pne_count = 0
    abnormal_slice_count = 0

    total_number_of_slices = len(ct_instances)

    if os.path.exists('points.pkl'):
        with open('points.pkl', 'rb') as f:
            rs = pickle.load(f)
    else:
        rs = process(ct_instances)
        with open('points.pkl', 'wb') as fp:
            pickle.dump(rs, fp)

    for r in rs:
        ct_slice, ggo, con, sub, fib, ple, pne, nor, affected_points, meta_data_dicom = r
        if not ct_slice.all_zeros:
            ct_slice_objects.append(ct_slice)
        if ggo or con or sub or fib or ple or pne > 10:
            abnormal_slice_count += 1
        ggo_count += ggo
        con_count += con
        sub_count += sub
        fib_count += fib
        ple_count += ple
        pne_count += pne
        if ct_slice.is_valid():
            valid_ct_slice_objects.append(ct_slice)

    scores = get_25_score(ct_slice_objects, study_instance_id)

    print('Total slice count: ' + str(total_number_of_slices))
    print('Abnormal slices  : ' + str(abnormal_slice_count))
    print('co_rads_score : ' + str(scores[5]))
    print('right_superior_lobe_percentage_affected : ' + str(scores[0]))
    print('right_middle_lobe_percentage_affected : ' + str(scores[1]))
    print('right_inferior_lobe_percentage_affected : ' + str(scores[2]))
    print('left_superior_lobe_percentage_affected : ' + str(scores[3]))
    print('left_inferior_lobe_percentage_affected : ' + str(scores[4]))

    final_json = create_json(study_instance_id,
                             scores,
                             ggo_count,
                             con_count,
                             sub_count,
                             fib_count,
                             ple_count,
                             abnormal_slice_count,
                             total_number_of_slices)

    type_map = dict()
    mp_slice_plot_2(rs, output_dir)
    
    for r in rs:
        for af in r[8]:
            x = int(math.floor(af[0][0]))
            y = int(math.floor(af[0][1]))
            z = int(math.floor(af[0][2]))
            v = af[1]
            type_map[z] = (v, (x, y, z))

    ct_ggo_dir = tempfile.mkdtemp()
    ct_con_dir = tempfile.mkdtemp()
    ct_fib_dir = tempfile.mkdtemp()

    mp_plot(rs, type_map, ct_ggo_dir, ct_con_dir, ct_fib_dir)
    three_d_plot(work_dir, output_dir, ct_ggo_dir, ct_con_dir, ct_fib_dir)

    shutil.rmtree(ct_ggo_dir)
    shutil.rmtree(ct_con_dir)
    shutil.rmtree(ct_fib_dir)

    return final_json


def predict(study_instance_id, work_dir, output_dir, logger):
    files = glob.glob(work_dir + '/**/*', recursive=True)
    final_json = process_ct_instances(study_instance_id, files, work_dir, output_dir)
    return final_json


def execute(study_instance_id, work_dir, output_dir, logger):
    try:
        fin_json_dict = predict(study_instance_id, work_dir, output_dir, logger)
        return fin_json_dict
    except Exception as e:
        rv = dict()
        rv['diagnosis'] = str(e)
        return rv


def store_and_verify_file(file_from_request, work_dir):
    if not file_from_request.filename:
        return -1, 'Empty file part provided!'
    try:
        file_path = os.path.join(work_dir, secure_filename(file_from_request.filename))
        file_from_request.save(file_path)
        return 0, file_path
    except Exception as ex:
        return -1, str(ex)


def generate_report(args, logger):
    try:
        file_from_request = args['zip_file']
        study_instance_id = output_dir = file_from_request.filename[:-4]
        if os.path.exists(output_dir):
            assemble_report(output_dir)
        else:
            os.makedirs(output_dir)
            file_dir = tempfile.mkdtemp()
            work_dir = tempfile.mkdtemp()
            ret, file_path = store_and_verify_file(file_from_request, work_dir=file_dir)
            if ret == 0:
                with ZipFile(file_path, 'r') as zipObj:
                    zipObj.extractall(work_dir)
                result = execute(study_instance_id, work_dir, output_dir, logger)
                shutil.rmtree(file_dir)
                shutil.rmtree(work_dir)
                return result
            else:
                shutil.rmtree(file_dir)
                shutil.rmtree(work_dir)
                rv = dict()
                rv['diagnosis'] = "Failed"
                return rv
    except Exception as e:
        rv = dict()
        rv['diagnosis'] = str(e)
        return rv
