import json
import math
import os
import glob
import pickle
import shutil
import tempfile
from zipfile import ZipFile

from sklearn.cluster import DBSCAN
from vedo import load, show, screenshot
from werkzeug.utils import secure_filename

from common import ERROR_FILE, BOX_SIZE
from driver2 import three_d_plot
from mp import process, process_2
from mp_plot import mp_plot, mp_plot_2
from mp_slice_plot import mp_slice_plot_2, mp_slice_plot

from report_assembler import assemble_report
from utils import get_25_score, create_json, write_progress


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
    write_progress(study_instance_id, "20")

    if os.path.exists('points.pkl'):
        with open('points.pkl', 'rb') as f:
            rs = pickle.load(f)
    else:
        rs = process(study_instance_id, ct_instances)
        with open('points.pkl', 'wb') as fp:
            pickle.dump(rs, fp)

    write_progress(output_dir, "50")

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

    write_progress(output_dir, "55")

    with open(output_dir + '/out.json', 'w') as f:
        final_json_str = json.dumps(final_json, indent=4)
        f.write(final_json_str)

    mp_slice_plot_2(rs, output_dir)
    write_progress(output_dir, "60")
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

    vtk_dir = tempfile.mkdtemp()
    ct_ggo_dir = tempfile.mkdtemp()
    ct_con_dir = tempfile.mkdtemp()
    ct_fib_dir = tempfile.mkdtemp()

    mp_plot(study_instance_id,
            rs,
            type_map,
            ct_ggo_dir,
            ct_con_dir,
            ct_fib_dir,
            vtk_dir)

    write_progress(output_dir, "80")

    # vtk_plot(vtk_dir, output_dir)
    three_d_plot(work_dir, output_dir, ct_ggo_dir, ct_con_dir, ct_fib_dir)

    shutil.rmtree(ct_ggo_dir)
    shutil.rmtree(ct_con_dir)
    shutil.rmtree(ct_fib_dir)

    write_progress(output_dir, "90")


def vtk_plot(work_dir, output_dir):
    volume = load(work_dir)
    show(volume)
    screenshot(filename=output_dir + '/vtk.png')


def predict(study_instance_id, work_dir, output_dir):
    files = glob.glob(work_dir + '/**/*', recursive=True)
    process_ct_instances(study_instance_id, files, work_dir, output_dir)


def execute(study_instance_id, work_dir, output_dir):
    try:
        predict(study_instance_id, work_dir, output_dir)
    except Exception as e:
        with open(output_dir + '/' + ERROR_FILE, "a+") as f:
            f.write(str(e))


def store_and_verify_file(file_from_request, work_dir):
    if not file_from_request.filename:
        return -1, 'Empty file part provided!'
    try:
        file_path = os.path.join(work_dir, secure_filename(file_from_request.filename))
        file_from_request.save(file_path)
        return 0, file_path
    except Exception as ex:
        return -1, str(ex)


def generate_report(study_instance_id, work_dir, output_dir):
    try:
        execute(study_instance_id, work_dir, output_dir)
        shutil.rmtree(work_dir)
        assemble_report(output_dir)
        write_progress(output_dir, "100")
    except Exception as e:
        if os.path.exists(output_dir) is False:
            os.makedirs(output_dir)
        with open(output_dir + '/' + ERROR_FILE, "a+") as f:
            f.write(str(e))
