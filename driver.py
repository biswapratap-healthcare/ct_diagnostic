import math
import os
import glob
import tempfile
from zipfile import ZipFile
from werkzeug.utils import secure_filename

from mp import process, process_2
from mp_plot import mp_plot, mp_plot_2
from mp_slice_plot import mp_slice_plot_2, mp_slice_plot
from sklearn.cluster import DBSCAN


ct_ggo_dir = 'ct_ggo_dir'
ct_con_dir = 'ct_con_dir'
ct_fib_dir = 'ct_fib_dir'


def process_ct_instances(ct_instances):
    rets_list = process(ct_instances)
    # with open('points.pkl', 'wb') as fp:
    #     pickle.dump(rets_list, fp)
    # with open('points.pkl', 'rb') as f:
    #     data = list()
    #     type_map = dict()
    #     rs = pickle.load(f)
    data = list()
    type_map = dict()
    rs = rets_list
    mp_slice_plot(rs)

    for r in rs:
        for af in r[8]:
            x = int(math.floor(af[0][0]))
            y = int(math.floor(af[0][1]))
            z = int(math.floor(af[0][2]))
            v = af[1]
            type_map[(x, y, z)] = v
            data.append([x, y, z])

    db = DBSCAN(eps=32, min_samples=1).fit(data)
    labels = db.labels_
    components = db.components_
    num_of_clusters = set(labels)
    print(num_of_clusters)
    clusters = dict()
    for label, point in zip(labels, components):
        if clusters.get(label) is None:
            clusters[label] = [point]
        else:
            clusters.get(label).append(point)

    if os.path.exists(ct_ggo_dir) is False and \
            os.path.exists(ct_con_dir) is False and \
            os.path.exists(ct_fib_dir) is False:
        os.makedirs(ct_ggo_dir)
        os.makedirs(ct_con_dir)
        os.makedirs(ct_fib_dir)

        mp_plot(rs, clusters, type_map, ct_ggo_dir, ct_con_dir, ct_fib_dir)


def predict(work_dir, logger):
    files = glob.glob(work_dir + '/**/*', recursive=True)
    final_json = process_ct_instances(files)
    return final_json


def execute(work_dir, logger):
    try:
        fin_json_dict = predict(work_dir, logger)
        rv = dict()
        rv['diagnosis'] = "Success"
        return rv
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
        file_dir = tempfile.mkdtemp()
        work_dir = tempfile.mkdtemp()
        ret, file_path = store_and_verify_file(file_from_request, work_dir=file_dir)
        if ret == 0:
            with ZipFile(file_path, 'r') as zipObj:
                zipObj.extractall(work_dir)
            result = execute(work_dir, logger)
            return result
        else:
            rv = dict()
            rv['diagnosis'] = "Failed"
            return rv
    except Exception as e:
        rv = dict()
        rv['diagnosis'] = str(e)
        return rv


