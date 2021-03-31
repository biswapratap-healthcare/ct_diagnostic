import math
import os
import glob
import pickle
import random
from os.path import isfile

from scipy.spatial import ConvexHull

from common import LABEL_GROUND_GLASS_OPACITY, LABEL_CONSOLIDATION, LABEL_SUB_PLEURAL_BAND, LABEL_FIBROSIS, \
    LABEL_PLEURAL_EFFUSION, LABEL_PNEUMOTHORAX
from mp import process
from mp_plot import mp_plot, mp_plot_2
from utils import get_instance_files
from vedo.io import load
from vedo.plotter import show
from sklearn.cluster import DBSCAN


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


def is_p_inside_points_hull(points, p):
    hull = ConvexHull(points)
    points.append(p)
    new_hull = ConvexHull(points)
    l1 = list(hull.vertices)
    l1.sort()
    l2 = list(new_hull.vertices)
    l2.sort()
    if l1 == l2:
        return True
    else:
        return False


if __name__ == "__main__":
    ct_mod_dir = 'ct_mod_dir'
    if os.path.exists(ct_mod_dir):
        g = load(ct_mod_dir)
        show(g)
    if os.path.exists('points.pkl'):
        with open('points.pkl', 'rb') as f:
            data = list()
            color_map = dict()
            rs = pickle.load(f)
            for r in rs:
                for af in r[8]:
                    x = int(math.floor(af[0][0]))
                    y = int(math.floor(af[0][1]))
                    z = int(math.floor(af[0][2]))
                    v = af[1]
                    color_map[(x, y, z)] = v
                    data.append([x, y, z])
            db = DBSCAN(eps=32, min_samples=1).fit(data)
            labels = db.labels_
            num_of_clusters = set(labels)
            print(num_of_clusters)
            clusters = dict()
            for label, point in zip(labels, data):
                if clusters.get(label) is None:
                    clusters[label] = [point]
                else:
                    clusters.get(label).append(point)

            if os.path.exists(ct_mod_dir) is False:
                os.makedirs(ct_mod_dir)
                mp_plot(rs, clusters, color_map, ct_mod_dir)
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
