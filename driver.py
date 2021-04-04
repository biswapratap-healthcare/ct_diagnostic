import math
import os
import glob
import pickle
import numpy as np
from os.path import isfile
import matplotlib.pyplot as plt

import pydicom
from scipy.spatial import ConvexHull

from mp import process, process_2
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


def twoDPlot2(manipulated_ct_dir):
    dicom_files = glob.glob(manipulated_ct_dir + '/**/*', recursive=True)
    files = [pydicom.dcmread(file) for file in dicom_files if isfile(file)]
    slices = []
    skip_count = 0
    for file in files:
        if hasattr(file, 'SliceLocation'):
            slices.append(file)
        else:
            skip_count = skip_count + 1

    print("skipped, no SliceLocation: {}".format(skip_count))
    slices = sorted(slices, key=lambda s: s.SliceLocation)

    ps = slices[0].PixelSpacing
    ss = slices[0].SliceThickness
    ax_aspect = ps[1] / ps[0]
    sag_aspect = ps[1] / ss
    cor_aspect = ss / ps[0]

    img_shape = list(slices[0].pixel_array.shape)
    img_shape.append(len(slices))
    img3d = np.zeros(img_shape)

    for i, s in enumerate(slices):
        img2d = s.pixel_array
        img2d = img2d.astype(float)
        img2d = (np.maximum(img2d, 0) / img2d.max()) * 255.0
        img2d = np.uint8(img2d)
        img3d[:, :, i] = img2d

    plt.imshow(img3d[img_shape[0] // 2, :, :].T, cmap='gray', vmin=0, vmax=255)
    plt.show()


def twoDPlot(manipulated_ct_dir):
    files = glob.glob(manipulated_ct_dir + '/**/*', recursive=True)
    lstFilesDCM = [file for file in files if isfile(file)]
    RefDs = pydicom.read_file(lstFilesDCM[0])
    ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFilesDCM))
    ConstPixelSpacing = (float(RefDs.PixelSpacing[0]),  float(RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))
    xx = np.arange(0.0, (ConstPixelDims[0] + 1) * ConstPixelSpacing[0], ConstPixelSpacing[0])
    yy = np.arange(0.0, (ConstPixelDims[1] + 1) * ConstPixelSpacing[1], ConstPixelSpacing[1])
    zz = np.arange(0.0, (ConstPixelDims[2] + 1) * ConstPixelSpacing[2], ConstPixelSpacing[2])
    ArrayDicom = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)
    for filenameDCM in lstFilesDCM:
        ds = pydicom.read_file(filenameDCM)
        ArrayDicom[:, :, lstFilesDCM.index(filenameDCM)] = ds.pixel_array
    plt.figure(dpi=1600)
    plt.axes().set_aspect('equal', 'datalim')
    plt.set_cmap(plt.gray())
    plt.pcolormesh(xx, yy, np.flipud(ArrayDicom[:, :, 80]))


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
        twoDPlot2(ct_mod_dir)
        # g = load(ct_mod_dir)
        # show(g)
        exit(0)
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
            components = db.components_
            num_of_clusters = set(labels)
            print(num_of_clusters)
            clusters = dict()
            for label, point in zip(labels, components):
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
        test_sids = ['1.2.826.0.1.3680043.8.1678.101.10637217542821864049.962592']
        for sid in test_sids:
            try:
                param = dict()
                param['study_instance_id'] = sid
                fin_json_dict = predict(param)
            except Exception as e:
                continue
