import concurrent.futures
import math
import os
# import cv2
import numpy as np
from sklearn.cluster import DBSCAN

from common import BOX_SIZE, LABEL_GROUND_GLASS_OPACITY, LABEL_CONSOLIDATION, LABEL_FIBROSIS


def box_plot(stacked_img, min_x, max_x, min_y, max_y, t):
    if t.lower() == LABEL_GROUND_GLASS_OPACITY.lower():
        r = 0
        g = 0
        b = 255
    elif t.lower() == LABEL_CONSOLIDATION.lower():
        r = 0
        g = 255
        b = 0
    elif t.lower() == LABEL_FIBROSIS.lower():
        r = 0
        g = 255
        b = 255
    else:
        b = 255
        g = 255
        r = 255
    for x in range(min_x, max_x):
        stacked_img[x][min_y][0] = r
        stacked_img[x][min_y][1] = g
        stacked_img[x][min_y][2] = b
    for x in range(min_x, max_x):
        stacked_img[x][min_y - 1][0] = r
        stacked_img[x][min_y - 1][1] = g
        stacked_img[x][min_y - 1][2] = b

    for x in range(min_x, max_x):
        stacked_img[x][max_y][0] = r
        stacked_img[x][max_y][1] = g
        stacked_img[x][max_y][2] = b
    for x in range(min_x, max_x):
        stacked_img[x][max_y - 1][0] = r
        stacked_img[x][max_y - 1][1] = g
        stacked_img[x][max_y - 1][2] = b

    for y in range(min_y, max_y):
        stacked_img[min_x][y][0] = r
        stacked_img[min_x][y][1] = g
        stacked_img[min_x][y][2] = b
    for y in range(min_y, max_y):
        stacked_img[min_x - 1][y][0] = r
        stacked_img[min_x - 1][y][1] = g
        stacked_img[min_x - 1][y][2] = b

    for y in range(min_y, max_y):
        stacked_img[max_x][y][0] = r
        stacked_img[max_x][y][1] = g
        stacked_img[max_x][y][2] = b
    for y in range(min_y, max_y):
        stacked_img[max_x - 1][y][0] = r
        stacked_img[max_x - 1][y][1] = g
        stacked_img[max_x - 1][y][2] = b
    return stacked_img


def worker_plot(inp):
    z_index = 0
    type_map = dict()

    if len(inp[8]) == 0:
        return

    metadata_dicom = inp[9]
    img = metadata_dicom.pixel_array
    img_2d = img.astype(float)
    img_2d_scaled = (np.maximum(img_2d, 0) / img_2d.max()) * 255.0
    img_2d_scaled = np.uint8(img_2d_scaled)
    stacked_img = np.stack((img_2d_scaled,) * 3, axis=-1)

    for af in inp[8]:
        x = int(math.floor(af[0][0]))
        y = int(math.floor(af[0][1]))
        z = z_index = int(math.floor(af[0][2]))
        v = af[1]
        if type_map.get(v) is None:
            type_map[v] = [[x, y, z]]
        else:
            type_map.get(v).append([x, y, z])

    for key, value in type_map.items():
        db = DBSCAN(eps=50, min_samples=1).fit(value)
        labels = list(db.labels_)
        components = [list(c) for c in db.components_]
        num_of_clusters = set(labels)
        print(num_of_clusters)
        clusters = dict()
        for label, point in zip(labels, components):
            if clusters.get(label) is None:
                clusters[label] = [point]
            else:
                clusters.get(label).append(point)

        for k, v in clusters.items():
            c_x = [c[0] for c in v]
            c_y = [c[1] for c in v]
            min_x = min(c_x) - int((BOX_SIZE / 2))
            max_x = max(c_x) + int((BOX_SIZE / 2))
            min_y = min(c_y) - int((BOX_SIZE / 2))
            max_y = max(c_y) + int((BOX_SIZE / 2))

            stacked_img = box_plot(stacked_img, min_x, max_x, min_y, max_y, key)

    # cv2.imwrite(os.path.join('slices', str(z_index) + '.jpg'), stacked_img)


def mp_slice_plot(rs):
    inps = list()
    for r in rs:
        inps.append(r)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(worker_plot, inps)


def mp_slice_plot_2(rs):
    for r in rs:
        worker_plot(r)
