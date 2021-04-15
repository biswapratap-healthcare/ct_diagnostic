import glob
import os
import pydicom
import numpy as np
from os.path import isfile

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
import matplotlib.pyplot as plt
from skimage.morphology import disk, opening, closing

from utils import get_instance_files


CT_OFFSET = 1024
ZERO_VALUE = -2000


def read_dicom_array(in_path):
    lung_dicom = pydicom.read_file(in_path)
    slice_array = lung_dicom.pixel_array
    slice_array[slice_array == -2000] = 0
    return int(lung_dicom.InstanceNumber), slice_array.astype(np.int16) - CT_OFFSET


def load_scans_2(folder):
    dicom_files = glob.glob(folder + '/**/*', recursive=True)
    slices = [pydicom.dcmread(dicom_file) for dicom_file in dicom_files]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    return slices


def load_scans(sid):
    src_path = os.path.dirname(os.path.abspath(__file__))
    ct_dir = src_path + '/../ai_ct_diagnostic_trainer/studies/' + str(sid) + '/'
    files = glob.glob(ct_dir + '/**/*', recursive=True)
    instance_files = [file for file in files if isfile(file)]
    ct_instance_files = get_instance_files(instance_files)

    if len(ct_instance_files) == 0:
        print('error: No valid CT instances found !!')
        exit(-1)

    slices = [pydicom.dcmread(ct_instance_file) for ct_instance_file in ct_instance_files]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    return slices


def set_outside_scanner_to_air(raw_pixel_arrays):
    raw_pixel_arrays[raw_pixel_arrays <= -1000] = 0
    return raw_pixel_arrays


def transform_to_hu(slices):
    images = np.stack([file.pixel_array for file in slices])
    images = images.astype(np.int16)
    images = set_outside_scanner_to_air(images)
    # convert to HU
    for n in range(len(slices)):
        intercept = slices[n].RescaleIntercept
        slope = slices[n].RescaleSlope
        if slope != 1:
            images[n] = slope * images[n].astype(np.float64)
            images[n] = images[n].astype(np.int16)
        images[n] += np.int16(intercept)
    return np.array(images, dtype=np.int16)


def segment_lung_mask(image):
    segmented = np.zeros(image.shape)
    for n in range(image.shape[0]):
        binary_image = np.array(image[n] > -320, dtype=np.int8) + 1
        labels = measure.label(binary_image)
        bad_labels = np.unique([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]])
        for bad_label in bad_labels:
            binary_image[labels == bad_label] = 2
        # We have a lot of remaining small signals outside of the lungs that need to be removed.
        # In our competition closing is superior to fill_lungs
        selem = disk(2)
        binary_image = opening(binary_image, selem)
        binary_image -= 1  # Make the image actual binary
        binary_image = 1 - binary_image  # Invert it, lungs are now 1
        segmented[n] = binary_image.copy() * image[n]
    return segmented


def set_manual_window(hu_image, custom_center, custom_width):
    w_image = hu_image.copy()
    min_value = custom_center - (custom_width/2)
    max_value = custom_center + (custom_width/2)
    w_image[w_image < min_value] = min_value
    w_image[w_image > max_value] = max_value
    return w_image


def plot_3d(p_segmented_lungs,
            output_dir,
            p_ggo_images,
            p_con_images,
            p_fib_images,
            p_threshold_lung,
            p_threshold_affected):
    p_ggo_images_t = p_ggo_images.transpose(2, 1, 0)
    p_con_images_t = p_con_images.transpose(2, 1, 0)
    p_fib_images_t = p_fib_images.transpose(2, 1, 0)
    p_segmented_lungs_t = p_segmented_lungs.transpose(2, 1, 0)

    verts_ggo, faces_ggo, _, _ = measure.marching_cubes(p_ggo_images_t,
                                                        p_threshold_affected,
                                                        step_size=3)

    verts_con, faces_con, _, _ = measure.marching_cubes(p_con_images_t,
                                                        p_threshold_affected,
                                                        step_size=3)

    verts_fib, faces_fib, _, _ = measure.marching_cubes(p_fib_images_t,
                                                        p_threshold_affected,
                                                        step_size=3)

    verts_lung, faces_lung, _, _ = measure.marching_cubes(p_segmented_lungs_t,
                                                          p_threshold_lung,
                                                          step_size=3)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    mesh_ggo = Poly3DCollection(verts_ggo[faces_ggo], alpha=0.8)
    mesh_ggo.set_facecolor("r")

    mesh_con = Poly3DCollection(verts_ggo[faces_con], alpha=0.8)
    mesh_con.set_facecolor("g")

    mesh_fib = Poly3DCollection(verts_ggo[faces_fib], alpha=0.8)
    mesh_fib.set_facecolor("y")

    mesh_lung = Poly3DCollection(verts_lung[faces_lung], alpha=0.1)
    mesh_lung.set_facecolor("navy")

    ax.add_collection3d(mesh_ggo)
    ax.add_collection3d(mesh_con)
    ax.add_collection3d(mesh_fib)
    ax.add_collection3d(mesh_lung)

    ax.set_xlim(0, p_segmented_lungs_t.shape[0])
    ax.set_ylim(0, p_segmented_lungs_t.shape[1])
    ax.set_zlim(0, p_segmented_lungs_t.shape[2])
    plt.savefig(output_dir + '/natural.png', bbox_inches='tight')
    ax.view_init(90, 0)
    plt.draw()
    plt.savefig(output_dir + '/top.png', bbox_inches='tight')
    ax.view_init(0, 180)
    plt.draw()
    plt.savefig(output_dir + '/lateral.png', bbox_inches='tight')
    ax.view_init(0, 270)
    plt.draw()
    plt.savefig(output_dir + '/front.png', bbox_inches='tight')


# if __name__ == "__main__":
def three_d_plot(sid_dir, output_dir, ggo_dir, con_dir, fib_dir):
    scans_lung = load_scans_2(sid_dir)
    hu_scans_lung = transform_to_hu(scans_lung)
    segmented_lungs = segment_lung_mask(hu_scans_lung)

    scans_ggo = load_scans_2(ggo_dir)
    ggo_images = np.stack([file.pixel_array for file in scans_ggo])
    ggo_images = ggo_images.astype(np.int16)

    min_ggo_v = ggo_images.min()
    max_ggo_v = ggo_images.max()

    if min_ggo_v == max_ggo_v == 0.0:
        ggo_images[0][0][0] = 3500

    scans_con = load_scans_2(con_dir)
    con_images = np.stack([file.pixel_array for file in scans_con])
    con_images = con_images.astype(np.int16)

    min_con_v = con_images.min()
    max_con_v = con_images.max()

    if min_con_v == max_con_v == 0.0:
        con_images[0][0][0] = 3400

    scans_fib = load_scans_2(fib_dir)
    fib_images = np.stack([file.pixel_array for file in scans_fib])
    fib_images = fib_images.astype(np.int16)

    min_fib_v = fib_images.min()
    max_fib_v = fib_images.max()

    if min_fib_v == max_fib_v == 0.0:
        fib_images[0][0][0] = 3300

    plot_3d(segmented_lungs, output_dir, ggo_images, con_images, fib_images, -600, 100)
