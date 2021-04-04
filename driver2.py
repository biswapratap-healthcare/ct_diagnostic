import glob
import os
import pydicom
import numpy as np
from os.path import isfile

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
import matplotlib.pyplot as plt
from skimage.morphology import disk, opening

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
        binary_image = np.array(image[n] > -200, dtype=np.int8) + 1
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


def plot_3d(p_segmented_lungs, p_affected_images, p_threshold_lung, p_threshold_affected):
    p_affected_images_t = p_affected_images.transpose(2, 1, 0)
    p_segmented_lungs_t = p_segmented_lungs.transpose(2, 1, 0)
    verts_affected, faces_affected, _, _ = measure.marching_cubes(p_affected_images_t,
                                                                  p_threshold_affected,
                                                                  step_size=3)
    verts_lung, faces_lung, _, _ = measure.marching_cubes(p_segmented_lungs_t,
                                                          p_threshold_lung,
                                                          step_size=3)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    mesh_affected = Poly3DCollection(verts_affected[faces_affected], alpha=0.8)
    mesh_affected.set_facecolor("r")
    mesh_lung = Poly3DCollection(verts_lung[faces_lung], alpha=0.2)
    mesh_lung.set_facecolor("navy")
    ax.add_collection3d(mesh_affected)
    ax.add_collection3d(mesh_lung)
    ax.set_xlim(0, p_segmented_lungs_t.shape[0])
    ax.set_ylim(0, p_segmented_lungs_t.shape[1])
    ax.set_zlim(0, p_segmented_lungs_t.shape[2])
    # plt.show()
    plt.savefig('foo.png', bbox_inches='tight')


if __name__ == "__main__":
    study_instance_id = '1.2.826.0.1.3680043.8.1678.101.10637217542821864049.962592'
    scans_lung = load_scans(study_instance_id)
    scans_affected = load_scans_2('./../ct_predictor_2/ct_mod_dir/')
    hu_scans_lung = transform_to_hu(scans_lung)
    segmented_lungs = segment_lung_mask(hu_scans_lung)
    affected_images = np.stack([file.pixel_array for file in scans_affected])
    affected_images = affected_images.astype(np.int16)
    plot_3d(segmented_lungs, affected_images, -600, 100)
    print('c')
