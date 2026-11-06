import os

import matplotlib as mpl
import matplotlib.cm
import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy.ndimage import binary_erosion

from charm.report import brainsprite_helper
from charm.utils import nifti

se = ndimage.generate_binary_structure(3, 1)


cmaplist_final = [
    (0, 0, 0, 0),
    (30 / 255, 30 / 255, 30 / 255, 1.0),
    (255 / 255, 255 / 255, 255 / 255, 1.0),
    (104 / 255, 163 / 255, 255 / 255, 1.0),
    (255 / 255, 239 / 255, 179 / 255, 1.0),
    (255 / 255, 166 / 255, 133 / 255, 1.0),
    (255 / 255, 240 / 255, 0, 1.0),
    (255 / 255, 239 / 255, 179 / 255, 1.0),
    (255 / 255, 138 / 255, 57 / 255, 1.0),
    (0, 65 / 255, 142 / 255, 1.0),
]


cmaplist_affine = [
    (0, 0, 0, 0),
    (0, 0, 0, 0),
    (0, 0, 0, 0),
    (0, 0, 0, 0),
    (0, 0, 0, 0),
    (255 / 255, 166 / 255, 133 / 255, 1.0),
    (255 / 255, 239 / 255, 179 / 255, 1.0),
]

cmap_final = mpl.colors.LinearSegmentedColormap.from_list("final_segmentation", cmaplist_final, len(cmaplist_final))

cmap_affine = mpl.colors.LinearSegmentedColormap.from_list("template", cmaplist_affine, len(cmaplist_affine))


def _registration_overlay(T2):
    T2_data = nifti.load(T2)
    bufferT2 = T2_data.get_fdata()
    bufferT2 = (bufferT2 - np.min(bufferT2)) / (np.max(bufferT2) - np.min(bufferT2))
    d = (
        ndimage.sobel(bufferT2, axis=0, mode="constant") ** 2
        + ndimage.sobel(bufferT2, axis=1, mode="constant") ** 2
        + ndimage.sobel(bufferT2, axis=2, mode="constant") ** 2
    )
    d = np.sqrt(d)
    p95 = np.percentile(d, 95)
    d[d > p95] = p95
    d /= d.max()
    # Keep only the strongest edges: map 0.7 to 0 and the max of 1 to 0.9.
    d = 3 * (d - 0.7)
    d[d < 0] = np.nan

    return nib.Nifti1Image(np.squeeze(d), T2_data.affine)


def _final_overlay(labeling, labels=(1, 2, 3, 6, 9, 8, 7, 5), medfilter_size=None):
    tissues_data = nifti.load(labeling)
    tissues_buffer = np.round(np.squeeze(tissues_data.get_fdata()))
    if medfilter_size is not None:
        tissues_buffer = ndimage.median_filter(tissues_buffer, size=medfilter_size)
    # All False.
    tissue_mask = tissues_buffer == -1000
    label_buffer = np.zeros_like(tissues_buffer)
    for label in labels:
        tissue_mask = tissue_mask | (tissues_buffer == label)
        ero = binary_erosion(np.squeeze(tissue_mask), se, 1)
        label_buffer[(tissue_mask & ~ero & ~(label_buffer > 0))] = label

    return nib.Nifti1Image(np.uint16(np.squeeze(label_buffer)), tissues_data.affine)


def _cap_intensities(im):
    im_data = nifti.load(im)
    im_buffer = im_data.get_fdata()
    p_up = np.percentile(im_buffer, 99)
    im_buffer[im_buffer > p_up] = p_up
    p_down = np.percentile(im_buffer, 1)
    im_buffer[im_buffer < p_down] = p_down
    capped_buffer = (im_buffer - np.min(im_buffer)) / (np.max(im_buffer) - np.min(im_buffer))
    return nib.Nifti1Image(np.squeeze(capped_buffer), im_data.affine)


def write(sub_files, templates):
    names = []
    imgs = []
    cmaps = []
    interpolation_order = []
    select = [0, 0]
    if os.path.exists(sub_files.reference_volume):
        imgs.append(_cap_intensities(sub_files.reference_volume))
        cmaps.append(matplotlib.cm.gray)
        interpolation_order.append(1)
        names.append("Reference volume (T1)")

    if os.path.exists(sub_files.T2_reg):
        imgs.append(_cap_intensities(sub_files.T2_reg))
        cmaps.append(matplotlib.cm.gray)
        interpolation_order.append(1)
        names.append("T2 volume")
        imgs.append(_registration_overlay(sub_files.T2_reg))
        cmaps.append(matplotlib.cm.inferno)
        interpolation_order.append(1)
        names.append("T2 registration overlay")
        select[1] = len(imgs) - 1

    if os.path.exists(sub_files.template_coregistered):
        imgs.append(_final_overlay(sub_files.template_coregistered, labels=(6, 5), medfilter_size=5))
        cmaps.append(cmap_affine)
        interpolation_order.append(0)
        names.append("Coregistered template")
        select[1] = len(imgs) - 1

    if os.path.exists(sub_files.final_labels):
        imgs.append(_final_overlay(sub_files.final_labels))
        cmaps.append(cmap_final)
        interpolation_order.append(0)
        names.append("Tissue labels")
        select[1] = len(imgs) - 1
    brainsprite_helper.write_viewer(
        imgs,
        cmaps,
        interpolation_order,
        sub_files.summary_report,
        sub_files.summary_report,
        templates.jquery,
        templates.brainsprite,
        names=names,
        select=select,
    )
