import copy
import os

import nibabel as nib
import numpy as np
import pytest
from conftest import TESTDATA
from scipy import ndimage
from scipy.io import loadmat

from charm.pipeline.run import _check_q_and_s_form
from charm.segmentation import charm_utils
from charm.segmentation.samseg import ResolutionLevel
from charm.segmentation.segmentation_utils import (
    write_bias_corrected_images_and_segmentation,
)
from charm.utils import nifti


@pytest.fixture(scope="module")
def testernie_nii():
    fn = str(TESTDATA / "ernie_T1_ds5.nii.gz")
    return fn


@pytest.fixture(scope="module")
def testmni_nii():
    fn = str(TESTDATA / "MNI_test_ds5.nii.gz")
    return fn


@pytest.fixture(scope="module")
def testtemplate_nii(atlas_v1_1):
    return str(atlas_v1_1 / "template.nii.gz")


@pytest.fixture(scope="module")
def testaffinemesh_msh(atlas_v1_1):
    return str(atlas_v1_1 / "affine_no_neck.h5")


@pytest.fixture(scope="module")
def testcubenoise_nii():
    fn = str(TESTDATA / "cube_noise.nii.gz")
    return fn


@pytest.fixture(scope="module")
def testcube_nii():
    fn = str(TESTDATA / "cube.nii.gz")
    return fn


@pytest.fixture(scope="module")
def testcubeatlas_path():
    fn = str(TESTDATA / "cube_atlas")
    return fn


def _calc_dice(vol1, vol2):
    return np.sum(vol2[vol1]) * 2.0 / (np.sum(vol1) + np.sum(vol2))


def generate_label_arr(ndim=2):
    size = ndim * (10,)
    arr = np.zeros(size, dtype=np.uint16)
    arr[5:] = 1
    arr[0, :5] = 2
    arr[5, 5] = 3
    arr[7:, 7:] = 5
    return arr


# Every case triggers remapping.
test_label_unassigned_elements_inputs = (
    (2, [0, 1, 2, 3, 5], 3, None, 0),
    (3, None, 3, None, 1),
    (3, None, 3, [1], 0),
    (5, None, 3, None, 1),
    (5, None, 3, [1], 5),  # should show a warning
)


@pytest.mark.parametrize(
    "label_unassign, labels, window_size, ignore_labels, expected_label",
    test_label_unassigned_elements_inputs,
)
def test_label_unassigned_elements(
    label_unassign,
    labels,
    window_size,
    ignore_labels,
    expected_label,
):
    label_arr = generate_label_arr(3)
    expected_arr = label_arr.copy()
    expected_arr[label_arr == label_unassign] = expected_label
    np.testing.assert_allclose(
        expected_arr,
        charm_utils.label_unassigned_elements(label_arr, label_unassign, labels, window_size, ignore_labels),
    )


def test_sanlm(tmpdir, testernie_nii):
    denoised_scan = tmpdir.mkdir("denoised").join("denoised.nii.gz")
    input_scan = nifti.load(testernie_nii)
    input_data = input_scan.get_fdata()
    charm_utils._denoise_input_and_save(testernie_nii, denoised_scan)
    output_scan = nifti.load(denoised_scan)
    output_data = output_scan.get_fdata()
    assert input_data.var() > output_data.var()


def test_mni_affine(tmpdir, testmni_nii):
    trans_scan_name = tmpdir.mkdir("shifted").join("shifted_MNI.nii.gz")
    input_scan = nifti.load(testmni_nii)
    trans_mat = np.eye(4)
    trans_mat[:3, 3] = -10
    trans_affine = trans_mat @ input_scan.affine
    trans_mni = nib.Nifti1Image(input_scan.get_fdata(), trans_affine)
    nib.save(trans_mni, trans_scan_name)
    trans_settings = {
        "translation_scale": -100,
        "max_iter": 10,
        "num_histogram_bins": 64,
        "shrink_factors": [0],
        "bg_value": 0,
        "smoothing_factors": [4.0],
        "center_of_mass": True,
        "samp_factor": 1.0,
        "intepolator": "b",
    }
    RAS2LPS = np.diag([-1, -1, 1, 1])
    estimated_trans_mat = charm_utils._init_atlas_affine(str(trans_scan_name), testmni_nii, trans_settings)
    np.testing.assert_allclose(trans_mat, RAS2LPS @ estimated_trans_mat @ RAS2LPS, atol=1e-2, rtol=1e-6)


def test_atlas_affine(tmpdir, testmni_nii, testtemplate_nii, testaffinemesh_msh):
    trans_scan_name = tmpdir.mkdir("shifted").join("shifted_MNI.nii.gz")
    template_coregistered_name = tmpdir.join("template_coregistered.mgz")
    input_scan = nifti.load(testmni_nii)
    trans_mat = np.eye(4)
    trans_mat[:3, 3] = -10
    trans_affine = trans_mat @ input_scan.affine
    trans_mni = nib.Nifti1Image(input_scan.get_fdata(), trans_affine)
    nib.save(trans_mni, trans_scan_name)
    init_atlas_settings = {
        "affine_scales": [[1.0, 1.0, 1.0]],
        "affine_rotations": [0],
        "affine_horizontal_shifts": [0],
        "affine_vertical_shifts": [0],
        "neck_search_bounds": [0, 0],
        "downsampling_factor_affine": 1.0,
    }
    charm_utils._register_atlas_to_input_affine(
        str(trans_scan_name),
        testtemplate_nii,
        testaffinemesh_msh,
        testaffinemesh_msh,
        testaffinemesh_msh,
        str(tmpdir),
        str(template_coregistered_name),
        init_atlas_settings,
        None,
        True,
        init_transform=None,
        world_to_world_transform_matrix=None,
        scaling_center=[0, 0, 0],
        k_values=[100],
    )

    matrices = loadmat(str(tmpdir.join("coregistrationMatrices.mat")))
    w2w = matrices["worldToWorldTransformMatrix"]
    # Looser tolerances than for the MNI to MNI registration above.
    np.testing.assert_allclose(trans_mat[:2, 3], w2w[:2, 3], rtol=5e-2, atol=1)
    np.testing.assert_allclose(trans_mat[:3, :3], w2w[:3, :3], rtol=5e-2, atol=5e-2)


def test_T1T2(tmpdir, testmni_nii):
    trans_scan_name = tmpdir.mkdir("shifted").join("shifted_MNI.nii.gz")
    registered_scan_name = tmpdir.join("registered.nii.gz")
    input_scan = nifti.load(testmni_nii)
    trans_mat = np.eye(4)
    trans_mat[:3, 3] = -10
    trans_affine = trans_mat @ input_scan.affine
    trans_mni = nib.Nifti1Image(input_scan.get_fdata(), trans_affine)
    nib.save(trans_mni, trans_scan_name)
    charm_utils._registerT1T2(str(trans_scan_name), testmni_nii, str(registered_scan_name))
    reg_scan = nifti.load(str(registered_scan_name))
    assert (np.corrcoef(reg_scan.get_fdata().flatten(), trans_mni.get_fdata().flatten()))[0, 1] > 0.99


def test_largest_components():
    se = ndimage.generate_binary_structure(3, 3)
    test_array = np.zeros((5, 5, 5))
    test_array[0:2, 0:2, 0:2] = 1
    test_array[3:5, 3:5, 3:5] = 2

    assert charm_utils._get_largest_components(test_array, se).sum() == 16
    assert charm_utils._get_largest_components(test_array, se, vol_limit=9).sum() == 0
    assert charm_utils._get_largest_components(test_array, se, num_limit=1).sum() == 8


def test_segmentation(tmpdir, testcube_nii, testcubenoise_nii, testcubeatlas_path):
    seg_dir = tmpdir.mkdir("segmentation")

    seg_settings = {
        "downsampling_targets": 1.0,
        "bias_kernel_width": 100,
        "background_mask_sigma": 1.0,
        "background_mask_threshold": 0.001,
        "mesh_stiffness": 0.1,
        "diagonal_covariances": False,
    }

    param_estimates = charm_utils._estimate_parameters(
        str(seg_dir),
        os.path.join(testcubeatlas_path, "template.nii.gz"),
        testcubeatlas_path,
        [testcubenoise_nii],
        seg_settings,
        os.path.join(testcubeatlas_path, "sharedGMMParameters.txt"),
        levels=[
            ResolutionLevel(
                os.path.join(testcubeatlas_path, "atlas.h5"),
                1.0,
                estimate_bias_field=False,
            )
        ],
    )

    bias_corr = os.path.join(str(seg_dir), "bias_temp.nii.gz")
    seg = os.path.join(str(seg_dir), "seg.nii.gz")
    mock_tissue_settings = {"segmentation_tissues": {"CSF": -1}}
    write_bias_corrected_images_and_segmentation(
        [bias_corr],
        seg,
        param_estimates,
        mock_tissue_settings,
        1,
        os.path.join(testcubeatlas_path, "template.nii.gz"),
    )

    orig_cube = nifti.load(testcube_nii)
    est_cube = nifti.load(seg)
    dice = _calc_dice(orig_cube.get_fdata() == 1, est_cube.get_fdata() == 1)
    print("Dice score: " + str(dice))
    assert dice > 0.95


def test_update_labeling_from_cortical_surfaces_():
    """Does not test protecting CSF/bone from becoming gray matter.

    The image and the regions are crafted by hand, so change them with care.
    """
    # 2D, so the result can be plotted and checked by eye.
    shape = np.array([51, 51])
    protect = {"gm_to_csf": [10], "wm_to_gm": [11], "gm_to_wm": [12], "csf_to_gm": [13]}
    tissue_mapping = {
        "WM": 1,
        "GM": 2,
        "CSF": 3,
        "Compact_bone": 4,
        "Spongy_bone": 5,
        "Blood": 6,
    }

    center = np.array((shape - 1) / 2, dtype=int)
    grid = np.meshgrid(*[np.arange(-c, c + 1) for c in center])
    grid = np.array(grid)
    manhattan_dist = np.abs(grid).max(0)
    max_dist = manhattan_dist[0, 0]
    radius = {
        "white": 0.4 * max_dist,
        "pial": 0.55 * max_dist,
        "csf": 0.7 * max_dist,
        "bone": 0.8 * max_dist,
    }

    wm_surf_mask = manhattan_dist <= radius["white"]
    gm_surf_mask = manhattan_dist <= radius["pial"]

    sm_labeling = np.zeros(shape, dtype=int)
    sm_labeling[manhattan_dist <= radius["bone"]] = tissue_mapping["Compact_bone"]
    sm_labeling[manhattan_dist <= radius["csf"]] = tissue_mapping["CSF"]
    sm_labeling[gm_surf_mask] = tissue_mapping["GM"]
    sm_labeling[wm_surf_mask] = tissue_mapping["WM"]

    sm_labeling[25, sm_labeling[25] == tissue_mapping["CSF"]] = tissue_mapping["Blood"]
    ground_truth = sm_labeling.copy()

    WM_GM_to_Blood = np.isin(sm_labeling[13:19, 13:19], (tissue_mapping["WM"], tissue_mapping["GM"]))
    sm_labeling[13:19, 13:19][WM_GM_to_Blood] = tissue_mapping["Blood"]

    protect_stuff = manhattan_dist <= 0.2 * max_dist

    x0n = protect_stuff & (grid[0] < -1.0)
    x0p = protect_stuff & (grid[0] > 1.0)
    x1n = protect_stuff & (grid[1] < -1.0)
    x1p = protect_stuff & (grid[1] > 1.0)

    wm_outside_white = np.zeros(shape, dtype=bool)
    wm_outside_white[20, sm_labeling[20] == tissue_mapping["GM"]] = True
    gm_outside_pial = np.zeros(shape, dtype=bool)
    gm_outside_pial[30, sm_labeling[30] == tissue_mapping["CSF"]] = True

    gm_inside_white = x0p & x1n  # e.g., subcortical
    csf_inside_white = x0n & x1p  # e.g., ventricle

    # These labels must match `protect`.
    fs_labeling = np.zeros_like(sm_labeling)
    fs_labeling[gm_outside_pial] = 10  # gm to csf
    fs_labeling[wm_outside_white] = 11  # wm to gm
    fs_labeling[gm_inside_white] = 12  # gm to wm
    fs_labeling[csf_inside_white] = 13  # csf to gm

    sm_labeling = sm_labeling.ravel()

    rng = np.random.default_rng(seed=0)
    x = rng.permutation(np.where(sm_labeling == 1)[0])
    wrong_wm_as_gm = x[:20]
    x = rng.permutation(np.where(sm_labeling == 2)[0])
    wrong_gm_as_wm = x[:20]
    x = rng.permutation(np.where(sm_labeling == 3)[0])
    wrong_csf_as_gm = x[:20]
    x = rng.permutation(np.where(sm_labeling == 2)[0])
    wrong_gm_as_csf = x[:20]

    sm_labeling[wrong_wm_as_gm] = tissue_mapping["GM"]
    sm_labeling[wrong_gm_as_wm] = tissue_mapping["WM"]
    sm_labeling[wrong_csf_as_gm] = tissue_mapping["GM"]
    sm_labeling[wrong_gm_as_csf] = tissue_mapping["CSF"]

    sm_labeling = sm_labeling.reshape(shape)
    fs_labeling = fs_labeling.reshape(shape)

    sm_labeling[wm_outside_white] = tissue_mapping["WM"]
    sm_labeling[gm_outside_pial] = tissue_mapping["GM"]
    sm_labeling[gm_inside_white] = tissue_mapping["GM"]
    sm_labeling[csf_inside_white] = tissue_mapping["CSF"]

    sm_labeling_clean = ground_truth.copy()
    sm_labeling_clean[wm_outside_white] = tissue_mapping["WM"]
    sm_labeling_clean[gm_outside_pial] = tissue_mapping["GM"]
    sm_labeling_clean[gm_inside_white] = tissue_mapping["GM"]
    sm_labeling_clean[csf_inside_white] = tissue_mapping["CSF"]

    sm_labeling_orig = sm_labeling.copy()

    charm_utils.update_labeling_from_cortical_surfaces_(
        sm_labeling, fs_labeling, wm_surf_mask, gm_surf_mask, protect, tissue_mapping
    )
    with_protect = sm_labeling

    sm_labeling = sm_labeling_orig.copy()

    protect = {"gm_to_csf": [], "wm_to_gm": [], "gm_to_wm": [], "csf_to_gm": []}
    charm_utils.update_labeling_from_cortical_surfaces_(
        sm_labeling, fs_labeling, wm_surf_mask, gm_surf_mask, protect, tissue_mapping
    )
    without_protect = sm_labeling

    assert np.all(with_protect[gm_outside_pial] == tissue_mapping["GM"])
    assert np.all(with_protect[wm_outside_white] == tissue_mapping["WM"])
    assert np.all(with_protect[gm_inside_white] == tissue_mapping["GM"])
    assert np.all(with_protect[csf_inside_white] == tissue_mapping["CSF"])

    assert np.all(without_protect[gm_outside_pial] == tissue_mapping["CSF"])
    assert np.all(without_protect[wm_outside_white] == tissue_mapping["GM"])
    assert np.all(without_protect[gm_inside_white] == tissue_mapping["WM"])
    assert np.all(without_protect[csf_inside_white] == tissue_mapping["WM"])

    # the surface masks are opened in `update_labeling_from_cortical_surfaces_`
    wm_surf_mask_open = ndimage.binary_opening(wm_surf_mask, iterations=1)
    gm_surf_mask_open = ndimage.binary_opening(gm_surf_mask, iterations=1)

    assert np.all(without_protect[wm_surf_mask_open] == tissue_mapping["WM"])
    assert np.all(without_protect[gm_surf_mask_open & ~wm_surf_mask_open] == tissue_mapping["GM"])

    comp = with_protect == sm_labeling_clean

    # Opening the surface masks changes these voxels.
    smooth_errors_OK = (wm_surf_mask != wm_surf_mask_open) | (gm_surf_mask != gm_surf_mask_open)
    # Dilating the protection masks changes these.
    dil_errors_OK = (19, 22), (36, 32)

    OK_error_img = np.zeros_like(comp)
    OK_error_img[smooth_errors_OK] = True
    OK_error_img[dil_errors_OK] = True

    assert np.all(comp | OK_error_img)


def test_q_and_sform():
    tmp_vol = np.zeros((3, 3, 3))
    tmp_affine = np.eye(4)
    tmp_im = nib.Nifti1Image(tmp_vol, tmp_affine)
    # By default nibabel sets only the sform, with code 2.
    tmp_im.set_qform(tmp_affine, code=0)

    with pytest.raises(ValueError):
        _check_q_and_s_form(tmp_im, force_qform=False, force_sform=False)

    # Noise within the tolerance passes.
    tmp_im.set_qform(tmp_affine + 1e-7, code=tmp_im.get_sform(coded=True)[1])
    _check_q_and_s_form(tmp_im, force_qform=False, force_sform=False)

    tmp_im.set_qform(tmp_affine + 1e-4)
    with pytest.raises(ValueError):
        _check_q_and_s_form(tmp_im, force_qform=False, force_sform=False)

    tmp_im_copy = copy.deepcopy(tmp_im)
    tmp_im_copy.set_qform(tmp_im.get_qform(), code=1)
    scan_forced = _check_q_and_s_form(tmp_im, force_qform=True, force_sform=False)
    assert np.equal(scan_forced.get_qform(coded=True)[1], scan_forced.get_sform(coded=True)[1]), (
        "Q- and S-form codes should be the same but they're not!"
    )
    assert np.allclose(scan_forced.get_qform(), scan_forced.get_sform()), (
        "Q- and S-form matrices should be the same but they're not!"
    )

    tmp_im_copy = copy.deepcopy(tmp_im)
    tmp_im_copy.set_sform(tmp_im.get_sform(), code=1)
    tmp_im_copy.set_qform(tmp_im.get_qform(), code=0)
    scan_forced = _check_q_and_s_form(tmp_im, force_qform=False, force_sform=True)
    assert np.equal(scan_forced.get_qform(coded=True)[1], scan_forced.get_sform(coded=True)[1]), (
        "Q- and S-form codes should be the same but they're not!"
    )
    assert np.allclose(scan_forced.get_qform(), scan_forced.get_sform()), (
        "Q- and S-form matrices should be the same but they're not!"
    )

    scan_forced.set_sform(scan_forced.get_sform(), code=0)
    scan_forced.set_qform(scan_forced.get_qform() + 0.5)
    with pytest.raises(ValueError):
        _check_q_and_s_form(scan_forced, force_qform=False, force_sform=True)
