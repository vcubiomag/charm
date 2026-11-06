import os
import shutil
import tempfile

import nibabel as nib
import nibabel.processing
import numpy as np
import numpy.typing as npt
from scipy import ndimage
from scipy.ndimage import gaussian_filter, label

from charm.logging import logger
from charm.mesh.io import load_subject_surfaces
from charm.segmentation import gems, segmentation_utils
from charm.segmentation._sanlm import sanlm
from charm.segmentation.affine import AffineRegistration, InitializationOptions
from charm.segmentation.io import read_shared_gmm_parameters
from charm.segmentation.samseg import (
    ModelSpecification,
    OptimizationOptions,
    ResolutionLevel,
    Samseg,
)
from charm.subject import SubjectFiles
from charm.surface.brain_surface import mask_from_surface
from charm.utils import nifti
from charm.utils._thickness import calc_thickness
from charm.utils.morphology import (
    dilate_cross,
    dilate_cube,
    erode_cross,
    erode_cube,
    fill_holes,
    open_cross,
    open_cube,
)
from charm.utils.transformations import resample_vol


def _register_atlas_to_input_affine(
    T1,
    template_file_name,
    affine_mesh_collection_name,
    mesh_level1,
    mesh_level2,
    save_path,
    template_coregistered_name,
    init_atlas_settings,
    neck_tissues,
    noneck,
    init_transform=None,
    world_to_world_transform_matrix=None,
    scaling_center=(0.0, -100.0, 0.0),
    k_values=(20.0, 10.0, 5.0),
):
    scales = init_atlas_settings["affine_scales"]
    thetas = init_atlas_settings["affine_rotations"]
    horizontal_shifts = init_atlas_settings["affine_horizontal_shifts"]
    vertical_shifts = init_atlas_settings["affine_vertical_shifts"]
    thetas_rad = [theta * np.pi / 180 for theta in thetas]
    neck_search_bounds = init_atlas_settings["neck_search_bounds"]
    ds_factor = init_atlas_settings["downsampling_factor_affine"]

    affine = AffineRegistration(T1, affine_mesh_collection_name, template_file_name)

    options = InitializationOptions(
        pitch_angles=thetas_rad,
        scales=scales,
        scaling_center=scaling_center,
        horizontal_table_shifts=horizontal_shifts,
        vertical_table_shifts=vertical_shifts,
    )

    (
        image_to_image_transform,
        world_to_world_transform,
        optimization_summary,
    ) = affine.register(
        world_to_world=world_to_world_transform_matrix,
        init_transform=init_transform,
        options=options,
        target_downsampled_voxel_spacing=ds_factor,
        noneck=noneck,
        ks=k_values,
    )

    affine.save_results(
        T1,
        template_file_name,
        save_path,
        template_coregistered_name,
        image_to_image_transform,
        world_to_world_transform,
    )
    if world_to_world_transform_matrix is None:
        logger.info("Template registration summary.")
        logger.info(
            f"Number of Iterations: {optimization_summary['numberOfIterations']:d}, "
            f"Cost: {optimization_summary['cost']:f}\n"
        )

    if not noneck:
        logger.info("Adjusting neck.")
        adjusted = affine.adjust_neck(
            T1,
            template_coregistered_name,
            mesh_level1,
            mesh_level2,
            neck_search_bounds,
            neck_tissues,
            downsampling_target=2.0,
        )
        if adjusted:
            logger.info("Neck adjustment done.")
            return
    else:
        logger.info("No neck, copying meshes over.")

    directory = os.path.dirname(template_coregistered_name)
    shutil.copy(mesh_level1, os.path.join(directory, "atlas_level1.h5"))
    shutil.copy(mesh_level2, os.path.join(directory, "atlas_level2.h5"))


def _denoise_input_and_save(input_name, output_name):
    input_raw = nifti.load(input_name)
    img = input_raw.get_fdata()
    # Some images have an extra unit dimension
    img = img.squeeze()
    img_smoothed = sanlm(img, 3, 1)
    output_smoothed = nib.Nifti1Image(img_smoothed, input_raw.affine)
    nifti.save(output_smoothed, output_name)


def _init_atlas_affine(t1_scan, mni_template, affine_settings):
    registerer = gems.KvlAffineRegistration(
        affine_settings["translation_scale"],
        affine_settings["max_iter"],
        affine_settings["num_histogram_bins"],
        affine_settings["shrink_factors"],
        affine_settings["bg_value"],
        affine_settings["smoothing_factors"],
        affine_settings["center_of_mass"],
        affine_settings["samp_factor"],
        "b",
    )
    registerer.read_images(str(t1_scan), str(mni_template))
    registerer.initialize_transform()
    registerer.register()
    trans_mat = registerer.get_transformation_matrix()
    logger.info(trans_mat)
    # ITK returns the matrix mapping the fixed image to the moving image
    return np.linalg.inv(trans_mat)


def _estimate_parameters(
    path_to_segment_folder,
    template_coregistered_name,
    path_to_atlas_folder,
    input_images,
    segment_settings,
    gmm_parameters,
    levels=None,
    parameter_filename=None,
):
    if levels is None:
        ds_targets = segment_settings["downsampling_targets"]
        levels = [
            ResolutionLevel(os.path.join(path_to_segment_folder, name), target)
            for name, target in zip(("atlas_level1.h5", "atlas_level2.h5"), ds_targets)
        ]

    specification = ModelSpecification.from_atlas(
        path_to_atlas_folder,
        atlas_file_name=levels[-1].atlas_file_name,
        bias_field_smoothing_kernel_size=segment_settings["bias_kernel_width"],
        brain_masking_smoothing_sigma=segment_settings["background_mask_sigma"],
        brain_masking_threshold=segment_settings["background_mask_threshold"],
        k=segment_settings["mesh_stiffness"],
        use_diagonal_covariances=segment_settings["diagonal_covariances"],
        shared_gmm_parameters=read_shared_gmm_parameters(gmm_parameters),
    )

    logger.info("Starting segmentation.")
    samseg = Samseg(
        input_images,
        path_to_segment_folder,
        template_coregistered_name,
        specification,
        OptimizationOptions(levels),
    )
    samseg.preprocess()
    samseg.fit_model()

    for level, item in enumerate(samseg.optimization_summary):
        logger.info(f"atlasRegistrationLevel{level:d} {item['numberOfIterations']:d} {item['perVoxelCost']:f}\n")

    return samseg.fitted_model(parameter_filename)


def _post_process_segmentation(
    bias_corrected_image_names,
    tissue_settings,
    csf_factor,
    model,
    transformed_template_name,
    affine_atlas,
    before_morpho_name,
    upper_mask,
    debug=False,
):
    """Returns the cleaned upsampled tissues and the upsampled images, for the caller to write out."""
    logger.info("Upsampling bias corrected images.")
    upsampled_images = []
    for bias_corrected in bias_corrected_image_names:
        corrected_input = nifti.load(bias_corrected)
        resampled_input, new_affine, _ = resample_vol(corrected_input.get_fdata(), corrected_input.affine, 0.5, order=1)
        upsampled_images.append(nib.Nifti1Image(resampled_input, new_affine))

    # segment_upsampled reads from files, which needn't be compressed for it
    with tempfile.TemporaryDirectory() as temporary_directory:
        upsampled_image_names = []
        for number, image in enumerate(upsampled_images):
            name = os.path.join(temporary_directory, f"upsampled_{number}.nii")
            nifti.save(nib.Nifti1Image(image.get_fdata(dtype=np.float32), image.affine), name)
            upsampled_image_names.append(name)
        upsampled_tissues, upper_part = segmentation_utils.segment_upsampled(
            upsampled_image_names,
            tissue_settings,
            model,
            transformed_template_name,
            affine_atlas,
            csf_factor,
        )

    affine_upsampled = upsampled_images[0].affine
    if debug:
        nifti.save(nib.Nifti1Image(upsampled_tissues, affine_upsampled), before_morpho_name)

    upper_part_im = nib.Nifti1Image(upper_part.astype(np.int16), affine_upsampled)
    upper_part_im.set_data_dtype(np.int16)
    nifti.save(upper_part_im, upper_mask)
    del upper_part_im

    upsampled_tissues = _morphological_operations(upsampled_tissues, upper_part, tissue_settings["simnibs_tissues"])
    return upsampled_tissues, upsampled_images


def _clean_brain(label_img, tissues, unass, se, se_n, vol_limit=10):
    brain = (label_img == tissues["WM"]) | (label_img == tissues["GM"])
    dil = dilate_cube(_get_largest_components(erode_cross(brain), se, vol_limit))

    unass |= brain ^ dil

    csf = label_img == tissues["CSF"]
    brain_csf = brain | csf
    dil = dilate_cube(_get_largest_components(erode_cube(brain_csf), se_n, vol_limit=80))
    unass |= csf & ~dil
    return brain_csf


def _get_skull(label_img, brain_csf, tissues, se_n, num_iter=2, vol_limit=50):
    bone = (label_img == tissues["Compact_bone"]) | (label_img == tissues["Spongy_bone"])
    veins = label_img == tissues["Blood"]
    air_pockets = label_img == tissues["Air_pockets"]
    scalp = label_img == tissues["Scalp"]
    muscle = label_img == tissues["Muscle"]
    eyes = label_img == tissues["Eyes"]

    def largest_opened(mask):
        return dilate_cube(
            _get_largest_components(erode_cube(fill_holes(mask), num_iter), se_n, vol_limit),
            num_iter,
        )

    # Use scalp to clean out noisy skull bits within the scalp
    skull_outer = largest_opened(brain_csf | bone | veins | air_pockets)
    skull_inner = largest_opened(bone | scalp | air_pockets | muscle | eyes)

    dil = bone & skull_outer & skull_inner
    return bone, skull_outer, dil


def _clean_veins(label_img, unass, tissues, se_n, num_iter=1, vol_limit=10):
    veins = label_img == tissues["Blood"]
    dil = dilate_cube(_get_largest_components(erode_cube(veins, num_iter), se_n, vol_limit), num_iter)
    unass |= dil ^ veins


def _clean_eyes(label_img, unass, tissues, se_n, num_iter=1, vol_limit=10):
    eyes = label_img == tissues["Eyes"]
    dil = dilate_cube(_get_largest_components(erode_cube(eyes, num_iter), se_n, vol_limit), num_iter)
    unass |= dil ^ eyes


def _clean_muscles(label_img, unass, tissues, num_iter=1):
    muscle = label_img == tissues["Muscle"]
    unass |= open_cube(muscle, num_iter) ^ muscle


def _clean_scalp(label_img, unass, skull_outer, tissues, se_n, num_iter=2, num_limit=1):
    scalp = label_img == tissues["Scalp"]
    eyes = label_img == tissues["Eyes"]
    muscle = label_img == tissues["Muscle"]
    head = scalp | skull_outer | eyes | muscle
    dil = dilate_cube(_get_largest_components(erode_cube(head, num_iter), se_n, num_limit), num_iter)
    unass |= scalp & ~dil


def _ensure_csf(label_img, tissues, upper_part, num_iter1=1, num_iter2=6):
    brain_gm = label_img == tissues["GM"]
    C_BONE = label_img == tissues["Compact_bone"]
    S_BONE = label_img == tissues["Spongy_bone"]
    U_SKIN = (label_img == tissues["Scalp"]) & upper_part

    brain_dilated = dilate_cube(brain_gm, num_iter1)
    overlap = brain_dilated & (C_BONE | S_BONE | U_SKIN)
    label_img[overlap] = tissues["CSF"]

    S_BONE = label_img == tissues["Spongy_bone"]

    CSF_brain = brain_gm | (label_img == tissues["CSF"])
    CSF_brain_dilated = dilate_cube(CSF_brain, num_iter1)
    spongy_csf = CSF_brain_dilated & S_BONE
    upper_part = erode_cube(upper_part, num_iter2)
    skin_csf = CSF_brain_dilated & (label_img == tissues["Scalp"]) & upper_part
    label_img[spongy_csf] = tissues["Compact_bone"]
    label_img[skin_csf] = tissues["Compact_bone"]


def _ensure_skull(label_img, tissues, num_iter=1):
    C_BONE = label_img == tissues["Compact_bone"]
    S_BONE = label_img == tissues["Spongy_bone"]
    bone = C_BONE | S_BONE
    label_img[bone & ~erode_cube(bone, num_iter)] = tissues["Compact_bone"]
    label_img[label_img == tissues["Air_pockets"]] = 0


def _morphological_operations(label_img, upper_part, simnibs_tissues):
    """Smooth the labeling and remove noise, ensure a CSF layer between GM and skull, and make the outer bone layer
    compact bone."""
    se = ndimage.generate_binary_structure(3, 3)
    se_n = ndimage.generate_binary_structure(3, 1)
    unass = np.zeros_like(label_img, dtype=bool)
    brain_csf = _clean_brain(label_img, simnibs_tissues, unass, se, se_n)
    bone, skull_outer, dil = _get_skull(label_img, brain_csf, simnibs_tissues, se_n)
    # Protect thin areas that would be removed by erosion
    bone_thickness = calc_thickness(dil)

    dil |= bone & (bone_thickness < 3.5) & (bone_thickness > 0)
    del bone_thickness

    unass |= dil ^ bone
    del bone, dil

    logger.info("Cleaning tissues")
    _clean_veins(label_img, unass, simnibs_tissues, se_n)
    _clean_eyes(label_img, unass, simnibs_tissues, se_n)
    _clean_muscles(label_img, unass, simnibs_tissues)
    _clean_scalp(label_img, unass, skull_outer, simnibs_tissues, se_n)

    logger.info("Labeling unassigned voxels")
    # The labeling is uint16, so code the unassigned voxels as 2**16 - 2
    label_unassign = 65534
    label_img[unass] = label_unassign
    simnibs_tissues["BG"] = 0
    label_img = label_unassigned_elements(label_img, label_unassign, list(simnibs_tissues.values()) + [label_unassign])

    logger.info("Ensuring CSF between gray matter and skull/blood")
    _ensure_csf(label_img, simnibs_tissues, upper_part)

    logger.info("Ensuring that the outer skull is compact bone")
    _ensure_skull(label_img, simnibs_tissues)

    return label_img


def generate_gaussian_kernel(i, ndim, sigma=1, zero_center=False):
    """Generate a gaussian kernel of size `i` in `ndim` dimensions."""
    assert i % 2 == 1, "Window size must be odd"
    center = ndim * (np.floor(i / 2).astype(int),)
    kernel = np.zeros(ndim * (i,), dtype=np.float32)
    kernel[center] = 1
    kernel = gaussian_filter(kernel, sigma)
    kernel /= kernel[center]
    if zero_center:
        kernel[center] = 0
    return kernel


def label_unassigned_elements(label_arr, label_unassign, labels=None, window_size=3, ignore_labels=None) -> np.ndarray:
    """Label unassigned elements in `label_arr`. For each unassigned element,
    find its neighbors within a certain `window_size`, weigh these according
    to their euclidean distance (Gaussian kernel) and assign the label with the
    highest weight.

    PARAMETERS
    ----------
    label_arr : ndarray
        The array
    label_unassign : int
        The label of the unassigned elements.
    labels : array-like | None
        The labels in `label_arr`. If None (default), will be inferred from the
        array using `np.unique`.
    window_size : int
        The size of the neighborhood around each element to consider when
        relabeling.
    ignore_labels : array-like | None
        Do not use these labels when labeling unassigned elements.

    RETURNS
    -------
    labeling : ndarray
        The relabeled array.
    """

    # Finding the indices of the unassigned voxels is slow with fortran ordered
    # arrays
    labeling = label_arr.T if label_arr.flags["F_CONTIGUOUS"] else label_arr

    pad_width = np.floor(window_size / 2).astype(int)
    kernel = generate_gaussian_kernel(window_size, labeling.ndim, zero_center=True)
    window = kernel.shape
    kernel = kernel.ravel()

    labels = np.unique(labeling) if labels is None else np.unique(labels)
    labels = labels.astype(labeling.dtype)
    n_labels = labels.size
    assert label_unassign in labels, "`label_unassign` is not in the provided `labels`!"

    if ignore_labels is None:
        ignore_labels = [label_unassign]
    else:
        assert all(i in labels for i in ignore_labels)
        ignore_labels = ignore_labels.copy()
        ignore_labels = ignore_labels + [label_unassign]  # avoid inplace

    # Map the labels to a contiguous range (e.g. [0, 1, 2, 3], not [0, 1, 10, 15])
    is_continuous_labels = labels[-1] - labels[0] + 1 == n_labels
    if is_continuous_labels:
        continuous_labels = labels
        mapped_ignore_labels = np.array(ignore_labels)
        labeling = labeling.copy()  # ensure that we do not modify the input
    else:
        continuous_labels = np.arange(n_labels)
        mapper = np.zeros(labels[-1] + 1, dtype=labeling.dtype)
        mapper[labels] = continuous_labels
        labeling = mapper[labeling]
        mapped_ignore_labels = mapper[ignore_labels]

    is_unassign = np.nonzero(labeling == mapped_ignore_labels[-1])
    while (n_unassign := is_unassign[0].size) > 0:
        labeling_view = np.lib.stride_tricks.sliding_window_view(np.pad(labeling, pad_width, "symmetric"), window)
        unassign_window = labeling_view[is_unassign].reshape(-1, kernel.size)

        # The sum of the kernel over the neighbors with each label
        n_rows = continuous_labels[-1] + 1
        offsets = continuous_labels[0]
        weights = (
            np.bincount(
                (unassign_window.astype(np.int64) - offsets).ravel()
                + np.repeat(np.arange(n_unassign) * (n_rows - offsets), kernel.size),
                np.tile(kernel, n_unassign),
                minlength=n_unassign * (n_rows - offsets),
            )
            .reshape(n_unassign, n_rows - offsets)
            .T
        )
        weights[np.isin(continuous_labels, mapped_ignore_labels)] = 0

        # Ties go to the lowest label. This is arbitrary, but it is what argmax does.
        new_label = continuous_labels[weights.argmax(0)]
        valid = weights.sum(0) > 0
        invalid = ~valid

        labeling[tuple(i[valid] for i in is_unassign)] = new_label[valid]
        is_unassign = tuple(i[invalid] for i in is_unassign)

        if is_unassign[0].size == n_unassign:
            logger.warning(
                "Some elements could not be labeled (probably because they are surrounded by labels in `ignore_labels`)"
            )
            break

    if not is_continuous_labels:
        labeling = labels[labeling]
    labeling = labeling.T if label_arr.flags["F_CONTIGUOUS"] else labeling

    return labeling


def _get_largest_components(vol, se, vol_limit=0, num_limit=-1):
    """Get the n largest components from a volume.

    PARAMETERS
    ----------
    vol : ndarray
        Image volume. A dimX x dimY x dimZ array containing the image data.
    se : ndarray
        Structuring element to use when detecting components (i.e. setting the
        connectivity which defines a component).
    vol_limit : int
        Only retain components with more voxels than this.
    num_limit : int
        Number of (largest) components to retain, all of them if -1.

    RETURNS
    ----------
    Components : ndarray
        Binary dimX x dimY x dimZ array where entries corresponding to retained
        components are True and the remaining entries are False.
    """
    vol_lbl = label(vol, se)[0]
    region_size = np.bincount(vol_lbl.ravel())
    region_size[0] = 0  # disregard background (label=0)
    labels = np.flatnonzero(region_size > vol_limit)

    if num_limit != -1:
        labels = labels[np.argsort(region_size[labels])[::-1][:num_limit]]

    is_retained = np.zeros(region_size.size, dtype=bool)
    is_retained[labels] = True
    return is_retained[vol_lbl]


def _registerT1T2(fixed_image, moving_image, output_image):
    registerer = gems.KvlRigidRegistration()
    registerer.read_images(fixed_image, moving_image)
    registerer.initialize_transform()
    registerer.register()
    registerer.write_out_result(output_image)

    # Registration writes doubles. Cast to float32 and copy the header of the fixed image.
    if os.path.exists(output_image):
        T2_reg = nifti.load(output_image)
        fixed_tmp = nifti.load(fixed_image)
        T2_data = T2_reg.get_fdata().astype(np.float32)
        T2_im = nib.Nifti1Image(T2_data, fixed_tmp.affine)
        nifti.save(T2_im, output_image)


def read_freesurfer_lut(filename):
    value, label, r, g, b, a = np.loadtxt(filename, dtype=str, unpack=True)
    value = value.astype(int)
    ctab = np.stack((r, g, b, a), axis=1).astype(int)
    return value, label, ctab


def update_labeling_from_cortical_surfaces(
    m2m: SubjectFiles,
    protect: dict[str, list],
    tissue_mapping: dict[str, int],
):
    """Relabel the tissue labeling using the estimated white and pial surfaces. For example, gray matter outside the
    pial surface becomes CSF. Subcortical structures and the cerebellum are protected from relabeling.

    Parameters
    ----------
    m2m : SubjectFiles
        SubjectFiles object of the subject directory.
    protect : dict
        The following keys are required (as strings)
            gm_to_csf
                Tissues to protect when relabeling gray matter outside pial
                surface to CSF
            wm_to_gm
                Tissues to protect when relabeling white matter outside white
                matter surface to gray matter.
            gm_to_wm
                Tissues to protect when relabeling gray matter inside white
                matter surface to white matter.
            csf_to_gm
                Tissues to protect when relabeling CSF inside pial surface to
                gray matter.
    tissue_mapping : dict
        The mapping of SimNIBS tissue names to numbers.
    """

    sm_label_img = nifti.load(m2m.tissue_labeling_upsampled)
    sm_label_data = np.asanyarray(sm_label_img.dataobj)

    # The full SAMSEG labeling decides which regions to protect
    fs_label_img = nifti.load(m2m.labeling)
    fs_label_img = nibabel.processing.resample_from_to(fs_label_img, sm_label_img, order=0)
    fs_label_data = np.asanyarray(fs_label_img.dataobj)

    white = load_subject_surfaces(m2m, "white")
    white = white["lh"].join_mesh(white["rh"])
    white_surf_mask = mask_from_surface(white.nodes[:], white.elm[:, :3] - 1, sm_label_img.affine, sm_label_img.shape)
    pial = load_subject_surfaces(m2m, "pial")
    pial = pial["lh"].join_mesh(pial["rh"])
    pial_surf_mask = mask_from_surface(pial.nodes[:], pial.elm[:, :3] - 1, sm_label_img.affine, sm_label_img.shape)

    sm_label_data = update_labeling_from_cortical_surfaces_(
        sm_label_data,
        fs_label_data,
        white_surf_mask,
        pial_surf_mask,
        protect,
        tissue_mapping,
    )

    sm_label_img_updated = nib.Nifti1Image(sm_label_data, sm_label_img.affine)
    nifti.save(sm_label_img_updated, m2m.tissue_labeling_upsampled)


def update_labeling_from_cortical_surfaces_(
    label_image: npt.NDArray,
    protect_labeling: npt.NDArray,
    white_surf_mask: npt.NDArray,
    pial_surf_mask: npt.NDArray,
    protect: dict[str, list[int]],
    tissue_mapping: dict[str, int],
):
    """The behavior of this function for WM and GM in bone, skin etc. is not
    well defined.

    Parameters
    ----------
    label_image
        Label image to be corrected.
    protect_labeling
        Label image from which areas to be "protected" are derived.
    white_surf_mask:
        Mask where all voxels inside the white matter surface is True.
    pial_surf_mask:
        Mask where all voxels inside the gray matter surface (also incl. those
        inside the white_surf_mask) is True.
    protect:
        Mapping of tissues to protect (labels refer to those in
        `protect_labeling`).
    tissue_mapping:
        Mapping of tissue names to tissue labels where the labels refer to
        those in `label_image`.

    Returns
    -------
    The corrected labeling image.
    """

    white_surf_mask = open_cross(white_surf_mask)

    wm_extra_mask = dilate_cross(np.isin(protect_labeling, protect["wm_to_gm"]), 2)

    pial_surf_mask = open_cross(pial_surf_mask)

    gm_extra_mask = dilate_cross(np.isin(protect_labeling, protect["gm_to_csf"]), 2)

    # Subcortical structures must not become WM
    wm_ignore_mask = dilate_cross(np.isin(protect_labeling, protect["gm_to_wm"]), 2)

    # Leave CSF and blood next to bone alone to keep a layer of CSF
    bone = np.isin(label_image, (tissue_mapping["Compact_bone"], tissue_mapping["Spongy_bone"]))
    dilated_bone = dilate_cube(bone)
    csf_blood_mask_shrunk = np.isin(label_image, (tissue_mapping["CSF"], tissue_mapping["Blood"])) & ~dilated_bone
    csf_ignore_mask = np.isin(protect_labeling, protect["csf_to_gm"])

    # The order of these steps matters

    # GM (inside WM surface but not subcortical) to WM
    replace = (label_image == tissue_mapping["GM"]) & white_surf_mask & ~wm_ignore_mask
    label_image[replace] = tissue_mapping["WM"]
    # CSF/blood (inside WM surface but not subcortical nor ventricles) to WM
    replace = csf_blood_mask_shrunk & white_surf_mask & ~csf_ignore_mask & ~wm_ignore_mask
    label_image[replace] = tissue_mapping["WM"]
    # WM (outside WM surface but not cerebellar WM, brainstem etc.) to GM
    replace = (label_image == tissue_mapping["WM"]) & ~(white_surf_mask | wm_extra_mask)
    label_image[replace] = tissue_mapping["GM"]

    # CSF/blood (inside GM surface but not inside WM surface nor ventricles) to GM
    replace = csf_blood_mask_shrunk & pial_surf_mask & ~white_surf_mask & ~csf_ignore_mask
    label_image[replace] = tissue_mapping["GM"]
    # GM (outside GM surface but not subcortical, cerebellar GM etc.) to CSF
    replace = (label_image == tissue_mapping["GM"]) & ~(pial_surf_mask | gm_extra_mask)
    label_image[replace] = tissue_mapping["CSF"]

    return label_image
