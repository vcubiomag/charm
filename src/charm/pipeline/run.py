import glob
import itertools
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import numpy.typing as npt

from charm import __version__
from charm.logging import (
    console_logging_enabled,
    log_to_file,
    log_uncaught_exceptions,
    logger,
    worker_initializer,
)
from charm.mesh.element_tags import ElementTags
from charm.mesh.io import write_gifti_surface, write_msh
from charm.mesh.meshing import create_mesh
from charm.mesh.msh import ElementData
from charm.report import html_writer
from charm.resources import config_file, templates
from charm.segmentation import charm_utils, gems, segmentation_utils
from charm.settings import read_ini
from charm.subject import HEMISPHERES, SubjectFiles
from charm.utils import _thickness, nifti, transformations
from charm.utils.parallel import BackgroundTasks
from charm.utils.timing import format_time


@dataclass
class _Atlas:
    template_name: str
    settings: dict
    path: Path
    level1: str
    level2: str
    affine_name: str
    gmm_parameters: str


def run(
    subject_dir: str,
    T1=None,
    T2=None,
    registerT2=False,
    initatlas=False,
    segment=False,
    create_surfaces=False,
    mesh_image=False,
    usesettings=None,
    noneck=False,
    init_transform=None,
    use_transform=None,
    force_qform=False,
    force_sform=False,
    options_str=None,
    debug=False,
):
    """Run the charm pipeline.

    Parameters
    ----------
    subject_dir : str
        Output directory.
    T1 : str
        T1-weighted image.
    T2 : str
        T2-weighted image.
    registerT2 : bool
        Register the T2 to the T1.
    initatlas : bool
        Register the atlas affinely to the input images.
    segment : bool
        Run the volume segmentation.
    create_surfaces : bool
        Create the cortical surfaces and update the label image.
    mesh_image : bool
        Run the tetrahedral meshing.
    usesettings : str
        Settings file to use instead of the packaged charm.ini.
    noneck : bool
        The scans don't include the neck.
    init_transform : path-like
        Space-delimited text file with a 4x4 matrix that takes the MNI template
        to subject space. Initializes the affine registration.
    use_transform : path-like
        Like init_transform, but used instead of the affine registration.
    force_qform, force_sform : bool
        Overwrite the sform with the qform, or the reverse.
    options_str : str
        Command line options to record in the log.
    debug : bool
        Write the results of intermediate steps.
    """
    if force_qform and force_sform:
        raise ValueError("Can't force both q- and s-forms, please use only one of the flags.")

    os.makedirs(subject_dir, exist_ok=True)
    sub_files = SubjectFiles(subpath=str(subject_dir))

    with log_to_file(sub_files.charm_log), log_uncaught_exceptions():
        _run(
            sub_files,
            T1,
            T2,
            registerT2,
            initatlas,
            segment,
            create_surfaces,
            mesh_image,
            usesettings,
            noneck,
            init_transform,
            use_transform,
            force_qform,
            force_sform,
            options_str,
            debug,
        )


def _run(
    sub_files,
    T1,
    T2,
    registerT2,
    initatlas,
    segment,
    create_surfaces,
    mesh_image,
    usesettings,
    noneck,
    init_transform,
    use_transform,
    force_qform,
    force_sform,
    options_str,
    debug,
):
    start = time.time()

    logger.info(f"charm version {__version__}")
    logger.info(f"charm run started: {time.asctime()}")
    logger.debug(f"options: {options_str}")

    settings = _read_settings_and_copy(usesettings, sub_files.settings)
    denoise_settings = settings["preprocess"]
    do_denoise = denoise_settings["denoise"]
    samseg_settings = settings["samseg"]
    logger.debug(settings)

    if init_transform:
        init_transform = _read_transform(init_transform)

    if use_transform:
        use_transform = _read_transform(use_transform)

    _prepare_t1(T1, sub_files.reference_volume, force_qform, force_sform)
    _prepare_t2(
        sub_files.reference_volume,
        T2,
        registerT2,
        sub_files.T2_reg,
        force_qform,
        force_sform,
    )

    os.makedirs(sub_files.segmentation_folder, exist_ok=True)

    if do_denoise:
        _denoise_inputs(sub_files.reference_volume, sub_files.T2_reg, sub_files)

    num_threads = settings["general"]["threads"]
    if isinstance(num_threads, int) and num_threads > 0:
        gems.set_number_of_threads(num_threads)
        _thickness.set_number_of_threads(num_threads)
        logger.info(f"Using {num_threads} threads, instead of all available.")

    atlas = _setup_atlas(samseg_settings, sub_files.T2_reg, usesettings)

    if initatlas:
        _register_atlas(
            sub_files,
            settings,
            atlas,
            noneck,
            init_transform,
            use_transform,
        )
    if segment:
        _segment(sub_files, settings, atlas, debug)
    registrations = None
    try:
        if create_surfaces:
            registrations = _create_surfaces(sub_files, settings, atlas)
        if mesh_image:
            _create_mesh(sub_files, settings, debug)
    except BaseException:
        if registrations is not None:
            registrations.cancel()
        raise
    if registrations is not None:
        registrations.wait()
        logger.info("Spherical registrations done")

    logger.info("Creating report")
    html_writer.write_report(sub_files)

    logger.info("charm run finished: " + time.asctime())
    logger.info("Total running time: " + format_time(time.time() - start))


def _register_atlas(
    sub_files,
    settings,
    atlas,
    noneck,
    init_transform,
    use_transform,
):
    samseg_settings = settings["samseg"]
    do_denoise = settings["preprocess"]["denoise"]

    logger.info("Starting affine registration and neck correction.")
    inputT1 = sub_files.T1_denoised if do_denoise else sub_files.reference_volume

    if use_transform is not None:
        logger.info("Using world-to-world transform provided by user.")
        trans_mat = use_transform
    else:
        logger.info("Estimating affine MNI to RAS transformation")
        init_type = samseg_settings["init_type"].lower()
        match init_type:
            case "atlas":
                logger.info("Using method: atlas")
                trans_mat = None
            case "mni":
                mni_template = templates.mni_volume
                mni_settings = settings["initmni"]
                logger.info("Using method: initmni")
                trans_mat = charm_utils._init_atlas_affine(inputT1, mni_template, mni_settings)
            case "trega":
                # Imported here because it loads torch.
                from charm.surface import estimation

                logger.info("Using method: TREGA")
                # Too small a model to gain anything from a GPU
                trans_mat = estimation.estimate_mni152_to_ras_affine([sub_files], device="cpu")[0]
                trans_mat = _RAS2LPS_transform(trans_mat)
            case _:
                logger.info(f"Invalid initialization '{init_type}'. Defaulting to 'atlas'")
                trans_mat = None

    charm_utils._register_atlas_to_input_affine(
        inputT1,
        atlas.template_name,
        atlas.affine_name,
        atlas.level1,
        atlas.level2,
        sub_files.segmentation_folder,
        sub_files.template_coregistered,
        settings["initatlas"],
        atlas.settings["neck_optimization"]["neck_tissues"],
        noneck,
        world_to_world_transform_matrix=trans_mat,
        init_transform=init_transform,
    )


def _segment(sub_files, settings, atlas, debug):
    do_denoise = settings["preprocess"]["denoise"]

    input_images = []
    input_images.append(sub_files.T1_denoised if do_denoise else sub_files.reference_volume)
    if os.path.exists(sub_files.T2_reg):
        input_images.append(sub_files.T2_reg_denoised if do_denoise else sub_files.T2_reg)

    segment_settings = settings["segment"]
    logger.info("Estimating parameters.")
    segment_parameters_and_inputs = charm_utils._estimate_parameters(
        sub_files.segmentation_folder,
        sub_files.template_coregistered,
        atlas.path,
        input_images,
        segment_settings,
        atlas.gmm_parameters,
        parameter_filename=os.path.join(sub_files.segmentation_folder, "parameters.p") if debug else None,
    )

    # The upsampled segmentation needs the bias corrected scans and the
    # segmentation on disk first.

    bias_corrected_image_names = [sub_files.T1_bias_corrected]
    if len(input_images) > 1:
        bias_corrected_image_names.append(sub_files.T2_bias_corrected)

    logger.info("Writing out normalized images and labelings.")
    os.makedirs(sub_files.surface_folder, exist_ok=True)

    tissue_settings = atlas.settings["conductivity_mapping"]
    csf_factor = segment_settings["csf_factor"]
    segmentation_utils.write_bias_corrected_images_and_segmentation(
        bias_corrected_image_names,
        sub_files.labeling,
        segment_parameters_and_inputs,
        tissue_settings,
        csf_factor,
        sub_files.template_coregistered,
    )

    fn_LUT = sub_files.labeling.rsplit(".", 2)[0] + "_LUT.txt"
    shutil.copyfile(templates.labeling_LUT, fn_LUT)

    logger.info("Writing out MNI warps.")
    os.makedirs(sub_files.mni_transf_folder, exist_ok=True)
    segmentation_utils.save_warp_field(
        atlas.template_name,
        sub_files.mni2conf_nonl,
        sub_files.conf2mni_nonl,
        segment_parameters_and_inputs,
    )

    logger.info("Post-processing segmentation")
    os.makedirs(sub_files.label_prep_folder, exist_ok=True)

    cleaned_upsampled_tissues, upsampled_images = charm_utils._post_process_segmentation(
        bias_corrected_image_names,
        tissue_settings,
        csf_factor,
        segment_parameters_and_inputs,
        sub_files.template_coregistered,
        atlas.affine_name,
        sub_files.tissue_labeling_before_morpho,
        sub_files.upper_mask,
        debug=debug,
    )

    reference = nifti.load_nifti(sub_files.reference_volume)
    scode = reference.get_sform(coded=True)[1]
    qcode = reference.get_qform(coded=True)[1]
    affine_upsampled = upsampled_images[0].affine
    upsampled_names = (sub_files.T1_upsampled, sub_files.T2_upsampled)

    upsampled_tissues = nib.Nifti1Image(cleaned_upsampled_tissues, affine_upsampled)
    for image, name in zip(
        (upsampled_tissues, *upsampled_images), (sub_files.tissue_labeling_upsampled, *upsampled_names)
    ):
        image.set_qform(affine_upsampled, qcode)
        image.set_sform(affine_upsampled, scode)
        if image is not upsampled_tissues:
            image.set_data_dtype(np.int16)
        nifti.save(image, name)

    del cleaned_upsampled_tissues

    fn_LUT = sub_files.tissue_labeling_upsampled.rsplit(".", 2)[0] + "_LUT.txt"
    shutil.copyfile(templates.final_tissues_LUT, fn_LUT)


def _create_surfaces(sub_files, settings, atlas):
    logger.info("Starting surface creation")
    timer_init = time.perf_counter()

    surface_settings = settings["surfaces"]
    tissue_map_simnibs = atlas.settings["conductivity_mapping"]["simnibs_tissues"]

    os.makedirs(sub_files.surface_folder, exist_ok=True)

    logger.info("Estimating cortical surfaces")

    from charm.surface import estimation

    cortex, curv = estimation.cortical_surface_estimation(
        [sub_files],
        surface_settings["topofit_contrast"],
        surface_settings["topofit_resolution"],
        surface_settings["topofit_device"],
    )
    # Only one subject.
    cortex = cortex[0]
    curv = curv[0]

    logger.info("Estimating the central gray matter surface")
    central = {
        hemi.name: hemi.estimate_layers(
            surface_settings["central_surface_method"],
            surface_settings["central_surface_fraction"],
            curv=_smoothed_mean_curvature(hemi),
            return_surface=True,
        )
        for hemi in cortex
    }

    _write_cortex_as_gifti(cortex, sub_files)
    _write_gifti(central["lh"], sub_files, "central", "lh")
    _write_gifti(central["rh"], sub_files, "central", "rh")
    _write_vertex_data_as_curv(curv, sub_files)

    # Each hemisphere is independent, so they run side by side. Nothing but the report comes after them, so they also
    # run alongside the rest of the pipeline.
    logger.info("Generating spherical registrations")
    from charm.surface.registration import spherical_registration

    registrations = BackgroundTasks(
        surface_settings["spherical_registration_process_pool"],
        spherical_registration,
        itertools.product([sub_files], HEMISPHERES),
        initializer=worker_initializer if console_logging_enabled() else None,
    )

    if surface_settings["update_segmentation_from_surfaces"]:
        logger.info("Updating the segmentation using the cortical surfaces")

        charm_utils.update_labeling_from_cortical_surfaces(
            sub_files,
            _fs_lut_labels_to_fs_lut_values(surface_settings["update_segmentation_protect"]),
            tissue_map_simnibs,
        )

    time_elapsed = time.strftime("%H:%M:%S", time.gmtime(time.perf_counter() - timer_init))
    logger.info(f"Surface creation : {time_elapsed}")
    return registrations


def _create_mesh(sub_files, settings, debug):
    logger.info("Starting mesh")
    label_image = nifti.load(sub_files.tissue_labeling_upsampled)
    label_buffer = np.round(label_image.get_fdata()).astype(np.uint16)  # meshing requires uint16
    label_affine = label_image.affine
    # Cropping reduces memory use.
    label_buffer, label_affine, _ = transformations.crop_vol(
        label_buffer, label_affine, label_buffer > 0, thickness_boundary=5
    )

    mesh_settings = settings["mesh"]
    elem_sizes = mesh_settings["elem_sizes"]
    smooth_size_field = mesh_settings["smooth_size_field"]
    skin_facet_size = mesh_settings["skin_facet_size"]
    if not skin_facet_size:
        skin_facet_size = None
    facet_distances = mesh_settings["facet_distances"]
    optimize = mesh_settings["optimize"]
    apply_cream = mesh_settings["apply_cream"]
    remove_spikes = mesh_settings["remove_spikes"]
    skin_tag = mesh_settings["skin_tag"]
    if not skin_tag:
        skin_tag = None
    hierarchy = mesh_settings["hierarchy"]
    if not hierarchy:
        hierarchy = None
    smooth_steps = mesh_settings["smooth_steps"]
    skin_care = mesh_settings["skin_care"]
    mmg_noinsert = mesh_settings["mmg_noinsert"]

    logger.info(f"Using skin tag: {skin_tag}")

    debug_path = None
    if debug:
        debug_path = sub_files.subpath

    final_mesh = create_mesh(
        label_buffer,
        label_affine,
        elem_sizes=elem_sizes,
        smooth_size_field=smooth_size_field,
        skin_facet_size=skin_facet_size,
        facet_distances=facet_distances,
        optimize=optimize,
        remove_spikes=remove_spikes,
        skin_tag=skin_tag,
        hierarchy=hierarchy,
        apply_cream=apply_cream,
        smooth_steps=smooth_steps,
        skin_care=skin_care,
        mmg_noinsert=mmg_noinsert,
        debug_path=debug_path,
        debug=debug,
    )
    logger.info("Relabeling internal air boundaries")
    final_mesh = final_mesh.relabel_internal_air()

    logger.info("Writing mesh")
    write_msh(final_mesh, sub_files.fnamehead)

    logger.info("Transforming EEG positions")
    idx = final_mesh.elm.get_triangles(skin_tag)
    mesh = final_mesh.crop_mesh(elements=final_mesh.elm.elm_number[idx])

    if not os.path.exists(sub_files.eeg_cap_folder):
        os.mkdir(sub_files.eeg_cap_folder)

    cap_files = glob.glob(os.path.join(templates.eeg_montage_dir, "*.csv"))
    warp_image = nifti.load(sub_files.mni2conf_nonl)
    warp = (np.asanyarray(warp_image.dataobj), warp_image.affine)
    for fn in cap_files:
        fn_out = os.path.splitext(os.path.basename(fn))[0]
        fn_out = os.path.join(sub_files.eeg_cap_folder, fn_out)
        transformations.warp_coordinates(
            fn,
            warp,
            out_name=fn_out + ".csv",
            out_geo=fn_out + ".geo",
            mesh=mesh,
            skin_tag=skin_tag,
        )

    logger.info("Write label image from mesh")
    MNI_template = templates.mni_volume
    mesh = final_mesh.crop_mesh(elm_type=4)
    field = mesh.elm.tag1.astype(np.uint16)
    ed = ElementData(field)
    ed.mesh = mesh
    ed.to_deformed_grid(
        sub_files.mni2conf_nonl,
        MNI_template,
        out=sub_files.final_labels_MNI,
        out_original=sub_files.final_labels,
        order=0,
        reference_original=sub_files.reference_volume,
    )

    fn_LUT = sub_files.final_labels.rsplit(".", 2)[0] + "_LUT.txt"
    shutil.copyfile(templates.final_tissues_LUT, fn_LUT)


def _read_settings_and_copy(usesettings, fn_settingslog):
    if usesettings is None:
        fn_settings = config_file("charm.ini")
    else:
        if type(usesettings) == list:
            usesettings = usesettings[0]
        fn_settings = usesettings
    settings = read_ini(fn_settings)
    try:
        shutil.copyfile(fn_settings, fn_settingslog)
    except shutil.SameFileError:
        pass
    return settings


def _prepare_t1(T1, reference_volume, force_qform, force_sform):
    if T1:
        if os.path.exists(T1):
            T1_tmp = nifti.load_nifti(T1)
            T1_tmp = _check_q_and_s_form(T1_tmp, force_qform, force_sform)

            # Squeeze after the form check so the affine is correct.
            if (np.array(T1_tmp.shape) == 1).any():
                data_tmp = np.squeeze(T1_tmp.get_fdata())
                T1_tmp = nib.Nifti1Image(data_tmp, T1_tmp.affine)

            T1_tmp.set_data_dtype(np.float32)

            nifti.save(T1_tmp, reference_volume)
        else:
            raise FileNotFoundError(f"Could not find input T1 file: {T1}")


def _prepare_t2(T1, T2, registerT2, T2_reg, force_qform, force_sform):
    if T2:
        T2_exists = os.path.exists(T2)
        if registerT2:
            if T2_exists:
                # T2_reg doubles as a temporary file holding the T2 with
                # fixed q- and s-forms.
                T2_tmp = nifti.load_nifti(T2)
                T2_tmp = _check_q_and_s_form(T2_tmp, force_qform, force_sform)

                if (np.array(T2_tmp.shape) == 1).any():
                    data_tmp = np.squeeze(T2_tmp.get_fdata())
                    T2_tmp = nib.Nifti1Image(data_tmp, T2_tmp.affine)

                nifti.save(T2_tmp, T2_reg)
                charm_utils._registerT1T2(T1, T2_reg, T2_reg)
            else:
                raise FileNotFoundError(f"Could not find input T2 file: {T2}")
        else:
            if T2_exists:
                T2_tmp = nifti.load_nifti(T2)
                T2_tmp = _check_q_and_s_form(T2_tmp, force_qform, force_sform)

                if (np.array(T2_tmp.shape) == 1).any():
                    data_tmp = np.squeeze(T2_tmp.get_fdata())
                    T2_tmp = nib.Nifti1Image(data_tmp, T2_tmp.affine)

                T2_tmp.set_data_dtype(np.float32)
                nifti.save(T2_tmp, T2_reg)


def _check_q_and_s_form(scan, force_qform=False, force_sform=False):
    if not scan.get_qform(coded=True)[1] > 0 and force_sform is False:
        raise ValueError(
            "The qform_code is 0. Please check the header of the input scan. You can use the sform instead by running charm with the --forcesform option."
        )

    if not np.allclose(scan.get_qform(), scan.get_sform(), rtol=1e-5, atol=1e-6):
        if not (force_qform or force_sform):
            raise ValueError(
                "The qform and sform matrices do not match. Please run charm with the --forceqform (preferred) or --forcesform option"
            )
        elif force_qform:
            (qmat, qcode) = scan.get_qform(coded=True)
            scan.set_sform(qmat, code=qcode)

        elif force_sform:
            if not scan.get_sform(coded=True)[1] > 0:
                raise ValueError(
                    "The sform_code is 0, but you are forcing it. Please fix the sform_code or use the qform instead."
                )
            # set_qform silently strips shears, so the sform is set from the
            # resulting qform to keep both consistent.
            else:
                (mat_tmp, code_tmp) = scan.get_sform(coded=True)
                scan.set_qform(mat_tmp, code=code_tmp)
                (mat_tmp, code_tmp) = scan.get_qform(coded=True)
                scan.set_sform(mat_tmp, code=code_tmp)

    return scan


def _setup_atlas(samseg_settings, T2_reg, usesettings):
    atlas_name = samseg_settings["atlas_name"]
    logger.info("Using " + atlas_name + " as charm atlas.")
    atlas_path = templates.atlas_path(atlas_name)
    atlas_settings = read_ini(os.path.join(atlas_path, atlas_name + ".ini"))
    atlas_settings_names = atlas_settings["names"]
    template_name = os.path.join(atlas_path, atlas_settings_names["template_name"])
    atlas_affine_name = os.path.join(atlas_path, atlas_settings_names["affine_atlas"])
    atlas_level1 = os.path.join(atlas_path, atlas_settings_names["atlas_level1"])
    atlas_level2 = os.path.join(atlas_path, atlas_settings_names["atlas_level2"])
    custom_gmm_parameters = samseg_settings["gmm_parameter_file"]

    if type(usesettings) == list:
        usesettings = usesettings[0]

    if not usesettings or not custom_gmm_parameters:
        if os.path.exists(T2_reg):
            gmm_parameters = os.path.join(atlas_path, atlas_settings_names["gaussian_parameters_t2"])
        else:
            gmm_parameters = os.path.join(atlas_path, atlas_settings_names["gaussian_parameters_t1"])
    else:
        settings_dir = os.path.dirname(usesettings)
        gmm_parameters = os.path.join(settings_dir, custom_gmm_parameters)
        if not os.path.exists(gmm_parameters):
            raise FileNotFoundError(f"Could not find gmm parameter file: {gmm_parameters}")

    return _Atlas(
        template_name,
        atlas_settings,
        atlas_path,
        atlas_level1,
        atlas_level2,
        atlas_affine_name,
        gmm_parameters,
    )


def _denoise_inputs(T1, T2, sub_files):
    if T1:
        logger.info("Denoising the T1 input and saving.")
        charm_utils._denoise_input_and_save(T1, sub_files.T1_denoised)
    if T2 and os.path.exists(sub_files.T2_reg):
        logger.info("Denoising the registered T2 and saving.")
        charm_utils._denoise_input_and_save(sub_files.T2_reg, sub_files.T2_reg_denoised)


def _RAS2LPS_transform(t: npt.NDArray):
    # ITK uses LPS internally.
    RAS2LPS = np.diag([-1, -1, 1, 1])
    return RAS2LPS @ t @ RAS2LPS


def _read_transform(transform_file):
    transform = np.loadtxt(transform_file)
    error_msg = f"`transform` should have shape (4, 4), got {transform.shape}"
    assert transform.shape == (4, 4), error_msg
    return _RAS2LPS_transform(transform)


def _fs_lut_labels_to_fs_lut_values(labels_dict: dict[str, list[str]]):
    values, labels, _ = charm_utils.read_freesurfer_lut(templates.labeling_LUT)
    mapper = dict(zip(labels, values))
    return {k: [mapper[i] for i in v] for k, v in labels_dict.items()}


def _smoothed_mean_curvature(hemi, n_iter: int = 10):
    # Surface.curvature(smooth_iter=...) cannot be used: cortech 0.1 passes the
    # curvature as the step size of smooth_laplacian and raises a ValueError.
    return 0.5 * sum(
        surface.smooth_laplacian(array=surface.curvature().H, n_iter=n_iter) for surface in (hemi.white, hemi.pial)
    )


def _write_gifti(surface, sub_files, name, hemi):
    tag = ElementTags.from_surface_file_name(name, hemi)
    filename = sub_files.get_surface_from_element_tag(tag)
    write_gifti_surface(surface, filename, element_tag=tag)


def _write_cortex_as_gifti(cortex, sub_files):
    for hemi in cortex:
        kw = {"sub_files": sub_files, "hemi": hemi.name}
        _write_gifti(hemi.white, name="white", **kw)
        _write_gifti(hemi.pial, name="pial", **kw)


def _write_vertex_data_as_curv(curv, sub_files):
    for h, v in curv.items():
        for s, vv in v.items():
            for n, vd in vv.items():
                nib.freesurfer.write_morph_data(Path(sub_files.surface_folder) / f"{h}.{s}.{n}", vd)
