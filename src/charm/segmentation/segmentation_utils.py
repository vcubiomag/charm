import nibabel as nib
import numpy as np

from charm.segmentation import gems
from charm.segmentation.gmm import GMM
from charm.segmentation.probabilistic_atlas import ProbabilisticAtlas
from charm.utils import nifti


def write_image(file_name, buffer, cropping, example):
    uncropped = np.zeros(example.get_image_buffer().shape, dtype=np.float32, order="F")
    uncropped[cropping] = buffer
    gems.KvlImage(uncropped).write(file_name, example.transform_matrix)


def log_transform(image_buffers, mask):
    log_buffers = image_buffers.astype(float)
    # The C++ code skips voxels whose log intensity is zero, so nudge intensities of one inside the mask
    log_buffers[log_buffers == 1] += 1e-5
    log_buffers[~mask, :] = 1
    return np.log(log_buffers, out=log_buffers)


def read_cropped_images(image_file_names, transformed_template_file_name):
    # The returned transform places the atlas mesh nodes in the coordinate system of the image
    image_buffers = None
    for contrast, image_file_name in enumerate(image_file_names):
        image = gems.KvlImage(image_file_name, transformed_template_file_name)
        transform = image.transform_matrix
        cropping = image.crop_slices
        buffer = image.get_image_buffer()
        if image_buffers is None:
            # F order is what everything downstream of this walks the images in
            image_buffers = np.empty(buffer.shape + (len(image_file_names),), np.float32, order="F")
        image_buffers[..., contrast] = buffer

    # Read the spacing from the header to avoid decompressing the image again
    voxel_spacing = np.array(nifti.load(image_file_names[0]).header.get_zooms()[:3], dtype=float)

    return image_buffers, transform, voxel_spacing, cropping


def mask_out_background(
    image_buffers,
    image_file_names,
    atlas_file_name,
    transform,
    smoothing_sigma,
    masking_threshold,
    atlas,
    upsampled,
    mask_out_zero_intensities=True,
):
    mesh = atlas.get_mesh(atlas_file_name, transform)

    # Dilate a non-background mask by smoothing, then crop to the area covered by the mesh. Without the crop some
    # voxels would have zero prior probability for every class.
    image_size = image_buffers.shape[0:3]
    background_prior = mesh.rasterize(image_size, 0)
    # Thresholding the background prior helps for atlases built from automatic segmentations, whose background areas
    # have nonzero probability for other structures
    background_threshold = 2**8
    background_peak = 2**16 - 1
    is_background = background_prior > background_threshold
    background_prior = background_prior.astype(np.float32)
    background_prior[is_background] = background_peak

    # smoothing_sigma is the appropriate smoothing factor for a 1mm^3 scan,
    # which should be scaled according to the actual resolution of the scan
    if not upsampled:
        voxel_size = np.array(nifti.load(image_file_names[0]).header.get_zooms())
    else:
        voxel_size = np.array([0.5, 0.5, 0.5])

    smoothing_sigmas = 1.0 / voxel_size * smoothing_sigma
    smoothed_background_prior = gems.KvlImage.smooth_image_buffer(background_prior, smoothing_sigmas)

    # Priors are 16-bit ints, so 65535 is a probability of one. With isotropic Gaussian smoothing of variance
    # sigma^2, a single "on" voxel at distance d gives 1/(sqrt(2*pi)*sigma)^3 * exp(-d^2/(2*sigma^2)). A threshold t
    # then includes every voxel within
    #
    #   d = sqrt(-2*log(t*(sqrt(2*pi)*sigma)^3) * sigma^2)
    #
    # of it. For 1mm data with sigma=3 and t=0.01, d is complex: a single "on" voxel keeps no voxel, since the peak
    # value is already below 0.01.
    brain_mask_threshold = 65535.0 * (1.0 - masking_threshold)
    brain_mask = smoothed_background_prior < brain_mask_threshold

    # A constant class rasterizes to nonzero exactly where the mesh covers the image
    alphas = mesh.alphas
    mesh.alphas = np.tile([0.0, 1.0], (alphas.shape[0], 1))
    brain_mask &= mesh.rasterize(image_size, 1).astype(bool)
    mesh.alphas = alphas

    if mask_out_zero_intensities:
        brain_mask &= (image_buffers > 0).all(axis=-1)

    masked_image_buffers = image_buffers.copy()
    masked_image_buffers[~brain_mask] = 0

    return masked_image_buffers, brain_mask


def _deformed_mesh(model, transform, k=None):
    return ProbabilisticAtlas().get_mesh(
        model.model_specification.atlas_file_name,
        transform,
        k,
        model.deformation,
        model.deformation_atlas,
    )


def write_bias_corrected_images_and_segmentation(
    output_names_bias,
    output_name_segmentation,
    model,
    tissue_settings,
    csf_factor,
    template_coregistered,
):
    example_image = gems.KvlImage(model.image_file_names[0])
    exp_bias_fields = _clipped_exp_bias_fields(model.bias_fields, model.mask)

    specification = model.model_specification
    mesh = _deformed_mesh(model, gems.KvlTransform(model.transform), specification.k)
    labels = np.array(specification.free_surfer_labels, dtype=np.uint16)
    background = specification.names.index("Background")

    segmentation = _segment(
        model.image_buffers - model.bias_fields,
        model.mask,
        mesh,
        model,
        labels,
        background,
        tissue_settings["segmentation_tissues"]["CSF"],
        csf_factor,
    )

    write_image(output_name_segmentation, segmentation, model.cropping, example_image)

    unmasked_buffers, _, _, _ = read_cropped_images(model.image_file_names, template_coregistered)
    bias_corrected = unmasked_buffers / exp_bias_fields
    bias_corrected[segmentation == labels[background]] = 0
    for contrast, out_name in enumerate(output_names_bias):
        write_image(out_name, bias_corrected[..., contrast], model.cropping, example_image)


def _clipped_exp_bias_fields(bias_fields, mask):
    # Clip outside the mask to the range seen inside it, with a margin, to avoid extreme extrapolated values
    clipping_margin = np.log(2)
    in_mask = bias_fields[mask]
    return np.exp(np.clip(bias_fields, in_mask.min(axis=0) - clipping_margin, in_mask.max(axis=0) + clipping_margin))


def segment_upsampled(
    input_bias_corrected,
    tissue_settings,
    model,
    transformed_template_file_name,
    affine_atlas,
    csf_factor,
):
    atlas = ProbabilisticAtlas()
    specification = model.model_specification

    image_buffers, transform, _, cropping = read_cropped_images(input_bias_corrected, transformed_template_file_name)
    image_buffers, mask = mask_out_background(
        image_buffers,
        input_bias_corrected,
        specification.atlas_file_name,
        transform,
        specification.brain_masking_smoothing_sigma,
        specification.brain_masking_threshold,
        atlas,
        upsampled=True,
    )

    # The scans are already bias corrected
    image_buffers = log_transform(image_buffers, mask)

    labels = np.array(specification.free_surfer_labels, dtype=np.uint16)
    segmentation_tissues = tissue_settings["segmentation_tissues"]
    segmentation = _segment(
        image_buffers,
        mask,
        _deformed_mesh(model, transform),
        model,
        labels,
        specification.names.index("Background"),
        segmentation_tissues["CSF"],
        csf_factor,
    )
    del image_buffers

    # Segmentation labels are uint16, so a lookup table maps them in one pass
    label_to_tissue = np.zeros(int(np.max(labels)) + 1, dtype=segmentation.dtype)
    for tissue, label in tissue_settings["simnibs_tissues"].items():
        label_to_tissue[labels[segmentation_tissues[tissue]]] = label

    uncropped_shape = nifti.load(input_bias_corrected[0]).shape
    tissue_labeling = np.zeros(uncropped_shape, dtype=np.uint16, order="F")
    tissue_labeling[cropping] = label_to_tissue[segmentation]

    # Head mask for post-processing
    affine_mesh = atlas.get_mesh(affine_atlas, transform)
    upper_part = np.zeros(uncropped_shape, bool, order="F")
    upper_part[cropping] = (65535 - affine_mesh.rasterize(mask.shape, 1)) > 32768
    return tissue_labeling, upper_part


def save_warp_field(template_name, warp_to_mni, warp_from_mni, model):
    # warp_to_mni holds the image world coordinates of every template voxel, and warp_from_mni the template world
    # coordinates of every image voxel
    specification = model.model_specification
    transform = gems.KvlTransform(model.transform)

    positions = _deformed_mesh(model, transform, specification.k).points

    # Undo the shift from cropping the image
    positions += [slc.start for slc in model.cropping]

    image = nifti.load(model.image_file_names[0])
    template = nifti.load(template_name)

    # Rasterize the deformed positions over the undeformed mesh, which lies on the template grid
    mesh = ProbabilisticAtlas().get_mesh(specification.atlas_file_name, k=specification.k)
    template_positions = mesh.points
    coordmap_template = mesh.rasterize_values(template.shape[:3], _transform_points(image.affine, positions))
    nifti.save(nib.Nifti1Image(coordmap_template, template.affine), warp_to_mni)

    coordmap_image = _deformed_mesh(model, transform).rasterize_values(
        model.image_buffers.shape[0:-1], _transform_points(template.affine, template_positions)
    )
    uncropped_warp = np.zeros(image.shape[:3] + (3,), dtype=np.float32, order="F")
    uncropped_warp[model.cropping] = coordmap_image
    nifti.save(nib.Nifti1Image(uncropped_warp, image.affine), warp_from_mni)


def _transform_points(matrix, points):
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def _segment(
    bias_corrected,
    mask,
    mesh,
    model,
    labels,
    background,
    csf_tissues,
    csf_factor,
):
    # Voxels are walked in F order, the order of the rasterized volumes
    voxels = np.flatnonzero(mask.ravel(order="F"))
    channels = bias_corrected.shape[-1]
    data = np.ascontiguousarray(bias_corrected.reshape(-1, channels, order="F")[voxels], dtype=np.float64)

    gmm = GMM(
        model.gaussians_per_class,
        channels,
        True,
        model.means,
        model.variances,
        model.mixture_weights,
    )

    fractions = np.ascontiguousarray(model.fractions_table)
    structure_factors = np.ones(fractions.shape[1])
    if channels > 1:
        structure_factors[csf_tissues] = csf_factor

    # The first structure's voxels get the background label
    structure_labels = labels.copy()
    structure_labels[0] = labels[background]

    return gmm.segment(
        mesh,
        gems.KvlVoxelSelection(mask.shape, voxels),
        data,
        fractions,
        structure_factors.tolist(),
        structure_labels.tolist(),
    )
