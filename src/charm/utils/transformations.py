"""
Linear and non-linear transformations.

    Derived from the SimNIBS package.
    Please check on www.simnibs.org how to cite our work in publications.
    Copyright (C) 2020 Kristoffer H Madsen, Guilherme B Saturnino, Axel Thielscher

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import os
import warnings
from functools import partial

import nibabel as nib
import numpy as np
import scipy.ndimage

from charm.mesh.element_tags import ElementTags
from charm.utils import nifti
from charm.utils.csv_reader import read_csv_positions, write_csv_positions

__all__ = [
    "warp_coordinates",
]


def volumetric_nonlinear(
    image,
    deformation,
    target_space_affine=None,
    target_dimensions=None,
    intorder=1,
    cpus=1,
    inverse_deformation=None,
    keep_vector_length=True,
    fix_boundary_zeros=True,
):
    """Apply a volumetric non-linear transformation.

    Parameters
    --------
    image: tuple
        tuple (ndarray, affine) with the data to be transformed
    deformation: tuple
        tuple (ndarray, affine) with deformation field. The deformation field is in the
        target space and specifies in each voxel the equivalent coordinates (x, y, z)
        of the original space
    target_space_affine: ndarray (Optional)
        4x4 matrix with the voxel indices to (x, y, z) coordinates in the target space.
        Default: same as the deformation field
    target_dimensions: array (Optional)
        Dimensions of volume in the final image. Default: same as the deformation field
    intorder: int (Optional)
        Order of interpolation. See the documentation for the
        scipy.ndimage.map_coordinates function. Default=1
    inverse_deformation: tuple
        tuple (ndarray, affine) with inverse field. The inverse deformation field is in the
        original space and specifies in each voxel the equivalent coordinates (x, y, z)
        of the target space. This is used only to rotate vectors. If not set,
        vectors will not be rotated
    keep_vector_length: bool (Optional)
        Whether to keep the length of the vectors unchanged. Only comes into effect if the
        image is 3 dimensional and inverse_deformation is set. (Default: True)
    fix_boundary_zeros: bool (Optional)
        Whether to set the zero vectors at the boundaries of the deformation
        field to Inf, which places them outside the image (Default: True)

    Returns
    ------
    outdata: ndarray
        ndarray corresponding to the transformed image
    """
    im_data, im_affine = image
    df_data, df_affine = deformation
    if len(df_data.shape) > 4:
        df_data = df_data.squeeze()

    # Boundaries of deformation fields sometimes contain (0, 0, 0) vectors.
    if fix_boundary_zeros:
        df_data = _fix_boundary_zeros(df_data)

    if target_dimensions is None:
        target_dimensions = df_data.shape[:3]

    if target_space_affine is not None and not np.allclose(target_space_affine, df_affine):
        xyzvox = np.array(
            np.meshgrid(
                np.arange(target_dimensions[0], dtype=float),
                np.arange(target_dimensions[1], dtype=float),
                np.arange(target_dimensions[2], dtype=float),
                indexing="ij",
            )
        )

        iM = np.linalg.inv(df_affine).dot(target_space_affine)
        t = iM[:3, :3].dot(xyzvox.reshape(3, -1)) + iM[:3, 3, None]
        f = partial(
            scipy.ndimage.map_coordinates,
            coordinates=t,
            output=np.float32,
            order=1,
            mode="constant",
            cval=np.inf,
        )
        voxvals = np.array([f(df_data[..., i]) for i in range(3)])
        iM = np.linalg.inv(im_affine)
        coords = iM[:3, :3].dot(voxvals) + iM[:3, 3, None]

    else:
        iM = np.linalg.inv(im_affine)
        coords = iM[:3, :3].dot(df_data.reshape(-1, 3).T) + iM[:3, 3, None]
        target_space_affine = df_affine

    # Rotate vector images while still in the original space.
    if len(im_data.shape) == 4 and im_data.shape[3] == 3 and inverse_deformation is not None:
        inv_df_data, inv_df_affine = inverse_deformation
        if len(inv_df_data.shape) > 4:
            inv_df_data = inv_df_data.squeeze()
        im_data_rotated = np.zeros_like(im_data)
        spacing = np.linalg.norm(inv_df_affine[:3, :3], axis=1)
        t = np.linalg.inv(inv_df_affine).dot(im_affine)
        for i in range(3):
            grad = np.gradient(inv_df_data[..., i], *spacing)
            grad = np.nan_to_num(np.asarray(grad), copy=False)
            for j in range(3):
                grad_t = scipy.ndimage.affine_transform(
                    grad[j],
                    t[:3, :3],
                    t[:3, 3],
                    output_shape=im_data.shape[:3],
                    order=1,
                    cval=0.0,
                    mode="constant",
                )
                im_data_rotated[..., i] += grad_t * im_data[..., j]

        # The gradients were calculated in voxel space, so they still need to be
        # rotated to world space.
        im_data_rotated = np.einsum(
            "ij,klmj->klmi",
            inv_df_affine[:3, :3] / spacing[:, None],
            im_data_rotated,
        )

        if keep_vector_length:
            im_data_rotated = (
                im_data_rotated
                * (np.linalg.norm(im_data, axis=3))[:, :, :, None]
                / (1e-32 + np.linalg.norm(im_data_rotated, axis=3))[:, :, :, None]
            )
        im_data = im_data_rotated

    coords[np.isnan(coords)] = np.inf  # inf is always outside the volume, map_coordinates on nan is unclear
    outdata = _interpolate(im_data, coords, intorder, target_dimensions, cpus)
    return outdata


def _fix_boundary_zeros(df_data):
    """Set the zero vectors outside the filled nonzero region of a deformation field to Inf."""
    assert df_data.ndim == 4
    assert df_data.shape[3] == 3

    mask_imgregion = ~np.all(df_data == 0, axis=3)
    mask_imgregion = scipy.ndimage.binary_fill_holes(mask_imgregion)
    if not np.all(mask_imgregion):
        df_data[~mask_imgregion, :] = np.inf

    return df_data


def volumetric_affine(
    image,
    affine,
    target_space_affine,
    target_dimensions,
    intorder=1,
    cpus=1,
    keep_vector_length=True,
):
    """Apply an affine transformation.

    Parameters
    --------
    image: tuple
        tuple (ndarray, affine) with the data to be transformed
    affine: ndarray
        4x4 matrix specifying the transformation from (x, y, z) in the target space to
        (x, y, z) in the original space
    target_space_affine: ndarray
        4x4 matrix with the voxel indices to (x, y, z) coordinates in the target space
    target_dimensions: array
        Dimensions of volume in the final image
    intorder: int (Optional)
        Order of interpolation. See the documentation for the
        scipy.ndimage.map_coordinates function. Default: 1
    keep_vector_length: bool
        Whether to keep the length of the vectors unchanged. Only comes into effect if the
        image is 3 dimensional. Default: True

    Returns
    ------
    outdata: ndarray
        ndarray corresponding to the transformed image
    """
    im_data, im_affine = image

    iM = np.linalg.inv(im_affine).dot(affine.dot(target_space_affine))
    if im_data.ndim == 4:
        outdata = np.stack(
            [
                scipy.ndimage.affine_transform(
                    im_data[..., i],
                    iM[:3, :3],
                    iM[:3, 3],
                    target_dimensions,
                    order=intorder,
                )
                for i in range(im_data.shape[3])
            ],
            axis=-1,
        )
    else:
        outdata = scipy.ndimage.affine_transform(im_data, iM[:3, :3], iM[:3, 3], target_dimensions, order=intorder)

    if len(outdata.shape) == 4 and outdata.shape[3] == 3:
        outdata_rotated = np.einsum("ij,klmj->klmi", np.linalg.inv(affine[:3, :3]), outdata)
        if keep_vector_length:
            outdata_rotated = (
                outdata_rotated
                * (np.linalg.norm(outdata, axis=3))[:, :, :, None]
                / (1e-32 + np.linalg.norm(outdata_rotated, axis=3))[:, :, :, None]
            )

        outdata = outdata_rotated

    return outdata


def _interpolate(im_data, coords, intorder, outdim, cpus=1):
    if len(im_data.shape) > 3:
        indim = im_data.shape[3]
        squeeze = False
    else:
        indim = 1
        im_data = im_data[..., None]
        squeeze = True
    f = partial(
        scipy.ndimage.map_coordinates,
        coordinates=coords,
        output=im_data.dtype,
        order=intorder,
        mode="constant",
        cval=0.0,
    )
    outdata = np.array([f(im_data[..., i]) for i in range(indim)]).T
    outdata = outdata.reshape(tuple(outdim) + (indim,))
    if squeeze:
        outdata = outdata.squeeze()
    return outdata


def nifti_transform(
    image,
    warp,
    ref,
    out=None,
    mask=None,
    order=1,
    inverse_warp=None,
    binary=False,
    fix_boundary_zeros=True,
):
    """Transform a nifti file to the reference space using the given warp.

    Parameters
    --------
    image: str or tuple
        Name of original file or tuple with (image, affine) pair
    warp: str
        Name of file with the transformation. Can either be a nifti file
        where each voxel corresponds to coordinates (x, y, z) in the original space
        or an affine transformation defined from the target space to the original
        space. In the latter case, the name must finish in ".mat", and it will be read
        with the numpy loadtxt file
    ref: str
        Name of reference file. The output will be in the same space
        as the reference (same affine transformation and same dimensions)
    out: str (optional)
        If not None, the result will be written to this file as a nifti
    mask: str (optional)
        Mask to be applied before transformation
    order: int (optional)
        Interpolation order to be used. Default: 1
    inverse_warp: str
        Name of nifti file with the inverse transformation. Used to rotate vectors to the
        target space in the case of non-linear transformations. If the transformation is
        linear, the inverse matrix is used.
    binary: bool (optional)
        If True, thresholds the image at 0.5 and converts to np.uint8. (Default: False)
    fix_boundary_zeros: bool (Optional)
        See volumetric_nonlinear. Relevant only for non-linear transforms
        (Default: True)

    Returns
    ------
    img: nibabel.Nifti1Pair
        Nibabel image object with transformed field
    """
    if isinstance(image, str):
        image = nifti.load(image)
        im_data = np.asanyarray(image.dataobj)
        im_affine = image.affine.copy()
        im_data = np.nan_to_num(np.asarray(im_data), copy=False)
    else:
        im_data, im_affine = image
        im_data = np.nan_to_num(np.asarray(im_data), copy=False)
    reference_nifti = nifti.load_nifti(ref)
    if mask is not None:
        mask = nifti.load(mask)
        mask_data = np.asanyarray(mask.dataobj)
        mask_data[np.isnan(mask_data)] = 0
        if not np.allclose(mask.affine, im_affine):
            raise ValueError(
                "Could not apply mask: the affine transformation must be the same as the one in the input image"
            )
        if not mask_data.shape[:2] == im_data.shape[:2]:
            raise ValueError("Could not apply mask: The shape of the mask must be the same as the image shape")
        im_data *= mask_data.astype(im_data.dtype)

    if isinstance(warp, np.ndarray):
        affdeform = warp
        image = volumetric_affine(
            (im_data, im_affine),
            affdeform,
            target_space_affine=reference_nifti.affine,
            target_dimensions=reference_nifti.header["dim"][1:4],
            intorder=order,
        )

    elif os.path.splitext(warp)[1] in [".mat", ".txt"]:
        affdeform = np.loadtxt(warp)
        image = volumetric_affine(
            (im_data, im_affine),
            affdeform,
            target_space_affine=reference_nifti.affine,
            target_dimensions=reference_nifti.header["dim"][1:4],
            intorder=order,
        )

    else:
        warp_nifti = nifti.load(warp)
        if inverse_warp is not None:
            inverse_warp = nifti.load(inverse_warp)
            inverse_warp = (np.asanyarray(inverse_warp.dataobj), inverse_warp.affine)
        image = volumetric_nonlinear(
            (im_data, im_affine),
            (np.asanyarray(warp_nifti.dataobj), warp_nifti.affine),
            target_space_affine=reference_nifti.affine,
            target_dimensions=reference_nifti.header["dim"][1:4],
            intorder=order,
            inverse_deformation=inverse_warp,
            fix_boundary_zeros=fix_boundary_zeros,
        )

    if binary:
        image = (image >= 0.5).astype(np.uint8)
    img = nib.Nifti1Image(image, reference_nifti.affine)
    img.header.set_xyzt_units(reference_nifti.header.get_xyzt_units()[0])
    img.header.set_qform(reference_nifti.header.get_qform(), code=2)
    img.update_header()

    if image.dtype == np.float64:
        img.set_data_dtype(np.float32)

    if out is not None:
        nifti.save(img, out)

    return img


def coordinates_nonlinear(coordinates, deformation, intorder=1, vectors=None, keep_length=True):
    """Apply a non-linear transformation to coordinates.

    Parameters
    --------
    coordinates: nx3 ndarray
        ndarray with coordinates to be transformed
    deformation: tuple
        tuple (ndarray, affine) with deformation field. The deformation field specifies
        in each voxel the coordinates (x, y, z) of the target space. This is the opposite
        transformation from the one used in the volumetric transform
    intorder: int
        Order of interpolation. See the documentation for the
        scipy.ndimage.map_coordinates function
    vectors: nx3 ndarray (optional)
        ndarray with vectors to be transformed. Needs to have the same length as
        "coordinates".
    keep_length: bool (optional)
        Whether to preserve the lengths of the vectors. Default: True

    Returns
    ------
    coordinates_transformed: nx3 ndarray
        ndarray corresponding to the transformed coordinates. If outside the volume, will
        return the transformation of the closest coordinate inside the volume
    """
    if len(coordinates.shape) == 1:
        coordinates = coordinates[None, :]
    if coordinates.shape[1] != 3:
        raise ValueError("Input coordinates should have a nx3 format")
    if vectors is not None and len(vectors.shape) == 1:
        vectors = vectors[None, :]

    df_data, df_affine = deformation
    df_data = df_data.squeeze()

    iM = np.linalg.inv(df_affine)
    t = iM[:3, :3].dot(coordinates.T) + iM[:3, 3, None]
    f = partial(
        scipy.ndimage.map_coordinates,
        coordinates=t,
        output=np.float32,
        order=intorder,
        mode="nearest",
    )
    coords = np.array([f(df_data[..., i]) for i in range(3)])

    if vectors is not None:
        original_l = np.linalg.norm(vectors, axis=1)
        v = coordinates + (vectors / original_l[:, None]) * 0.1
        t = iM[:3, :3].dot(v.T) + iM[:3, 3, None]
        f = partial(
            scipy.ndimage.map_coordinates,
            coordinates=t,
            output=np.float32,
            order=intorder,
            mode="nearest",
        )
        v_trafo = np.array([f(df_data[..., i]) for i in range(3)])
        vec_transf = (v_trafo - coords) / 0.1
        if keep_length:
            vec_transf /= np.linalg.norm(vec_transf, axis=0)[None, :]
        vec_transf *= original_l[None, :]

        return coords.T, vec_transf.T
    return coords.T


def warp_coordinates(coordinates, mni2conf_nonl, out_name, out_geo, mesh, skin_tag=None):
    """Warp positions from MNI space to the subject scalp.

    Parameters
    ------------
    coordinates: str
        Name of the csv file with the positions in MNI space
    mni2conf_nonl: str or tuple
        Name of the non-linear MNI to subject deformation field, or its (data, affine)
    out_name: str
        Name of the output csv file
    out_geo: str
        Name of the output geo file, for visualization
    mesh: charm.mesh.msh.Msh
        Mesh with the surface to project the positions onto
    skin_tag: int
        Tag of that surface. Default: ElementTags.SCALP_TH_SURFACE
    """
    if skin_tag is None:
        skin_tag = ElementTags.SCALP_TH_SURFACE

    type_, coordinates, extra, name, extra_cols, header = read_csv_positions(coordinates)

    if isinstance(mni2conf_nonl, tuple):
        warp = mni2conf_nonl
    else:
        image = nifti.load(mni2conf_nonl)
        warp = (np.asanyarray(image.dataobj), image.affine)

    def to_scalp(points):
        return mesh.project_points_on_surface(coordinates_nonlinear(points, warp), surface_tags=skin_tag)

    transformed_coords = to_scalp(coordinates)

    # Electrodes may also carry a point that defines their orientation
    with_pos_y = [i for i, e in enumerate(extra) if e is not None]
    if with_pos_y:
        pos_y = to_scalp(np.array([extra[i] for i in with_pos_y]))
        for i, y in zip(with_pos_y, pos_y):
            extra[i] = y

    write_csv_positions(out_name, type_, transformed_coords, name, extra, extra_cols, header)
    csv_to_geo(out_name, out_geo)


def csv_to_geo(fn_csv, fn_out):
    """Write a .geo file based on a .csv file.

    Parameters
    -------
    fn_csv: str
       name of csv file
    fn_out: str
        name of output geo file
    """
    type_, coordinates, extra, name, _, _ = read_csv_positions(fn_csv)
    file_text = 'View"' + "" + '"{\n'

    def write_electrode(file_text, coordinates, extra, name):
        file_text += "SP(" + ", ".join([str(c) for c in coordinates]) + "){0};\n"
        if extra is not None:
            file_text += (
                "SL("
                + ", ".join([str(c) for c in coordinates])
                + ", "
                + ", ".join([str(c) for c in extra])
                + "){0, 0};\n"
            )
        if name is not None:
            text_coords = coordinates + [0.0, 0.0, 5.0]
            file_text += "T3(" + ", ".join([str(c) for c in text_coords]) + ', 0){"' + name + '"};\n'
        return file_text

    for t, c, e, n in zip(type_, coordinates, extra, name):
        file_text = write_electrode(file_text, c, e, n)

    file_text += "};\n"
    file_text += "myView = PostProcessing.NbViews-1;\n"
    file_text += "View[myView].PointType=1;\n"
    file_text += "View[myView].PointSize=6;\n"
    file_text += "View[myView].LineType=1;\n"
    file_text += "View[myView].LineWidth=2;\n"

    with open(fn_out, "w") as f:
        f.write(file_text)


def get_vox_size(affine):
    """Return the voxel sizes of an affine transformation."""
    return np.linalg.norm(affine[:3, :3], axis=0)


def crop_vol(vol, affine, mask, thickness_boundary=0):
    """Crop vol to the bounding box of mask and adjust the affine to match.

    Parameters
    ------------
    vol: 3D np.ndarray
        Volume to be cropped

    affine: 4x4 np.ndarray
        Affine transformation from the original volume to world space

    mask: 3D np.ndarray of floats
        Volume defining the bounding box

    thickness_boundary: float (optional)
        Thickness (in mm) to add to each side of the bounding box defined by the mask.
        Default: 0

    Returns
    ---------
    cropped_vol: 3D np.ndarray
        Cropped volume

    new_affine: 4x4 np.ndarray
        Affine transformation from the cropped volume to world space

    bound_box: 3x2 np.ndarray
        Bounding box used to crop the volume
    """
    non_zero = np.where(mask)
    bound_box = np.array([(np.min(nz), np.max(nz)) for nz in non_zero], dtype=int)

    vx_size = get_vox_size(affine)
    thickness_boundary = (thickness_boundary / vx_size).astype(int)

    bound_box[:, 0] -= thickness_boundary
    bound_box[bound_box[:, 0] < 0, 0] = 0

    bound_box[:, 1] += thickness_boundary
    out_of_bound = bound_box[:, 1] >= vol.shape
    bound_box[out_of_bound, 1] = np.array(vol.shape)[out_of_bound] - 1

    cropped_vol = vol[
        bound_box[0, 0] : bound_box[0, 1] + 1,
        bound_box[1, 0] : bound_box[1, 1] + 1,
        bound_box[2, 0] : bound_box[2, 1] + 1,
    ]

    new_affine = affine.copy()
    new_affine[:3, 3] += affine[:3, :3].dot(bound_box[:, 0])
    return cropped_vol, new_affine, bound_box


def pad_vol(vol, affine, voxel_pad):
    """Pad vol with voxel_pad zero voxels on every side and adjust the affine to match.

    Parameters
    ------------
    vol: 3D np.ndarray
        Volume to be padded

    affine: 4x4 np.ndarray
        Affine transformation from the original volume to world space

    voxel_pad: int
        The number of voxels the volume will be padded with all around

    Returns
    ---------
    padded_vol: 3D np.ndarray
        Padded volume

    new_affine: 4x4 np.ndarray
        Affine transformation from the padded volume to world space
    """
    padded_shape = (
        vol.shape[0] + 2 * voxel_pad,
        vol.shape[1] + 2 * voxel_pad,
        vol.shape[2] + 2 * voxel_pad,
    )
    padded_vol = np.zeros(padded_shape, dtype=vol.dtype)
    padded_vol[
        voxel_pad : voxel_pad + vol.shape[0],
        voxel_pad : voxel_pad + vol.shape[1],
        voxel_pad : voxel_pad + vol.shape[2],
    ] = vol

    new_affine = np.copy(affine)
    new_affine[:3, 3] = affine[:3, :3] @ [-voxel_pad, -voxel_pad, -voxel_pad] + affine[:3, 3]

    return padded_vol, new_affine


def resample_vol(vol, affine, target_res, order=1, mode="nearest"):
    """Change the resolution of the volume.

    Parameters
    -------------
    vol: 3D np.ndarray
        Volume to have the resolution changed

    affine: 4x4 np.ndarray
        Affine transformation from the original volume to world space

    target_res: 3x1 ndarray
        Target resolution

    order: int (optional)
        Interpolation order. Default: 1
        see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.affine_transform.html#scipy.ndimage.affine_transform

    mode: str (optional)
        How to handle boundaries. Default: 'nearest'
        see https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.affine_transform.html#scipy.ndimage.affine_transform

    Returns
    ------------
    resampled: 3D np.ndarray
        Resampled volume

    new_affine: 4x4 np.ndarray
        Affine transformation from the resampled volume to world space

    original_res: 3x1 ndarray
        Resolution of the ORIGINAL image (before resampling)
    """
    target_res = np.squeeze(target_res)
    if target_res.ndim == 0:
        target_res = target_res * np.ones(3)
    original_res = get_vox_size(affine)

    transform = np.squeeze(target_res / original_res)
    new_affine = affine.astype(float)
    new_affine[:3, :3] *= transform
    # Keep the outer corner of the volume in place by solving
    # R*(-0.5, -0.5, -0.5) + t  = R'*(-0.5, -0.5, -0.5) + t'
    corner = -0.5 * np.ones(3)
    offset = affine[:3, :3].dot(corner) - new_affine[:3, :3].dot(corner)
    new_affine[:3, 3] += offset
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        resampled = scipy.ndimage.affine_transform(
            vol,
            transform,
            # the voxel coordinates define the center of the voxels
            offset=(transform - 1) / 2.0,
            output_shape=(vol.shape / transform).astype(int),
            order=order,
            mode=mode,
        )

    return resampled, new_affine, original_res
