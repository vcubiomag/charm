import csv

import nibabel as nib
import numpy as np
import pytest
from conftest import TESTDATA

from charm.mesh.io import read_msh
from charm.utils import transformations


@pytest.fixture
def sphere3_msh():
    fn = str(TESTDATA / "sphere3.msh")
    return read_msh(fn)


@pytest.fixture
def image():
    x, y, z = np.meshgrid(np.arange(-5, 6), np.arange(-5, 6), np.arange(-5, 6), indexing="ij")
    image = np.array((x, y, z)).transpose(1, 2, 3, 0).astype(float)
    affine = np.array([[1, 0, 0, -5], [0, 1, 0, -5], [0, 0, 1, -5], [0, 0, 0, 1]], dtype=float)
    return image, affine


@pytest.fixture
def nonl_transfomation_id():
    x, y, z = np.meshgrid(np.arange(-6, 7), np.arange(-6, 7), np.arange(-6, 7), indexing="ij")
    nonl_transform = np.concatenate((x[..., None], y[..., None], z[..., None]), axis=3).astype(float)
    affine = np.array([[1, 0, 0, -6], [0, 1, 0, -6], [0, 0, 1, -6], [0, 0, 0, 1]], dtype=float)
    return nonl_transform, affine


@pytest.fixture
def nonl_transfomation_reflex():
    x, y, z = np.meshgrid(np.arange(-6, 7)[::-1], np.arange(-6, 7), np.arange(-6, 7), indexing="ij")
    nonl_transform = np.concatenate((x[..., None], y[..., None], z[..., None]), axis=3).astype(float)
    affine = np.array([[1, 0, 0, -6], [0, 1, 0, -6], [0, 0, 1, -6], [0, 0, 0, 1]], dtype=float)
    return nonl_transform, affine


@pytest.fixture
def affine_transformation():
    affine = np.array([[1, 0, 0, -1], [0, 1, 0, -1], [0, 0, 1, -1], [0, 0, 0, 1]], dtype=float)

    target_affine = np.array([[1, 0, 0, -6], [0, 1, 0, -6], [0, 0, 1, -6], [0, 0, 0, 1]], dtype=float)
    dim = [13, 13, 13]
    return affine, target_affine, dim


@pytest.fixture
def nonl_transfomation_rotate():
    x, y, z = np.meshgrid(np.arange(-6, 7, 2), np.arange(-6, 7, 2), np.arange(-6, 7, 2), indexing="ij")
    nonl_transform = np.concatenate((-y[..., None], x[..., None], z[..., None]), axis=3).astype(float)
    affine = np.array([[2, 0, 0, -6], [0, 2, 0, -6], [0, 0, 2, -6], [0, 0, 0, 1]], dtype=float)
    return nonl_transform, affine


@pytest.fixture
def nonl_transfomation_shear():
    x, y, z = np.meshgrid(np.arange(-6, 7), np.arange(-12, 13, 2), np.arange(-6, 7), indexing="ij")
    nonl_transform = np.concatenate((-y[..., None], x[..., None], z[..., None]), axis=3).astype(float)
    affine = np.array([[1, 0, 0, -6], [0, 1, 0, -6], [0, 0, 1, -6], [0, 0, 0, 1]], dtype=float)
    return nonl_transform, affine


@pytest.fixture
def nonl_large_rotate():
    x, y, z = np.meshgrid(
        np.linspace(-120, 120, 10),
        np.linspace(-120, 120, 10),
        np.linspace(-120, 120, 10),
        indexing="ij",
    )
    nonl_transform = np.concatenate((-y[..., None], x[..., None], z[..., None]), axis=3).astype(float)
    s = 240 / 9.0
    affine = np.array([[s, 0, 0, -120], [0, s, 0, -120], [0, 0, s, -120], [0, 0, 0, 1]], dtype=float)
    return nonl_transform, affine


@pytest.fixture
def affine_rotate():
    affine = np.array([[0, -1, 0, 1], [1, 0, 0, -1], [0, 0, 1, -1], [0, 0, 0, 1]], dtype=float)

    target_affine = np.array([[1, 0, 0, -6], [0, 1, 0, -6], [0, 0, 1, -6], [0, 0, 0, 1]], dtype=float)
    dim = [13, 13, 13]
    return affine, target_affine, dim


@pytest.fixture
def affine_reflex():
    affine = np.array([[-1, 0, 0, 1], [0, 1, 0, -1], [0, 0, 1, -1], [0, 0, 0, 1]], dtype=float)

    target_affine = np.array([[1, 0, 0, -6], [0, 1, 0, -6], [0, 0, 1, -6], [0, 0, 0, 1]], dtype=float)
    dim = [13, 13, 13]
    return affine, target_affine, dim


@pytest.fixture
def shear():
    affine = np.array([[0, -2, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)

    target_affine = np.array([[1, 0, 0, -6], [0, 1, 0, -6], [0, 0, 1, -6], [0, 0, 0, 1]], dtype=float)
    dim = [13, 13, 13]
    return affine, target_affine, dim


class TestNonlTransformVolume:
    def test_nonl_identity(self, image, nonl_transfomation_id):
        transformed = transformations.volumetric_nonlinear(image, nonl_transfomation_id)
        assert np.allclose(image[0], transformed[1:-1, 1:-1, 1:-1])

    def test_nonl_ressample(self, image, nonl_transfomation_id):
        t_affine = nonl_transfomation_id[1]
        dimensions = (13, 13, 13)
        transformed = transformations.volumetric_nonlinear(
            image,
            nonl_transfomation_id,
            target_space_affine=t_affine,
            target_dimensions=dimensions,
        )
        assert np.allclose(image[0], transformed[1:-1, 1:-1, 1:-1])

    def test_nonl_ressample2(self, image, nonl_transfomation_id):
        t_affine = nonl_transfomation_id[1].copy()
        t_affine[0, 0] = 2
        dimensions = [7, 13, 13]
        transformed = transformations.volumetric_nonlinear(
            image,
            nonl_transfomation_id,
            target_space_affine=t_affine,
            target_dimensions=dimensions,
        )
        assert np.allclose(image[0][1::2, :, :], transformed[1:-1, 1:-1, 1:-1])

    def test_nonl_reflex(self, image, nonl_transfomation_reflex):
        transformed = transformations.volumetric_nonlinear(image, nonl_transfomation_reflex)
        assert np.allclose(image[0][..., 0], transformed[-2:0:-1, 1:-1, 1:-1, 0])
        assert np.allclose(image[0][..., 1], transformed[-2:0:-1, 1:-1, 1:-1, 1])
        assert np.allclose(image[0][..., 2], transformed[-2:0:-1, 1:-1, 1:-1, 2])

        transformed = transformations.volumetric_nonlinear(
            image,
            nonl_transfomation_reflex,
            inverse_deformation=nonl_transfomation_reflex,
        )
        assert np.allclose(image[0][..., 0], -transformed[-2:0:-1, 1:-1, 1:-1, 0])
        assert np.allclose(image[0][..., 1], transformed[-2:0:-1, 1:-1, 1:-1, 1])
        assert np.allclose(image[0][..., 2], transformed[-2:0:-1, 1:-1, 1:-1, 2])

    def test_nonl_rotate(self, image, nonl_transfomation_rotate):
        transformed = transformations.volumetric_nonlinear(image, nonl_transfomation_rotate)
        assert np.allclose(
            image[0][..., 0][1::2, 1::2, 1::2],
            transformed[..., 0].transpose(1, 0, 2)[-2:0:-1, 1:-1, 1:-1],
        )
        assert np.allclose(
            image[0][..., 1][1::2, 1::2, 1::2],
            transformed[..., 1].transpose(1, 0, 2)[-2:0:-1, 1:-1, 1:-1],
        )
        assert np.allclose(
            image[0][..., 2][1::2, 1::2, 1::2],
            transformed[..., 2].transpose(1, 0, 2)[-2:0:-1, 1:-1, 1:-1],
        )

        inv = nonl_transfomation_rotate[0].copy()
        inv[..., 1] *= -1
        inv[..., 0] *= -1
        inv = (inv, nonl_transfomation_rotate[1])
        transformed = transformations.volumetric_nonlinear(image, nonl_transfomation_rotate, inverse_deformation=inv)
        assert np.allclose(
            -image[0][..., 0][1::2, 1::2, 1::2],
            transformed[..., 1].transpose(1, 0, 2)[-2:0:-1, 1:-1, 1:-1],
        )
        assert np.allclose(
            image[0][..., 1][1::2, 1::2, 1::2],
            transformed[..., 0].transpose(1, 0, 2)[-2:0:-1, 1:-1, 1:-1],
        )
        assert np.allclose(
            image[0][..., 2][1::2, 1::2, 1::2],
            transformed[..., 2].transpose(1, 0, 2)[-2:0:-1, 1:-1, 1:-1],
        )

    def test_nonl_shear(self, image, nonl_transfomation_shear):
        transformed = transformations.volumetric_nonlinear(image, nonl_transfomation_shear)
        assert np.allclose(
            image[0][..., 0][1::2, ...],
            transformed[..., 0].transpose(1, 0, 2)[-5:3:-1, 1:-1, 1:-1],
        )
        assert np.allclose(
            image[0][..., 1][1::2, ...],
            transformed[..., 1].transpose(1, 0, 2)[-5:3:-1, 1:-1, 1:-1],
        )
        assert np.allclose(
            image[0][..., 2][1::2, ...],
            transformed[..., 2].transpose(1, 0, 2)[-5:3:-1, 1:-1, 1:-1],
        )

        inv = nonl_transfomation_shear[0].copy()
        inv[..., 1] *= -0.5
        inv[..., 0] *= -0.5
        inv = (inv, nonl_transfomation_shear[1])
        transformed = transformations.volumetric_nonlinear(
            image,
            nonl_transfomation_shear,
            inverse_deformation=inv,
            keep_vector_length=False,
        )
        assert np.allclose(
            -0.5 * image[0][..., 0][1::2, ...],
            transformed[..., 1].transpose(1, 0, 2)[-5:3:-1, 1:-1, 1:-1],
        )
        assert np.allclose(
            image[0][..., 1][1::2, ...],
            transformed[..., 0].transpose(1, 0, 2)[-5:3:-1, 1:-1, 1:-1],
        )
        assert np.allclose(
            image[0][..., 2][1::2, ...],
            transformed[..., 2].transpose(1, 0, 2)[-5:3:-1, 1:-1, 1:-1],
        )

        transformed = transformations.volumetric_nonlinear(
            image,
            nonl_transfomation_shear,
            inverse_deformation=inv,
            keep_vector_length=True,
        )
        assert np.allclose(
            np.linalg.norm(image[0][1::2, ...], axis=3),
            np.linalg.norm(transformed[1:-1, -5:3:-1, 1:-1], axis=3).transpose(1, 0, 2),
        )


class TestAffineTransfom:
    def test_affine(self, image, affine_transformation):
        transformed = transformations.volumetric_affine(
            image,
            affine_transformation[0],
            affine_transformation[1],
            affine_transformation[2],
        )
        assert np.allclose(image[0], transformed[2:, 2:, 2:])

    def test_affine_reflex(self, image, affine_reflex):
        transformed = transformations.volumetric_affine(image, affine_reflex[0], affine_reflex[1], affine_reflex[2])
        assert np.allclose(image[0][..., 0], -transformed[-1:1:-1, 2:, 2:, 0])
        assert np.allclose(image[0][..., 1], transformed[-1:1:-1, 2:, 2:, 1])
        assert np.allclose(image[0][..., 2], transformed[-1:1:-1, 2:, 2:, 2])

    def test_affine_shear(self, image, shear):
        im = (image[0][..., 0], image[1])
        transformed = transformations.volumetric_affine(im, shear[0], shear[1], shear[2])
        assert np.allclose(image[0][-2:0:-2, :, :, 0].transpose(1, 0, 2), transformed[1:-1, 4:-4, 1:-1])

        transformed = transformations.volumetric_affine(image, shear[0], shear[1], shear[2], keep_vector_length=False)
        assert np.allclose(
            -0.5 * image[0][-2:0:-2, :, :, 0].transpose(1, 0, 2),
            transformed[1:-1, 4:-4, 1:-1, 1],
        )
        assert np.allclose(
            image[0][-2:0:-2, :, :, 1].transpose(1, 0, 2),
            transformed[1:-1, 4:-4, 1:-1, 0],
        )
        assert np.allclose(
            image[0][-2:0:-2, :, :, 2].transpose(1, 0, 2),
            transformed[1:-1, 4:-4, 1:-1, 2],
        )

        transformed = transformations.volumetric_affine(image, shear[0], shear[1], shear[2], keep_vector_length=True)
        assert np.allclose(
            np.linalg.norm(image[0][-2:0:-2, :, :, :], axis=3).transpose(1, 0, 2),
            np.linalg.norm(transformed[1:-1, 4:-4, 1:-1], axis=3),
        )


class TestCropVol:
    def test_crop_vol_simple(self):
        data = np.zeros((50, 100, 150))
        data[:] = np.arange(50)[:, None, None]

        mask = np.zeros(data.shape, dtype=bool)
        mask[10:20, 30:40, 50:60] = True

        affine = np.eye(4)
        affine[:3, :3] *= np.arange(1, 4)[::-1]

        cropped, new_affine, bb = transformations.crop_vol(data, affine, mask)
        assert cropped.shape == (10, 10, 10)
        assert np.allclose(cropped, np.arange(10, 20)[:, None, None])
        assert np.allclose(new_affine[:3, :3], affine[:3, :3])
        assert np.allclose(new_affine[:3, 3], [30, 60, 50])
        assert np.allclose(bb, [[10, 19], [30, 39], [50, 59]])

    def test_crop_vol_simple_boundary(self):
        data = np.zeros((50, 100, 200))
        data[:] = np.arange(50)[:, None, None]

        mask = np.zeros(data.shape, dtype=bool)
        mask[10:20, 30:40, 50:60] = True

        affine = np.eye(4)
        affine[:3, :3] *= [4, 2, 1]

        cropped, new_affine, bb = transformations.crop_vol(data, affine, mask, 4)

        assert cropped.shape == (12, 14, 18)
        assert np.allclose(cropped, np.arange(9, 21)[:, None, None])
        assert np.allclose(new_affine[:3, :3], affine[:3, :3])
        assert np.allclose(new_affine[:3, 3], [(10 - 1) * 4, (30 - 2) * 2, 50 - 4])
        assert np.allclose(bb, [[9, 20], [28, 41], [46, 63]])


class TestPadVol:
    def test_pad_simple(self):
        data = np.random.random((10, 20, 30))
        affine = np.eye(4)
        voxel_pad = 22

        padded, new_affine = transformations.pad_vol(data, affine, voxel_pad)

        assert padded.shape == (
            data.shape[0] + voxel_pad * 2,
            data.shape[1] + voxel_pad * 2,
            data.shape[2] + voxel_pad * 2,
        )
        assert (affine[0, 3] - new_affine[0, 3]) / (voxel_pad) == affine[0, 0]
        assert (affine[1, 3] - new_affine[1, 3]) / (voxel_pad) == affine[1, 1]
        assert (affine[2, 3] - new_affine[2, 3]) / (voxel_pad) == affine[2, 2]
        assert np.allclose(affine[:3, :3], new_affine[:3, :3])
        assert np.allclose(padded[22:32, 22:42, 22:52], data)


class TestGetVoxSize:
    def test_get_vox_size(self):
        voxsize = transformations.get_vox_size(np.array([[0, 0, 4, 0], [0, 2, 0, 1], [1, 0, 0, 5], [0, 0, 0, 1]]))
        assert np.allclose(voxsize, (1, 2, 4))


class TestResample:
    def test_resample_iso_0(self):
        vol = np.random.randint(0, 100, (50, 50, 50))
        vol = vol.astype(np.int32)
        affine = np.eye(4)
        target_res = 0.5

        resampled, _new_affine, orig_res = transformations.resample_vol(vol, affine, target_res, order=0)
        assert np.allclose(orig_res, 1.0)
        assert np.all(resampled.shape == (100, 100, 100))
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    assert np.all(resampled[i::2, j::2, k::2] == vol)

    def test_resample_iso(self):
        vol = np.zeros((50, 50, 50))
        vol[:] = np.arange(50)[:, None, None]
        affine = np.eye(4)
        target_res = 0.5

        resampled, new_affine, orig_res = transformations.resample_vol(vol, affine, target_res, order=1)
        assert np.allclose(orig_res, 1.0)
        assert np.allclose(resampled.shape, (100, 100, 100))
        assert np.allclose(resampled[1:-1], np.arange(0.25, 49, step=0.5)[:, None, None])
        assert np.allclose(new_affine[:3, :3], 0.5 * np.eye(3))
        assert np.allclose(new_affine[:3, 3], -0.25)
        assert np.allclose(new_affine[3, :], [0, 0, 0, 1])

    def test_resample_aniso(self):
        vol = np.zeros((40, 20, 80))
        vol[:] = np.arange(80, step=4)[None, :, None]
        affine = np.eye(4)
        affine[:3, :3] = np.array([[0, 4, 0], [2, 0, 0], [0, 0, 1]])
        target_res = np.array([1, 1, 1])

        resampled, new_affine, orig_res = transformations.resample_vol(vol, affine, target_res)

        assert np.allclose(orig_res, [2, 4, 1])
        assert np.allclose(resampled.shape, (80, 80, 80))
        assert np.allclose(resampled[:, 2:-2, :], np.arange(0.5, 76)[None, :, None])
        assert np.allclose(new_affine[:3, :3], [[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        assert np.allclose(new_affine[:3, 3], [-1.5, -0.5, 0])

    def test_decrease_res(self):
        vol = np.zeros((50, 50, 50))
        vol[:] = np.arange(50)[:, None, None]
        affine = np.eye(4)
        target_res = 2

        resampled, new_affine, orig_res = transformations.resample_vol(vol, affine, target_res, order=1)
        assert np.allclose(orig_res, [1, 1, 1])
        assert np.allclose(resampled.shape, (25, 25, 25))
        assert np.allclose(resampled, np.arange(0.5, 49, step=2)[:, None, None])
        assert np.allclose(new_affine[:3, :3], 2 * np.eye(3))
        assert np.allclose(new_affine[:3, 3], 0.5)
        assert np.allclose(new_affine[3, :], [0, 0, 0, 1])


def test_fix_boundary_zeros():
    # The one (0, 0, 0) voxel inside the volume is the origin and must stay.
    df_data = np.array(
        np.meshgrid(
            np.arange(-100, 51, 10, dtype=float),
            np.arange(-200, 101, 20, dtype=float),
            np.arange(-30, 51, 5, dtype=float),
            indexing="ij",
        )
    )
    df_data = df_data.reshape((3, -1)).T.reshape((16, 16, 17, 3))

    df_data = transformations._fix_boundary_zeros(df_data)

    assert np.sum(np.all(df_data == 0, axis=3)) == 1
    assert not np.any(np.isinf(df_data))
    assert np.max(df_data[:, :, :, 0]) == 50.0
    assert np.min(df_data[:, :, :, 0]) == -100.0
    assert np.max(df_data[:, :, :, 1]) == 100.0
    assert np.min(df_data[:, :, :, 1]) == -200.0
    assert np.max(df_data[:, :, :, 2]) == 50.0
    assert np.min(df_data[:, :, :, 2]) == -30.0

    # Zeros on the boundary along two axes are replaced, the origin still stays.
    df_data = np.array(
        np.meshgrid(
            np.arange(-100, 51, 10, dtype=float),
            np.arange(-200, 101, 20, dtype=float),
            np.arange(-30, 51, 5, dtype=float),
            indexing="ij",
        )
    )
    df_data = df_data.reshape((3, -1)).T.reshape((16, 16, 17, 3))

    df_data[0, :, :, :] = 0.0
    df_data[-2:, :, :, :] = 0.0
    df_data[:, :, 0, :] = 0.0
    df_data[:, :, -1, :] = 0.0

    df_data = transformations._fix_boundary_zeros(df_data)

    assert np.sum(np.all(df_data == 0, axis=3)) == 1
    assert np.sum(np.all(np.isinf(df_data), axis=3)) == 1232

    df_data[np.isinf(df_data)] = np.nan
    assert np.nanmax(df_data[:, :, :, 0]) == 30.0
    assert np.nanmin(df_data[:, :, :, 0]) == -90.0
    assert np.nanmax(df_data[:, :, :, 1]) == 100.0
    assert np.nanmin(df_data[:, :, :, 1]) == -200.0
    assert np.nanmax(df_data[:, :, :, 2]) == 45.0
    assert np.nanmin(df_data[:, :, :, 2]) == -25.0


class TestWarpCoordinates:
    def test_positions_land_on_the_scalp(self, sphere3_msh, tmp_path):
        cap = tmp_path / "cap.csv"
        cap.write_text("Type,X,Y,Z,Name\nFiducial,0,0,90,top\nElectrode,90,0,0,right\n")
        grid = np.stack(np.meshgrid(*[np.arange(-120.0, 121, 10)] * 3, indexing="ij"), -1)
        affine = np.diag([10.0, 10.0, 10.0, 1.0])
        affine[:3, 3] = -120
        warp = tmp_path / "warp.nii.gz"
        nib.save(nib.Nifti1Image(grid, affine), warp)

        out = tmp_path / "out.csv"
        geo = tmp_path / "out.geo"
        transformations.warp_coordinates(str(cap), str(warp), str(out), str(geo), sphere3_msh, skin_tag=1005)

        rows = list(csv.reader(out.read_text().splitlines()))
        assert [r[0] for r in rows[1:]] == ["Fiducial", "Electrode"]
        assert [r[-1] for r in rows[1:]] == ["top", "right"]
        coords = np.array([[float(x) for x in r[1:4]] for r in rows[1:]])
        # Identity warp, so these only move onto the outer surface, radius 95.
        assert np.allclose(np.linalg.norm(coords, axis=1), 95, atol=1)
        assert np.allclose(coords / np.linalg.norm(coords, axis=1, keepdims=True), [[0, 0, 1], [1, 0, 0]], atol=0.05)
        assert "T3(" in geo.read_text()
