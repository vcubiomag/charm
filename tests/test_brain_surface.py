import numpy as np
import pytest
from conftest import TESTDATA

from charm.mesh.io import read_msh
from charm.surface import brain_surface


@pytest.fixture
def sphere_surf():
    fn = str(TESTDATA / "sphere3.msh")
    mesh = read_msh(fn)
    return mesh.crop_mesh(1005)


@pytest.fixture
def sphere_2_surfs():
    fn = str(TESTDATA / "sphere3.msh")
    mesh = read_msh(fn)
    return mesh.crop_mesh([1004, 1005])


def one_sphere_image():
    """Nested spheres with WM 3, GM 2 and CSF 1. Returns (img, vox2mm, radius_wm, radius_gm, radius_csf)."""

    imgsize = [161, 141, 141]
    voxsize = 0.25

    radius_wm = 8
    radius_gm = 12
    radius_csf = 15

    vox2mm = np.array(
        (
            [voxsize, 0, 0, -voxsize * (imgsize[0] - 1) / 2],
            [0, voxsize, 0, -voxsize * (imgsize[1] - 1) / 2],
            [0, 0, voxsize, -voxsize * (imgsize[2] - 1) / 2],
            [0, 0, 0, 1],
        ),
        dtype=np.float32,
    )

    G = np.meshgrid(
        np.arange(imgsize[0]),
        np.arange(imgsize[1]),
        np.arange(imgsize[2]),
        indexing="ij",
    )

    xyz_mm = np.inner(
        vox2mm,
        np.vstack(
            (
                G[0].flatten(),
                G[1].flatten(),
                G[2].flatten(),
                np.ones_like(G[0].flatten()),
            )
        ).T,
    )[0:3]

    R_sp = np.linalg.norm(xyz_mm, axis=0)

    img = np.zeros_like(R_sp)
    img[R_sp < radius_csf] = 1
    img[R_sp < radius_gm] = 2
    img[R_sp < radius_wm] = 3
    img = img.reshape(imgsize)

    return img, vox2mm, radius_wm, radius_gm, radius_csf


class TestSegmentTriangleIntersect:
    def test_segment_triangle_intersect(self, sphere_surf):
        bar = sphere_surf.elements_baricenters()[:]
        normals = sphere_surf.triangle_normals()[:]
        intersect, int_points = brain_surface.segment_triangle_intersect(
            sphere_surf.nodes[:],
            sphere_surf.elm[:, :3] - 1,
            bar - 1e-1 * normals,
            bar + 1e-1 * normals,
        )
        assert np.all(intersect[:, 0] == intersect[:, 1])
        assert np.allclose(int_points, bar)

    def test_segment_triangle_no_intersect(self, sphere_surf):
        bar = sphere_surf.elements_baricenters()[:]
        normals = sphere_surf.triangle_normals()[:]
        intersect, _int_points = brain_surface.segment_triangle_intersect(
            sphere_surf.nodes[:],
            sphere_surf.elm[:, :3] - 1,
            bar + 1e-1 * normals,
            bar + 2e-1 * normals,
        )

        assert len(intersect) == 0


class TestCreateSurfaceMask:
    @pytest.mark.parametrize("axis", ["z", "y", "x"])
    def test_rasterize_surface(self, axis, sphere_surf):
        vertices = sphere_surf.nodes[:]
        faces = sphere_surf.elm[:, :3] - 1
        affine = np.eye(4)
        # Off the voxel grid, because the mesh has nodes on the x, y and z axes.
        affine[:3, 3] = -99.9
        shape = [200, 200, 200]
        mask = brain_surface._rasterize_surface(vertices, faces, affine, shape, axis=axis)
        assert np.all(mask[100, 100, 5:195])
        assert np.all(mask[100, 5:195, 100])
        assert np.all(mask[5:195, 100, 100])
        assert not np.any(mask[100, 100, :5]) and not np.any(mask[100, 100, 195:])
        assert not np.any(mask[100, :5, 100]) and not np.any(mask[100, 195:, 100])
        assert not np.any(mask[:5, 100, 100]) and not np.any(mask[195:, 100, 100])

    @pytest.mark.parametrize("axis", ["z", "y", "x"])
    def test_rasterize_surface_concave(self, axis, sphere_2_surfs):
        vertices = sphere_2_surfs.nodes[:]
        faces = sphere_2_surfs.elm[:, :3] - 1
        affine = np.eye(4)
        affine[:3, 3] = -99.9
        shape = [200, 200, 200]
        mask = brain_surface._rasterize_surface(vertices, faces, affine, shape, axis=axis)
        assert np.all(mask[100, 100, 5:10]) and np.all(mask[100, 100, 190:195])
        assert np.all(mask[100, 5:10, 100]) and np.all(mask[100, 190:195, 100])
        assert np.all(mask[5:10, 100, 100]) and np.all(mask[190:195, 100, 100])
        assert not np.any(mask[100, 100, 10:190])
        assert not np.any(mask[100, 10:190, 100])
        assert not np.any(mask[10:190, 100, 100])

    @pytest.mark.parametrize("axis", ["z", "y", "x"])
    def test_rasterize_surface_aniso(self, axis, sphere_surf):
        vertices = sphere_surf.nodes[:]
        faces = sphere_surf.elm[:, :3] - 1
        affine = np.diag([1.0, 1.0, 2.0, 1])
        affine[:3, 3] = -99.9
        shape = [200, 200, 100]
        mask = brain_surface._rasterize_surface(vertices, faces, affine, shape, axis=axis)
        assert np.all(mask[100, 100, 3:97])
        assert np.all(mask[100, 5:195, 50])
        assert np.all(mask[5:195, 100, 50])
        assert not np.any(mask[100, 100, :3]) and not np.any(mask[100, 100, 98:])
        assert not np.any(mask[100, :5, 50]) and not np.any(mask[100, 195:, 50])
        assert not np.any(mask[:5, 100, 50]) and not np.any(mask[195:, 100, 50])

    @pytest.mark.parametrize("axis", ["z", "y", "x"])
    def test_rasterize_surface_slice(self, axis, sphere_surf):
        vertices = sphere_surf.nodes[:]
        faces = sphere_surf.elm[:, :3] - 1
        affine = np.eye(4)
        affine[:3, 3] = [-99.9, -99.9, 0.1]
        shape = [200, 200, 1]
        mask = brain_surface._rasterize_surface(vertices, faces, affine, shape, axis=axis)
        assert np.all(mask[100, 5:195, 0])
        assert np.all(mask[5:195, 100, 0])
        assert not np.any(mask[100, :5, 0]) and not np.any(mask[100, 195:, 0])
        assert not np.any(mask[:5, 100, 0]) and not np.any(mask[195:, 100, 0])

    def test_rasterize_surface_bad_topology(self, sphere_surf):
        vertices = sphere_surf.nodes[:]
        faces = sphere_surf.elm[:, :3] - 1
        # Clone part of the faces to create an "outer shell" in part of the sphere
        faces_bar = np.average(vertices[faces], axis=1)
        faces_to_clone = faces[faces_bar[:, 2] > 50]
        verts_to_clone, reverse = np.unique(faces_to_clone, return_inverse=True)
        verts_cloned = vertices[verts_to_clone] + [0, 0, 2]
        faces_cloned = reverse.reshape(-1, 3) + len(vertices)
        faces = np.vstack([faces, faces_cloned])
        vertices = np.vstack([vertices, verts_cloned])
        affine = np.eye(4)
        affine[:3, 3] = -99.9
        shape = [200, 200, 200]
        mask = brain_surface._rasterize_surface(vertices, faces, affine, shape)
        assert np.all(mask[100, 100, 5:195])
        assert np.all(mask[100, 5:195, 100])
        assert np.all(mask[5:195, 100, 100])
        assert not np.any(mask[100, 100, :5]) and not np.any(mask[100, 100, 195:])
        assert not np.any(mask[100, :5, 100]) and not np.any(mask[100, 195:, 100])
        assert not np.any(mask[:5, 100, 100]) and not np.any(mask[195:, 100, 100])

    def test_mask_from_surface(self, sphere_surf):
        vertices = sphere_surf.nodes[:]
        faces = sphere_surf.elm[:, :3] - 1
        affine = np.eye(4)
        affine[:3, 3] = -99.9
        shape = [200, 200, 200]
        mask = brain_surface.mask_from_surface(vertices, faces, affine, shape)
        assert np.all(mask[100, 100, 5:195])
        assert np.all(mask[100, 5:195, 100])
        assert np.all(mask[5:195, 100, 100])
        assert not np.any(mask[100, 100, :5]) and not np.any(mask[100, 100, 195:])
        assert not np.any(mask[100, :5, 100]) and not np.any(mask[100, 195:, 100])
        assert not np.any(mask[:5, 100, 100]) and not np.any(mask[195:, 100, 100])

    def test_mask_from_surface_concave(self, sphere_2_surfs):
        vertices = sphere_2_surfs.nodes[:]
        faces = sphere_2_surfs.elm[:, :3] - 1
        affine = np.eye(4)
        affine[:3, 3] = -99.9
        shape = [200, 200, 200]
        mask = brain_surface.mask_from_surface(vertices, faces, affine, shape)
        assert np.all(mask[100, 100, 5:10]) and np.all(mask[100, 100, 190:195])
        assert np.all(mask[100, 5:10, 100]) and np.all(mask[100, 190:195, 100])
        assert np.all(mask[5:10, 100, 100]) and np.all(mask[190:195, 100, 100])
        assert not np.any(mask[100, 100, 10:190])
        assert not np.any(mask[100, 10:190, 100])
        assert not np.any(mask[10:190, 100, 100])

    def test_mask_from_surface_bad_topology(self, sphere_surf):
        vertices = sphere_surf.nodes[:]
        faces = sphere_surf.elm[:, :3] - 1
        # Clone part of the faces to create an "outer shell" in part of the sphere
        faces_bar = np.average(vertices[faces], axis=1)
        faces_to_clone = faces[faces_bar[:, 2] > 93]
        verts_to_clone, reverse = np.unique(faces_to_clone, return_inverse=True)
        verts_cloned = vertices[verts_to_clone] + [0, 0, 3]
        faces_cloned = reverse.reshape(-1, 3) + len(vertices)
        faces = np.vstack([faces, faces_cloned])
        vertices = np.vstack([vertices, verts_cloned])
        affine = np.eye(4)
        affine[:3, 3] = -99.9
        shape = [200, 200, 200]
        mask = brain_surface.mask_from_surface(vertices, faces, affine, shape)
        assert np.all(mask[100, 100, 5:195])
        assert np.all(mask[100, 5:195, 100])
        assert np.all(mask[5:195, 100, 100])
        assert not np.any(mask[100, :5, 100]) and not np.any(mask[100, 195:, 100])
        assert not np.any(mask[:5, 100, 100]) and not np.any(mask[195:, 100, 100])
