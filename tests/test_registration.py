import brainsynth.resources
import cortech
import nibabel as nib
import numpy as np
import pytest
from conftest import TESTDATA
from scipy.spatial.transform import Rotation

from charm.mesh.io import read_msh, write_gifti_surface
from charm.subject import SubjectFiles
from charm.surface import registration


def folded_triangles(vertices, faces):
    volume = np.einsum("ij,ij->i", vertices[faces[:, 0]], np.cross(vertices[faces[:, 1]], vertices[faces[:, 2]]))
    return int((volume * np.sign(np.median(volume)) <= 0).sum())


def median_distance_mm(a, b):
    return np.median(100 * np.arccos(np.clip(np.sum(a * b, 1), -1, 1)))


def test_spherical_registration_recovers_the_template_registration(tmp_path, monkeypatch):
    # The TopoFit template comes with FreeSurfer's registration of it to fsaverage, so registering its white surface
    # from a rotated sphere should bring the sphere back.
    template = brainsynth.resources.Template().load_surface("lh", "white")
    white = cortech.Surface(template["vertices"].numpy().astype(float), template["faces"].numpy())
    truth = registration.template_sphere(white, "lh")
    start = Rotation.from_rotvec(np.radians(8) * np.array([1, 1, 0]) / np.sqrt(2)).apply(truth)
    monkeypatch.setattr(registration, "template_sphere", lambda white, hemi: start)

    sub_files = SubjectFiles(subpath=str(tmp_path / "m2m_x"))
    (tmp_path / "m2m_x" / "surfaces").mkdir(parents=True)
    write_gifti_surface(white, sub_files.get_surface("lh", "white"))
    registration.spherical_registration(sub_files, "lh")

    spheres = {}
    for name in ("sphere", "sphere.reg"):
        gii = nib.load(sub_files.get_surface("lh", name))
        assert isinstance(gii, nib.gifti.GiftiImage)
        vertices, faces = gii.agg_data(("pointset", "triangle"))
        np.testing.assert_allclose(np.linalg.norm(vertices, axis=1), 1, atol=1e-5)
        np.testing.assert_array_equal(faces, white.faces)
        spheres[name] = vertices.astype(float)
    assert median_distance_mm(spheres["sphere"], truth) > 10
    assert median_distance_mm(spheres["sphere.reg"], truth) < 3
    assert folded_triangles(spheres["sphere.reg"], white.faces) < 50


def test_template_sphere_needs_the_topofit_mesh():
    surface = read_msh(str(TESTDATA / "sphere3.msh")).crop_mesh(1005)
    white = cortech.Surface(surface.nodes.node_coord, surface.elm.node_number_list[:, :3] - 1)
    with pytest.raises(ValueError, match="TopoFit template mesh"):
        registration.template_sphere(white, "lh")
