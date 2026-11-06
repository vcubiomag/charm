import cat_surf
import nibabel as nib
import numpy as np
from conftest import TESTDATA

from charm.mesh.io import read_msh, write_gifti_surface
from charm.subject import SubjectFiles
from charm.surface.registration import spherical_registration


def test_spherical_registration_writes_both_spheres(tmp_path, monkeypatch):
    # Spherical Demons needs real cortical folding to converge in reasonable time.
    def identity_warp(source_surface, source_sphere, target_surface, target_sphere):
        return source_sphere

    monkeypatch.setattr(cat_surf, "spherical_demon", identity_warp)

    sub_files = SubjectFiles(subpath=str(tmp_path / "m2m_x"))
    (tmp_path / "m2m_x" / "surfaces").mkdir(parents=True)
    surface = read_msh(str(TESTDATA / "sphere3.msh")).crop_mesh(1005)
    write_gifti_surface(surface, sub_files.get_surface("lh", "white"))

    spherical_registration(sub_files, "lh")

    for name in ("sphere", "sphere.reg"):
        gii = nib.load(sub_files.get_surface("lh", name))
        assert isinstance(gii, nib.gifti.GiftiImage)
        vertices = gii.agg_data("pointset")
        assert vertices.shape == (surface.nodes.nr, 3)
        radii = np.linalg.norm(vertices - vertices.mean(0), axis=1)
        assert radii.std() / radii.mean() < 0.01
