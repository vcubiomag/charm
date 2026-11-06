import cortech
import numpy as np
from conftest import TESTDATA

from charm.mesh.io import read_msh
from charm.pipeline.run import _smoothed_mean_curvature


class _Hemisphere:
    def __init__(self, white, pial):
        self.white = white
        self.pial = pial


def _sphere(scale=1.0):
    m = read_msh(str(TESTDATA / "sphere3.msh")).crop_mesh(1005)
    used, faces = np.unique(m.elm.node_number_list[:, :3] - 1, return_inverse=True)
    return cortech.Surface(m.nodes.node_coord[used] * scale, faces.reshape(-1, 3))


def test_smoothed_mean_curvature_averages_white_and_pial():
    white, pial = _sphere(), _sphere(1.1)
    curv = _smoothed_mean_curvature(_Hemisphere(white, pial))
    assert curv.shape == (white.n_vertices,)
    # Mean curvature of a sphere is 1 / radius, up to discretization.
    radius = np.linalg.norm(white.vertices, axis=1).mean()
    assert np.isclose(np.abs(curv).mean(), 0.5 * (1 / radius + 1 / (1.1 * radius)), rtol=0.1)
    # Smoothing must have run: the raw estimate varies more on this coarse mesh.
    raw = 0.5 * (white.curvature().H + pial.curvature().H)
    assert curv.std() < raw.std()
