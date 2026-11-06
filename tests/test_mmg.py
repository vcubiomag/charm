import numpy as np
import pytest
from conftest import TESTDATA

from charm.mesh import meshing
from charm.mesh._mmg3d import remesh
from charm.mesh.io import read_msh


@pytest.fixture
def sphere3():
    return read_msh(str(TESTDATA / "sphere3.msh")).crop_mesh(elm_type=[2, 4])


def _tissue_volumes(m):
    volumes = m.elements_volumes_and_areas().value
    tets = m.elm.elm_type == 4
    return {int(t): volumes[tets & (m.elm.tag1 == t)].sum() for t in np.unique(m.elm.tag1[tets])}


def test_run_mmg_preserves_surfaces_and_tissue_volumes(sphere3):
    tris_before = np.sort(sphere3.elm.tag1[sphere3.elm.elm_type == 2])
    volumes_before = _tissue_volumes(sphere3)

    m = meshing._run_mmg(sphere3, repeats=1, mmg_noinsert=False)

    assert np.array_equal(np.sort(m.elm.tag1[m.elm.elm_type == 2]), tris_before)
    volumes = _tissue_volumes(m)
    assert volumes.keys() == volumes_before.keys()
    for tag, volume in volumes.items():
        assert np.isclose(volume, volumes_before[tag], rtol=1e-6)
    assert np.array_equal(m.elm.tag1, m.elm.tag2)
    assert np.all(m.elm.node_number_list[m.elm.elm_type == 2, 3] == -1)
    assert m.elm.node_number_list[m.elm.elm_type == 4].min() >= 1


def test_noinsert_does_not_add_nodes(sphere3):
    n_nodes = len(np.unique(sphere3.elm.node_number_list[sphere3.elm.elm_type == 4]))
    m = meshing._run_mmg(sphere3, repeats=1, mmg_noinsert=True)
    assert m.nodes.nr <= n_nodes


def test_remesh_rejects_mismatched_tags():
    nodes = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    tets = np.array([[1, 2, 3, 4]], dtype=np.int64)
    with pytest.raises(ValueError, match="one tag"):
        remesh(
            nodes,
            tets,
            np.array([1, 2], dtype=np.int64),
            np.zeros((0, 3), dtype=np.int64),
            np.zeros(0, dtype=np.int64),
        )


def test_spatial_ordering_preserves_the_mesh(sphere3):
    is_tet = sphere3.elm.elm_type == 4
    is_tri = sphere3.elm.elm_type == 2
    nodes = sphere3.nodes.node_coord
    tets = sphere3.elm.node_number_list[is_tet].astype(np.int64)
    tags = sphere3.elm.tag1[is_tet].astype(np.int64)
    tris = sphere3.elm.node_number_list[is_tri, :3].astype(np.int64)

    nodes2, tets2, tags2, tris2 = meshing._spatially_ordered(nodes, tets, tags, tris)

    def tet_positions(n, t):
        # Each element as a canonical set of coordinates, independent of numbering.
        corners = n[t - 1].round(6)
        return np.sort(corners.reshape(len(t), -1, 3).sum(axis=1), axis=0)

    assert np.allclose(tet_positions(nodes, tets), tet_positions(nodes2, tets2))
    assert np.allclose(nodes[tris - 1].sum(axis=1), nodes2[tris2 - 1].sum(axis=1))
    assert np.array_equal(np.sort(tags), np.sort(tags2))
    # Neighboring tetrahedra now sit next to each other in memory.
    centers = nodes2[tets2 - 1].mean(axis=1)
    original = nodes[tets - 1].mean(axis=1)
    jump = lambda c: np.linalg.norm(np.diff(c, axis=0), axis=1).mean()
    assert jump(centers) < jump(original)
