import numpy as np
import pytest
from conftest import TESTDATA

from charm.mesh.io import read_msh, write_msh
from charm.mesh.msh import Elements, Msh, Nodes


@pytest.mark.parametrize("mode", ["binary", "ascii"])
def test_write_and_read_back_every_element_type(tmp_path, mode):
    nodes = np.random.default_rng(0).normal(size=(6, 3))
    elements = Elements(triangles=np.array([[1, 2, 3], [2, 3, 4]]), tetrahedra=np.array([[1, 2, 3, 4], [2, 3, 4, 5]]))
    elements.tag1 = np.array([1001, 1002, 1, 2])
    elements.tag2 = elements.tag1.copy()
    points = Elements(points=np.array([6]))
    points.tag1 = np.array([7])
    points.tag2 = np.array([7])
    lines = Elements(lines=np.array([[5, 6]]))
    lines.tag1 = np.array([8])
    lines.tag2 = np.array([8])
    mesh = Msh(Nodes(nodes), elements)
    for other in (lines, points):
        mesh.elm.node_number_list = np.vstack((other.node_number_list, mesh.elm.node_number_list))
        mesh.elm.tag1 = np.hstack((other.tag1, mesh.elm.tag1))
        mesh.elm.tag2 = np.hstack((other.tag2, mesh.elm.tag2))
        mesh.elm.elm_type = np.hstack((other.elm_type, mesh.elm.elm_type))

    fn = str(tmp_path / "mesh.msh")
    write_msh(mesh, fn, mode=mode)
    copy = read_msh(fn)

    np.testing.assert_array_equal(copy.nodes.node_coord, nodes)
    np.testing.assert_array_equal(copy.elm.elm_type, mesh.elm.elm_type)
    np.testing.assert_array_equal(copy.elm.tag1, mesh.elm.tag1)
    np.testing.assert_array_equal(copy.elm.tag2, mesh.elm.tag2)
    np.testing.assert_array_equal(copy.elm.node_number_list, mesh.elm.node_number_list)


def test_binary_file_is_unchanged_by_a_round_trip(tmp_path):
    fn = str(tmp_path / "copy.msh")
    source = read_msh(str(TESTDATA / "sphere3.msh"))
    write_msh(source, fn)
    copy = read_msh(fn)
    np.testing.assert_array_equal(copy.nodes.node_coord, source.nodes.node_coord)
    np.testing.assert_array_equal(copy.elm.node_number_list, source.elm.node_number_list)
    np.testing.assert_array_equal(copy.elm.tag1, source.elm.tag1)
