import numpy as np

from charm.mesh.io import read_medit


def test_read_medit_skips_cgal_comments(tmp_path):
    fn = tmp_path / "mesh.mesh"
    fn.write_text(
        "MeshVersionFormatted 1\n"
        "Dimension 3\n"
        "# CGAL::Mesh_complex_3_in_triangulation_3\n"
        "Vertices\n4\n"
        "0 0 0 1\n1 0 0 1\n0 1 0 1\n0 0 1 1\n"
        "Triangles\n1\n1 2 3 7\n"
        "Tetrahedra\n1\n1 2 3 4 2\n"
        "End\n"
    )
    m = read_medit(fn)
    assert m.nodes.nr == 4
    assert np.all(m.elm.elm_type == [2, 4])
    assert np.all(m.elm.tag1 == [7, 2])
