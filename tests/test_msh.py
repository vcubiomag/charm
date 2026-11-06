import copy
import os

import numpy as np
import pytest
from conftest import TESTDATA
from scipy.spatial import ConvexHull

from charm.mesh.element_tags import ElementTags, ElementTypes
from charm.mesh.io import read_gifti_surface, read_msh, write_gifti_surface
from charm.mesh.msh import ElementData, Elements, Msh, Nodes, _hash_rows


@pytest.fixture
def simple_element():
    triangles = np.arange(1, 13).reshape(4, 3)
    tetrahedra = np.arange(1, 17).reshape(4, 4)
    elm = Elements(triangles, tetrahedra)
    elm.tag1 = elm.tag2 = np.array([1, 1, 2, 3, 11, 12, 11, 13])
    return elm


def get_angle(v1, v2):
    angle = np.sum(v1 * v2, axis=1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1))
    angle[angle > 1.0] = 1.0
    angle = np.arccos(angle) / np.pi * 180
    return angle


@pytest.fixture(scope="module")
def sphere3_msh():
    fn = str(TESTDATA / "sphere3.msh")
    return read_msh(fn)


@pytest.fixture(scope="module")
def sphere3_baricenters(sphere3_msh):
    baricenters = np.zeros((sphere3_msh.elm.nr, 3), dtype=float)
    th_indexes = sphere3_msh.elm.get_tetrahedra(return_indices=True)
    baricenters[th_indexes] = np.average(
        sphere3_msh.nodes.node_coord[sphere3_msh.elm.node_number_list[th_indexes, :4] - 1],
        axis=1,
    )

    tr_indexes = sphere3_msh.elm.get_triangles(return_indices=True)
    baricenters[tr_indexes] = np.average(
        sphere3_msh.nodes.node_coord[sphere3_msh.elm.node_number_list[tr_indexes, :3] - 1],
        axis=1,
    )
    return baricenters


def toCartesian(pts):
    r = pts[:, 0]
    phi = pts[:, 1]
    theta = pts[:, 2]

    pts_cart = np.zeros(pts.shape, dtype="float64")
    pts_cart[:, 0] = r * np.cos(phi * np.pi / 180.0) * np.sin(theta * np.pi / 180.0)
    pts_cart[:, 1] = r * np.sin(phi * np.pi / 180.0) * np.sin(theta * np.pi / 180.0)
    pts_cart[:, 2] = r * np.cos(theta * np.pi / 180.0)
    return pts_cart


class TestGmshRead:
    def test_node_nr(self, sphere3_msh):
        assert sphere3_msh.nodes.nr == 4556

    def test_node_coord(self, sphere3_msh):
        np.testing.assert_array_almost_equal(
            sphere3_msh.nodes.node_coord[-1, :],
            np.array([-27.4983367, 72.72180083, 9.66883006]),
        )

    def test_elm_nr(self, sphere3_msh):
        assert sphere3_msh.elm.nr == 30530

    def test_elm_tag1(self, sphere3_msh):
        assert np.unique(sphere3_msh.elm.tag1).tolist() == [3, 4, 5, 1003, 1004, 1005]

    def test_elm_node_number_list(self, sphere3_msh):
        np.testing.assert_array_equal(sphere3_msh.elm.node_number_list[-1, :], np.array([31, 4149, 4272, 1118]))


class TestNodes:
    def test_node_init_nrs(self):
        node_coord = np.array(((3, 2, 4), (5, 3, 3)))
        nodes = Nodes(node_coord)
        assert nodes.nr == 2

    def test_node_init_node_nr(self):
        node_coord = np.array(((3.0, 2.0, 4), (5, 3, 3)))
        nodes = Nodes(node_coord)
        np.testing.assert_array_almost_equal(nodes.node_number, np.array((1, 2)))

    def test_getitem_simple_index(self):
        nodes = Nodes()
        nodes.node_coord = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        assert np.all(nodes[1] == [1, 0, 0])

    def test_getitem_list_index(self):
        nodes = Nodes()
        nodes.node_coord = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        assert np.all(nodes[[1, 2]] == [[1, 0, 0], [0, 1, 0]])
        assert np.all(nodes[np.array([1, 2])] == [[1, 0, 0], [0, 1, 0]])

    def test_getitem_slice(self):
        nodes = Nodes()
        nodes.node_coord = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        assert np.all(nodes[1:] == [[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    def test_getitem_raise(self):
        nodes = Nodes()
        nodes.node_coord = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        with pytest.raises(IndexError):
            nodes[0]


class TestElements:
    def test_elm_init_nr(self):
        triangles = np.array(((1, 3, 2), (4, 1, 2)))
        tetrahedra = np.array(((1, 3, 2, 4), (4, 1, 3, 7)))
        elm = Elements(triangles, tetrahedra)
        assert elm.nr == 4
        assert np.all(elm[:] == np.array([[1, 3, 2, -1], [4, 1, 2, -1], [1, 3, 2, 4], [4, 1, 3, 7]]))

    @pytest.mark.parametrize("tags", [1, (1, 2), [11, 12]])
    @pytest.mark.parametrize("invert", [False, True])
    @pytest.mark.parametrize(
        ("return_element_numbers", "return_indices"),
        ([False, False], [False, True], [True, False]),
    )
    def test_get_tags(self, simple_element, tags, invert, return_element_numbers, return_indices):
        target = np.isin(simple_element.tag1, tags, invert=invert)
        if return_element_numbers:
            target = np.where(target)[0] + 1
        elif return_indices:
            target = np.where(target)[0]

        np.testing.assert_allclose(
            target,
            simple_element.get_tags(
                tags,
                invert,
                return_element_numbers=return_element_numbers,
                return_indices=return_indices,
            ),
        )

    @pytest.mark.parametrize("tags", [None, 1, (1, 2), [11, 12]])
    @pytest.mark.parametrize(
        ("return_element_numbers", "return_indices"),
        ([False, False], [False, True], [True, False]),
    )
    def test_get_triangles(self, simple_element, tags, return_element_numbers, return_indices):
        target = np.isin(simple_element.elm_type, ElementTypes.TRIANGLE)
        if tags is not None:
            target &= np.isin(simple_element.tag1, tags)

        if return_element_numbers:
            target = np.where(target)[0] + 1
        elif return_indices:
            target = np.where(target)[0]

        np.testing.assert_allclose(
            target,
            simple_element.get_triangles(
                tags,
                return_element_numbers=return_element_numbers,
                return_indices=return_indices,
            ),
        )

    @pytest.mark.parametrize("tags", [None, 11, (1, 2), [11, 12]])
    @pytest.mark.parametrize(
        ("return_element_numbers", "return_indices"),
        ([False, False], [False, True], [True, False]),
    )
    def test_get_tetrahedra(self, simple_element, tags, return_element_numbers, return_indices):
        target = np.isin(simple_element.elm_type, ElementTypes.TETRAHEDRON)
        if tags is not None:
            target &= np.isin(simple_element.tag1, tags)

        if return_element_numbers:
            target = np.where(target)[0] + 1
        elif return_indices:
            target = np.where(target)[0]

        np.testing.assert_allclose(
            target,
            simple_element.get_tetrahedra(
                tags,
                return_element_numbers=return_element_numbers,
                return_indices=return_indices,
            ),
        )

    def test_elm_init_node_number_list(self):
        triangles = np.array(((1, 3, 2), (4, 1, 2)))
        tetrahedra = np.array(((1, 3, 2, 4), (4, 1, 3, 7)))
        elm = Elements(triangles, tetrahedra)
        np.testing.assert_array_almost_equal(
            elm.node_number_list,
            np.array(((1, 3, 2, -1), (4, 1, 2, -1), (1, 3, 2, 4), (4, 1, 3, 7))),
        )

    def test_elements_getitem(self):
        elm = Elements()
        elm.node_number_list = np.array([[1, 2, 3, 4], [3, 2, 5, 7], [1, 8, 2, 6]])
        assert np.all(elm[[2, 3]] == np.array([[3, 2, 5, 7], [1, 8, 2, 6]]))

    def test_get_faces(self, sphere3_msh):
        elm = sphere3_msh.elm
        faces, th_faces, adjacency_list = elm.get_faces()
        th = elm[elm.tetrahedra]
        faces_th = th[:, [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]]
        assert np.all(faces_th[0] == faces[th_faces[0]])
        assert np.sum(np.isin(th[adjacency_list[0, 0]], th[adjacency_list[0, 1]])) == 3
        assert np.any(np.isin(th_faces[0], th_faces[adjacency_list[th_faces[0, 0], 1]]))

        # The orientation might be flipped, so the order is ignored.
        assert np.all(np.unique(faces_th[-1]) == np.unique(faces[th_faces[-1]]))
        assert np.sum(np.isin(th[adjacency_list[-1, 0]], th[adjacency_list[-1, 1]])) == 3
        assert np.any(np.isin(th_faces[-1], th_faces[adjacency_list[th_faces[-1, 0], 1]]))

    def test_connected_components_connected_mesh(self, sphere3_msh):
        cc = sphere3_msh.elm.connected_components()
        assert len(cc) == 1
        assert np.all(cc[0] == sphere3_msh.elm.elm_number)

    def test_connected_components_disconnected_th(self, sphere3_msh):
        elm_select = sphere3_msh.elm.elm_number[sphere3_msh.elm.get_tags(3) | sphere3_msh.elm.get_tags(5)]
        cc = sphere3_msh.elm.connected_components(elm_select)
        assert len(cc) == 2
        assert np.all(cc[0] == sphere3_msh.elm.get_tags(3, return_element_numbers=True))
        assert np.all(cc[1] == sphere3_msh.elm.get_tags(5, return_element_numbers=True))

    def test_connected_components_disconnected_th_tr(self, sphere3_msh):
        elm_select = sphere3_msh.elm.elm_number[sphere3_msh.elm.get_tags(3) | sphere3_msh.elm.get_tags(5)]
        cc = sphere3_msh.elm.connected_components(elm_select)
        assert len(cc) == 2
        assert np.all(cc[0] == sphere3_msh.elm.get_tags(3, return_element_numbers=True))
        assert np.all(cc[1] == sphere3_msh.elm.get_tags(5, return_element_numbers=True))

    def test_add_triangles(self, sphere3_msh):
        m = copy.deepcopy(sphere3_msh)
        m.elm.add_triangles(m.elm[m.elm.get_tags(1005), :3], 1006)
        assert m.elm.nr == sphere3_msh.elm.nr + m.elm.get_tags(1005).sum()
        assert np.all(m.elm[m.elm.get_tags(1006)] == sphere3_msh.elm[sphere3_msh.elm.get_tags(1005)])

    def test_add_triangles_tags(self, sphere3_msh):
        m = copy.deepcopy(sphere3_msh)
        tr = m.elm[m.elm.get_tags(1005), :3]
        tags = m.elm.tag1[m.elm.get_tags(1005)] + 1
        m.elm.add_triangles(tr, tags)
        assert m.elm.nr == sphere3_msh.elm.nr + m.elm.get_tags(1005).sum()
        assert np.all(m.elm[m.elm.get_tags(1006)] == sphere3_msh.elm[sphere3_msh.elm.get_tags(1005)])


class TestMsh:
    def test_compact_ordering_elm(self, sphere3_msh):
        m = copy.deepcopy(sphere3_msh)
        m.elm.node_number_list += 10
        m.compact_ordering(m.nodes.node_number + 10)
        assert np.all(m.elm.node_number_list == sphere3_msh.elm.node_number_list)

    def test_crop_mesh(self, sphere3_msh):
        cropped_mesh = sphere3_msh.crop_mesh([1003, 5])
        nr_elements_tags = np.sum(cropped_mesh.elm.get_tags(1003) | cropped_mesh.elm.get_tags(5))
        assert nr_elements_tags == cropped_mesh.elm.nr

    def test_crop_mesh_type(self, sphere3_msh):
        cropped_mesh = sphere3_msh.crop_mesh(elm_type=ElementTypes.TETRAHEDRON)
        assert np.all(cropped_mesh.elm.get_tetrahedra())

    def test_crop_mesh_nodes(self, sphere3_msh):
        target = range(1, 11)
        w_node = np.any(np.isin(sphere3_msh.elm.node_number_list, target).reshape(-1, 4), axis=1)
        neighbours = np.unique(sphere3_msh.elm.node_number_list[w_node].T)[1:]
        orig_coords = sphere3_msh.nodes[neighbours]
        cropped_mesh = sphere3_msh.crop_mesh(nodes=range(1, 11))
        assert np.allclose(orig_coords, cropped_mesh.nodes.node_coord)

    def test_crop_mesh_elements(self, sphere3_msh):
        target = [5, 104, 405, 1000]
        bar = sphere3_msh.elements_baricenters()[target]
        cropped_mesh = sphere3_msh.crop_mesh(elements=target)
        assert np.allclose(bar, cropped_mesh.elements_baricenters().value)
        assert np.allclose(sphere3_msh.elm.tag1[np.array(target) - 1], cropped_mesh.elm.tag1)

    def test_join_mesh(self, sphere3_msh):
        m1 = sphere3_msh.crop_mesh([3, 1003])
        v1 = m1.elements_volumes_and_areas().value.sum()
        m2 = sphere3_msh.crop_mesh([5, 1005])
        v2 = m2.elements_volumes_and_areas().value.sum()
        m = m1.join_mesh(m2)
        v = m.elements_volumes_and_areas().value.sum()
        assert m.elm.nr == m1.elm.nr + m2.elm.nr
        assert m.nodes.nr == m1.nodes.nr + m2.nodes.nr
        assert np.isclose(v, v1 + v2)

    def test_elements_baricenters(self, sphere3_msh):
        baricenters = sphere3_msh.elements_baricenters()

        b = np.zeros((sphere3_msh.elm.nr, 3), dtype=float)
        th_indexes = sphere3_msh.elm.get_tetrahedra(return_indices=True)
        b[th_indexes] = np.average(
            sphere3_msh.nodes.node_coord[sphere3_msh.elm.node_number_list[th_indexes, :4] - 1],
            axis=1,
        )

        tr_indexes = sphere3_msh.elm.get_triangles(return_indices=True)
        b[tr_indexes] = np.average(
            sphere3_msh.nodes.node_coord[sphere3_msh.elm.node_number_list[tr_indexes, :3] - 1],
            axis=1,
        )

        np.testing.assert_almost_equal(baricenters.value, b)

    def test_areas(self, sphere3_msh):
        v = sphere3_msh.elements_volumes_and_areas()
        tot_area = np.sum(v[sphere3_msh.elm.triangles])

        np.testing.assert_allclose(tot_area, 4 * np.pi * (95**2 + 90**2 + 85**2), rtol=1e-1)

    def test_volume(self, sphere3_msh):
        v = sphere3_msh.elements_volumes_and_areas()
        tot_vol = np.sum(v[sphere3_msh.elm.tetrahedra])

        # high tolerance due to low resolution
        np.testing.assert_allclose(tot_vol, 4.0 / 3.0 * np.pi * 95**3, rtol=2e-1)

    def test_triangle_angles(self):
        msh = Msh()
        msh.elm = Elements(triangles=np.array([[1, 2, 3], [1, 2, 3], [1, 4, 2], [1, 4, 2]], dtype=int))
        msh.nodes = Nodes(
            np.array(
                [
                    [0, 0, 0],
                    [1, 0, 0],
                    [0, np.tan(np.pi / 6), 0],
                    [0, 0, np.tan(np.pi / 6)],
                ],
                dtype=float,
            )
        )
        angles = msh.triangle_angles()
        assert np.allclose(angles[:3], [90, 30, 60])
        assert np.allclose(angles[3:], [90, 60, 30])

    def test_curvature(self, sphere3_msh):
        # https://computergraphics.stackexchange.com/a/1721
        surf_1005 = np.unique(sphere3_msh.elm[sphere3_msh.elm.get_tags(1005), :3])
        surf_1004 = np.unique(sphere3_msh.elm[sphere3_msh.elm.get_tags(1004), :3])
        surf_1003 = np.unique(sphere3_msh.elm[sphere3_msh.elm.get_tags(1003), :3])
        curv = sphere3_msh.gaussian_curvature()
        assert np.allclose(curv[surf_1005 - 1], (1 / 95**2), atol=1e-4)
        assert np.allclose(curv[surf_1004 - 1], (1 / 90**2), atol=1e-4)
        assert np.allclose(curv[surf_1003 - 1], (1 / 85**2), atol=1e-4)

        assert np.average(curv[surf_1005 - 1]) < np.average(curv[surf_1004 - 1])
        assert np.average(curv[surf_1004 - 1]) < np.average(curv[surf_1003 - 1])

    def test_node_area(self, sphere3_msh):
        msh = sphere3_msh.crop_mesh(1005)
        tot_area = np.sum(msh.nodes_areas())
        # high tolerance due to low resolution
        np.testing.assert_allclose(tot_area, 4.0 * np.pi * 95**2, rtol=2e-1)

    def tests_node_normal(self, sphere3_msh):
        normals = sphere3_msh.nodes_normals()
        outer_nodes = np.unique(sphere3_msh.elm.node_number_list[sphere3_msh.elm.get_tags(1005), :3])
        # high tolerance due to low resolution
        np.testing.assert_allclose(
            normals[outer_nodes - 1],
            sphere3_msh.nodes[outer_nodes] / 95,
            atol=1e-1,
            rtol=1e-1,
        )

        normals = sphere3_msh.nodes_normals(smooth=3)
        # high tolerance due to low resolution
        np.testing.assert_allclose(
            normals[outer_nodes - 1],
            sphere3_msh.nodes[outer_nodes] / 95,
            atol=1e-1,
            rtol=1e-1,
        )

    def tests_node_area2(self, sphere3_msh):
        areas = sphere3_msh.nodes_areas()
        outer_nodes = np.unique(sphere3_msh.elm.node_number_list[sphere3_msh.elm.get_tags(1005), :3])
        # high tolerance due to low resolution
        assert np.isclose(np.sum(areas[outer_nodes]), np.pi * 4.0 * 95**2, rtol=1)

    def test_fix_node_ordering_unchanged(self, sphere3_msh):
        m = copy.deepcopy(sphere3_msh)
        m.fix_th_node_ordering()
        assert np.all(m.elm.node_number_list == sphere3_msh.elm.node_number_list)

    def test_fix_node_ordering_changed(self, sphere3_msh):
        m = copy.deepcopy(sphere3_msh)
        m.elm.node_number_list[:, 1] = sphere3_msh.elm.node_number_list[:, 2]
        m.elm.node_number_list[:, 2] = sphere3_msh.elm.node_number_list[:, 1]
        m.fix_th_node_ordering()
        th = m.nodes[m.elm.node_number_list[m.elm.get_tetrahedra(), :]]
        M = th[:, 1:] - th[:, 0, None]
        assert np.all(np.linalg.det(M) > 0)

    def test_fix_tr_node_ordering_unchanged(self, sphere3_msh):
        m = copy.deepcopy(sphere3_msh)
        m.fix_tr_node_ordering()
        assert np.all(m.elm.node_number_list == sphere3_msh.elm.node_number_list)

    def test_fix_tr_node_ordering_changed(self, sphere3_msh):
        m = copy.deepcopy(sphere3_msh)
        tmp = np.copy(m.elm.node_number_list[:, 1])
        m.elm.node_number_list[:, 1] = m.elm.node_number_list[:, 2]
        m.elm.node_number_list[:, 2] = tmp
        m.fix_tr_node_ordering()
        triangles = m.elm.get_triangles()
        normal_sphere = m.elements_baricenters().value[triangles]
        normal_sphere /= np.linalg.norm(normal_sphere, axis=1)[:, None]
        dotp = np.sum(m.triangle_normals().value[triangles] * normal_sphere, axis=1)
        assert np.all(dotp > 0.9)

    def test_find_corresponding_tetrahedra(self, sphere3_msh):
        corrensponding = sphere3_msh.find_corresponding_tetrahedra()
        bar = sphere3_msh.elements_baricenters()
        direction = ElementData(bar.value / np.linalg.norm(bar.value, axis=1)[:, None])
        tr_data = direction[sphere3_msh.elm.triangles]
        direction[sphere3_msh.elm.triangles] = 0
        assert np.allclose(direction[corrensponding], tr_data, rtol=1e-1, atol=1e-1)

        corrensponding = sphere3_msh.find_corresponding_tetrahedra()
        m = sphere3_msh.crop_mesh([3, 4, 5, 1005])
        corrensponding = m.find_corresponding_tetrahedra()
        bar = m.elements_baricenters()
        direction = ElementData(bar.value / np.linalg.norm(bar.value, axis=1)[:, None])
        tr_data = direction[m.elm.triangles]
        direction[m.elm.triangles] = 0
        assert np.allclose(direction[corrensponding], tr_data, rtol=1e-1, atol=1e-1)

    def test_intersect_segment(self, sphere3_msh):
        m = sphere3_msh.crop_mesh(1005)
        near = np.array([[120, 19, 32], [89, 142, 65]])
        far = -near
        idx, pos = m.intersect_segment(near, far)
        proj2surf = near / np.linalg.norm(near, axis=1)[:, None] * 95
        assert np.all(np.linalg.norm(pos[[0, 2], :] + proj2surf, axis=1) < 0.3)
        assert np.all(np.linalg.norm(pos[[1, 3], :] - proj2surf, axis=1) < 0.3)
        assert np.all(np.linalg.norm(m.elements_baricenters()[idx[:, 1]] - pos, axis=1) < 3)

    def test_intersect_segment_no_hit(self, sphere3_msh):
        m = sphere3_msh.crop_mesh(1005)
        near = np.array([[120, 19, 32], [89, 142, 65]])
        far = 0.99 * near
        idx, pos = m.intersect_segment(near, far)
        assert len(idx) == 0
        assert len(pos) == 0

    def test_intersect_segment_getfarpoint(self):
        vertices = np.array(
            [
                [-1, -3, -5],
                [2, -3, -5],
                [2, 4, -5],
                [-1, 4, -5],
                [-1, -3, 6],
                [2, -3, 6],
                [2, 4, 6],
                [-1, 4, 6],
            ]
        )
        hull = ConvexHull(vertices)
        elements = hull.simplices + 1
        S = Msh(Nodes(vertices), Elements(triangles=elements))

        points = np.random.uniform(-10, 10, size=(100000, 3))
        directions = np.random.uniform(-1, 1, size=(100000, 3))

        idx, endpoints = S._intersect_segment_getfarpoint(points, directions)

        # The far points lie on the bounding box used internally.
        eps = 0.01  # internal eps in _intersect_segment_getfarpoint to create ROI
        ROI = np.array([[-1 - eps, 2 + eps], [-3 - eps, 4 + eps], [-5 - eps, 6 + eps]])

        eps = 100 * np.finfo(float).eps
        if len(idx) > 0:
            nhits = (
                (np.abs(endpoints[:, 0] - ROI[0, 0]) < eps).astype(int)
                + (np.abs(endpoints[:, 0] - ROI[0, 1]) < eps).astype(int)
                + (np.abs(endpoints[:, 1] - ROI[1, 0]) < eps).astype(int)
                + (np.abs(endpoints[:, 1] - ROI[1, 1]) < eps).astype(int)
                + (np.abs(endpoints[:, 2] - ROI[2, 0]) < eps).astype(int)
                + (np.abs(endpoints[:, 2] - ROI[2, 1]) < eps).astype(int)
            )
            assert np.all(nhits == 1)

    def test_intersect_ray(self):
        vertices = np.array(
            [
                [-1, -3, -5],
                [2, -3, -5],
                [2, 4, -5],
                [-1, 4, -5],
                [-1, -3, 6],
                [2, -3, 6],
                [2, 4, 6],
                [-1, 4, 6],
            ]
        )
        hull = ConvexHull(vertices)
        elements = hull.simplices + 1
        S = Msh(Nodes(vertices), Elements(triangles=elements))

        # Orient the triangles outward.
        normals = S.triangle_normals()[:]
        baricenters = S.elements_baricenters()[:]
        baricenters -= np.mean(baricenters, axis=0)
        dotp = np.einsum("ij, ij -> i", normals, baricenters)
        switch = dotp < 0
        buffer = S.elm.node_number_list[switch, 1]
        S.elm.node_number_list[switch, 1] = S.elm.node_number_list[switch, 2]
        S.elm.node_number_list[switch, 2] = buffer

        points = np.random.uniform(-10, 10, size=(100000, 3))
        directions = np.random.uniform(-1, 1, size=(100000, 3))

        idx, inters_pos = S.intersect_ray(points, directions)

        ROI = np.array([[-1, 2], [-3, 4], [-5, 6]])
        eps = 100 * np.finfo(float).eps
        if len(idx) > 0:
            nhits = (
                (np.abs(inters_pos[:, 0] - ROI[0, 0]) < eps).astype(int)
                + (np.abs(inters_pos[:, 0] - ROI[0, 1]) < eps).astype(int)
                + (np.abs(inters_pos[:, 1] - ROI[1, 0]) < eps).astype(int)
                + (np.abs(inters_pos[:, 1] - ROI[1, 1]) < eps).astype(int)
                + (np.abs(inters_pos[:, 2] - ROI[2, 0]) < eps).astype(int)
                + (np.abs(inters_pos[:, 2] - ROI[2, 1]) < eps).astype(int)
            )
            assert np.all(nhits == 1)

    def test_partition_skin_surface(self, sphere3_msh):
        mesh = copy.deepcopy(sphere3_msh)
        nn = mesh.nodes_normals()

        triangles = mesh.elm.get_tags(1005, return_element_numbers=True)
        elm_idx = triangles[0]
        v = mesh.elm[elm_idx][0]

        v_tris = triangles[(mesh.elm.node_number_list[triangles - 1, :3] == v).sum(1) > 0]
        v_to_move = np.unique(mesh.elm.node_number_list[v_tris - 1, :3])

        f = np.full(v_to_move.shape, 30)

        mesh.nodes.node_coord[v_to_move - 1] -= nn[v - 1] * f[:, None]

        v_in_1, _v_out_1, f_in_1, f_out_1 = mesh.partition_skin_surface()
        # v is considered outside even though it is surrounded by inside nodes
        v_in, _v_out, f_in, f_out = mesh.partition_skin_surface(assume_single_outside_component=False)
        assert v in v_in_1
        assert np.isin(v_to_move, v_in_1).all(), "All updated nodes should be considered 'inside'."
        assert v not in v_in
        # Currently, f_in/f_out is the same irrespective of the call
        assert np.all(f_in == f_in_1)
        assert np.all(f_out == f_out_1)
        assert np.all(np.isin(mesh.elm[f_in_1][:, :3], v_to_move).sum(1) >= 2)

    def test_reconstruct_unique_surface(self, sphere3_msh):
        m2 = sphere3_msh.crop_mesh(elm_type=ElementTypes.TRIANGLE)
        m = sphere3_msh.crop_mesh(elm_type=ElementTypes.TETRAHEDRON)
        m.reconstruct_unique_surface()
        m = m.crop_mesh(elm_type=ElementTypes.TRIANGLE)
        m_tags, m_cts = np.unique(m.elm.tag1, return_counts=True)
        m2_tags, m2_cts = np.unique(m2.elm.tag1, return_counts=True)
        assert m.elm.nr == m2.elm.nr
        assert np.all(m2_tags == m_tags)
        assert np.all(m2_cts == m_cts)

    def test_compute_normalized_gamma(self):
        # The same regular tetrahedron with correct and inverted node ordering.
        v = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
        m = Msh(v, {"tetrahedra": np.array([[1, 2, 3, 4], [1, 2, 4, 3]])})
        assert np.allclose(m.compute_normalized_gamma(), [1.0, np.nan], rtol=1e-3, equal_nan=True)

    def test_smooth_without_th(self, sphere3_msh):
        mesh = sphere3_msh.crop_mesh(elm_type=2)
        tr_nodes = np.unique(mesh.elm[mesh.elm.get_tags(1003), :3])
        mesh.nodes.node_coord[tr_nodes - 1] += np.random.standard_normal((len(tr_nodes), 3)) * 1
        curvature_before = mesh.gaussian_curvature()[tr_nodes]
        mesh.smooth_surfaces(10)
        curvature_after = mesh.gaussian_curvature()[tr_nodes]
        assert np.std(curvature_after) < np.std(curvature_before)

    def test_smooth_with_tetrahedra(self, sphere3_msh):
        mesh = copy.deepcopy(sphere3_msh)
        tr_nodes = np.unique(mesh.elm[mesh.elm.get_tags(1003), :3])
        mesh.nodes.node_coord[tr_nodes - 1] += np.random.standard_normal((len(tr_nodes), 3)) * 1
        curvature_before = mesh.gaussian_curvature()[tr_nodes]
        mesh.smooth_surfaces(10, max_gamma=100)
        curvature_after = mesh.gaussian_curvature()[tr_nodes]
        assert np.std(curvature_after) < np.std(curvature_before)

    def test_smooth_with_tag(self, sphere3_msh):
        mesh = sphere3_msh.crop_mesh(elm_type=2)
        tr_nodes = np.unique(mesh.elm[mesh.elm.get_tags(1003), :3])
        mesh.nodes.node_coord[tr_nodes - 1] += np.random.standard_normal((len(tr_nodes), 3)) * 1
        curvature_before = mesh.gaussian_curvature()
        mask_nodes = np.zeros(mesh.nodes.nr, dtype=bool)
        mask_nodes[tr_nodes - 1] = True
        mesh.smooth_surfaces(10, tags=1003, max_gamma=100)
        curvature_after = mesh.gaussian_curvature()
        assert np.std(curvature_after[mask_nodes]) < np.std(curvature_before[mask_nodes])
        assert np.allclose(curvature_after[~mask_nodes], curvature_before[~mask_nodes])

    def test_smooth_quality(self, sphere3_msh):
        mesh = copy.deepcopy(sphere3_msh)
        tr_nodes = np.unique(mesh.elm[mesh.elm.get_tags(1003), :3])
        mesh.nodes.node_coord[tr_nodes - 1] += np.random.standard_normal((len(tr_nodes), 3)) * 0.1
        mesh_before = copy.deepcopy(mesh)
        mesh.smooth_surfaces(10, max_gamma=3)
        high_q = mesh_before.compute_normalized_gamma() <= 3
        assert np.all(mesh.compute_normalized_gamma()[high_q] <= 3)

    def test_split_tets_along_line(self, sphere3_msh):
        m = sphere3_msh.crop_mesh(elm_type=4)
        n_nodes_pre = m.nodes.nr
        n_tets_pre = m.elm.nr
        idx_tet1, _idx_tet2 = m.split_tets_along_line(1246, 2337, return_tetindices=True)
        assert m.nodes.nr == n_nodes_pre + 1
        assert m.elm.nr == n_tets_pre + len(idx_tet1)

    @pytest.mark.parametrize(
        "points",
        [
            ("inside", "all"),
            ("outside", "all"),
            ("inside", "single"),
            ("outside", "single"),
        ],
    )
    @pytest.mark.parametrize("surface_tags", [None, ElementTags.CSF_TH_SURFACE, ElementTags.SCALP_TH_SURFACE])
    @pytest.mark.parametrize(
        "distance",
        [0.0, 10.0, [10.0, 10.0, 10.0, 10.0, 10.0, 10.0]],
    )
    def test_project_points_on_surface(self, sphere3_msh, points, surface_tags, distance):
        # r phi theta
        pts_sph = np.array(
            (
                (80, 0, 0),
                (80, 0, 45),
                (80, 0, 90),
                (80, 45, 0),
                (80, 45, 45),
                (80, 45, 90),
            )
        )
        if points[0] == "outside":
            pts_sph[:, 0] = 100.0

        # Radius of the surface the points should project to.
        if surface_tags is None:
            if points[0] == "inside":
                diameter = 85.0
            elif points[0] == "outside":
                diameter = 95.0
        else:
            diameter = np.linalg.norm(
                sphere3_msh.nodes[sphere3_msh.elm.node_number_list[sphere3_msh.elm.get_tags(surface_tags), :3]],
                axis=2,
            ).mean()

        if isinstance(distance, list):
            if points[1] == "single":
                distance = distance[0]
            else:
                distance = np.array(distance)

        diameter = diameter + distance

        pts_sph_expected = copy.deepcopy(pts_sph)
        pts_sph_expected[:, 0] = diameter

        pts_cart = toCartesian(pts_sph)
        pts_cart_expected = toCartesian(pts_sph_expected)

        match points[1]:
            case "all":
                pass
            case "single":
                pts_cart = pts_cart[0]
                pts_cart_expected = pts_cart_expected[0]

        pts_prj = sphere3_msh.project_points_on_surface(pts_cart, surface_tags, distance)

        # The coarse sphere limits the angular accuracy of the projection.
        scale = np.abs(pts_sph_expected[0, 0] - pts_sph[0, 0])
        if scale > 20:
            TOL_ANGLE = 1.0
        elif scale > 10:
            TOL_ANGLE = 0.75
        else:
            TOL_ANGLE = 0.5
        TOL_NORM = 0.5

        assert np.all(get_angle(pts_prj, np.atleast_2d(pts_cart_expected)) < TOL_ANGLE)
        assert np.all(np.abs(np.linalg.norm(pts_prj, axis=1) - diameter) < TOL_NORM)


class TestElmData:
    def test_interpolate_grid_const_nn(self, sphere3_msh):
        data = sphere3_msh.elm.tag1
        f = ElementData(data, mesh=sphere3_msh)
        n = (200, 10, 1)
        affine = np.array([[1, 0, 0, -100.5], [0, 1, 0, -5], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
        interp = f.interpolate_to_grid(n, affine)
        assert np.isclose(interp[100, 5, 0], 3)
        assert np.isclose(interp[187, 5, 0], 4)
        assert np.isclose(interp[193, 5, 0], 5)
        assert np.isclose(interp[198, 5, 0], 0)

    def test_interpolate_grid_vec_nn(self, sphere3_msh):
        data = np.vstack([sphere3_msh.elm.tag1, sphere3_msh.elm.tag1 + 10, sphere3_msh.elm.tag1 + 20]).T
        f = ElementData(data, mesh=sphere3_msh)
        n = (200, 10, 1)
        affine = np.array([[1, 0, 0, -100], [0, 1, 0, -5], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
        interp = f.interpolate_to_grid(n, affine)
        assert np.allclose(interp[100, 5, 0, :], [3, 13, 23])
        assert np.allclose(interp[187, 5, 0, :], [4, 14, 24])
        assert np.allclose(interp[193, 5, 0, :], [5, 15, 25])
        assert np.allclose(interp[198, 5, 0, :], [0, 0, 0])

    def test_interpolate_grid_size_nn(self, sphere3_msh):
        data = sphere3_msh.elm.tag1
        f = ElementData(data, mesh=sphere3_msh)
        n = (100, 5, 1)
        affine = np.array([[2, 0, 0, -100], [0, 2, 0, -5], [0, 0, 2, 0], [0, 0, 0, 1]], dtype=float)
        interp = f.interpolate_to_grid(n, affine)
        assert np.isclose(interp[50, 2, 0], 3)
        assert np.isclose(interp[93, 2, 0], 4)
        assert np.isclose(interp[96, 2, 0], 5)
        assert np.isclose(interp[98, 2, 0], 0)

    def test_interpolate_grid_rotate_nn(self, sphere3_msh):
        data = np.zeros(sphere3_msh.elm.nr)
        b = sphere3_msh.elements_baricenters().value
        f = ElementData(data, mesh=sphere3_msh)
        f.value[(b[:, 0] > 0) * (b[:, 1] > 0)] = 1.0
        f.value[(b[:, 0] < 0) * (b[:, 1] > 0)] = 2.0
        f.value[(b[:, 0] < 0) * (b[:, 1] < 0)] = 3.0
        f.value[(b[:, 0] > 0) * (b[:, 1] < 0)] = 4.0
        n = (200, 200, 1)
        affine = np.array(
            [
                [np.cos(np.pi / 4.0), np.sin(np.pi / 4.0), 0, -141],
                [-np.sin(np.pi / 4.0), np.cos(np.pi / 4.0), 0, 0],
                [0, 0, 1, 0.5],
                [0, 0, 0, 1],
            ],
            dtype=float,
        )
        interp = f.interpolate_to_grid(n, affine)
        assert np.isclose(interp[190, 100, 0], 4)
        assert np.isclose(interp[100, 190, 0], 1)
        assert np.isclose(interp[10, 100, 0], 2)
        assert np.isclose(interp[100, 10, 0], 3)

    def test_to_affine_grid(self, sphere3_msh, tmp_path):
        import nibabel

        data = sphere3_msh.elm.tag1
        f = ElementData(data, mesh=sphere3_msh)
        affine = np.array([[1, 0, 0, -100.5], [0, 1, 0, -5], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
        img = nibabel.Nifti1Pair(np.zeros((200, 10, 1)), affine)
        fn = str(tmp_path / "image.nii")
        nibabel.save(img, fn)
        affine = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
        fn2 = str(tmp_path / "affine.mat")
        np.savetxt(fn2, affine)
        interp = f.to_deformed_grid(fn2, fn)
        interp = np.asanyarray(interp.dataobj)
        assert np.isclose(interp[100, 5, 0], 3)
        assert np.isclose(interp[187, 5, 0], 4)
        assert np.isclose(interp[193, 5, 0], 5)
        assert np.isclose(interp[198, 5, 0], 0)

    def test_to_nonlinear_grid(self, sphere3_msh, tmp_path):
        import nibabel

        data = sphere3_msh.elm.tag1
        f = ElementData(data, mesh=sphere3_msh)
        affine = np.array([[1, 0, 0, -100.5], [0, 1, 0, -5], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
        x, y, z = np.meshgrid(np.arange(-100, 100), np.arange(-5, 5), np.arange(0, 1), indexing="ij")
        nonl_transform = np.concatenate((x[..., None], y[..., None], z[..., None]), axis=3).astype(float)
        img = nibabel.Nifti1Pair(nonl_transform, affine)
        fn = str(tmp_path / "image.nii")
        nibabel.save(img, fn)
        interp = f.to_deformed_grid(fn, fn, fix_boundary_zeros=False)
        interp = np.asanyarray(interp.dataobj)
        assert np.isclose(interp[100, 5, 0], 3)
        assert np.isclose(interp[187, 5, 0], 4)
        assert np.isclose(interp[193, 5, 0], 5)
        assert np.isclose(interp[198, 5, 0], 0)

    def test_from_data_grid(self, sphere3_msh):
        X, _, _ = np.meshgrid(
            np.arange(-100, 100),
            np.arange(-100, 100),
            np.arange(-100, 100),
            indexing="ij",
        )
        X = X.astype(float)
        affine = np.array(
            [[1, 0, 0, -100], [0, 1, 0, -100], [0, 0, 1, -100], [0, 0, 0, 1]],
            dtype=float,
        )
        ed = ElementData.from_data_grid(sphere3_msh, X, affine)
        assert np.allclose(sphere3_msh.elements_baricenters().value[:, 0], ed.value)

    def test_from_data_grid_vec(self, sphere3_msh):
        X, Y, Z = np.meshgrid(
            np.arange(-100, 100),
            np.arange(-100, 100),
            np.arange(-100, 100),
            indexing="ij",
        )
        V = np.stack([X, Y, Z], axis=3)
        V = V.astype(float)
        affine = np.array(
            [[1, 0, 0, -100], [0, 1, 0, -100], [0, 0, 1, -100], [0, 0, 0, 1]],
            dtype=float,
        )
        ed = ElementData.from_data_grid(sphere3_msh, V, affine)
        assert np.allclose(sphere3_msh.elements_baricenters().value, ed.value)

    def test_from_data_grid_slice(self, sphere3_msh):
        X, Y, Z = np.meshgrid(np.arange(-100, 100), np.arange(-100, 100), [0], indexing="ij")
        V = np.stack([X, Y, Z], axis=3)
        V = V.astype(float)
        affine = np.array([[1, 0, 0, -100], [0, 1, 0, -100], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
        ed = ElementData.from_data_grid(sphere3_msh, V, affine, cval=np.nan, order=1)
        bar = sphere3_msh.elements_baricenters().value
        in_slice = (bar[:, 2] >= -0.5) * (bar[:, 2] <= 0.5)
        assert np.allclose(bar[in_slice, :2], ed.value[in_slice, :2])
        assert np.all(np.isnan(ed.value[~in_slice]))


class TestSurfaceIO:
    def test_gifti_io(self, sphere3_msh):
        surf = sphere3_msh.crop_mesh(1005)
        write_gifti_surface(surf, "tst.gii")
        surf_read = read_gifti_surface("tst.gii")
        os.remove("tst.gii")
        assert np.allclose(surf.nodes[surf.elm[:, :3]], surf_read.nodes[surf_read.elm[:, :3]])


class TestHashing:
    def test_collisions(self):
        np.random.seed(0)
        array = np.random.randint(0, 1_000_000, size=(10_000_000, 3), dtype=np.int32)
        array = np.unique(array, axis=0)
        hash_ = _hash_rows(np.array(array))
        _, count = np.unique(hash_, return_counts=True)
        assert np.all(count == 1)
