"""
Gmsh-style mesh structures
Derived from the SimNIBS package.
Please check on www.simnibs.org how to cite our work in publications.

Copyright (C) 2013-2020 Andre Antunes, Guilherme B Saturnino, Kristoffer H Madsen

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

import copy
import warnings
from functools import partial

import cortech
import nibabel
import numpy as np
import numpy.typing as npt
import scipy.ndimage
import scipy.sparse
import scipy.sparse.csgraph
import scipy.spatial

from charm.mesh import _kernels, cgal
from charm.mesh.element_tags import ElementTags, ElementTypes
from charm.utils import nifti
from charm.utils.transformations import nifti_transform


class InvalidMeshError(ValueError):
    pass


class Nodes:
    """Mesh nodes.

    Parameters
    -----------------------
    node_coord (optional): (Nx3) ndarray
        Coordinates of the nodes

    Attributes
    ----------------------
    node_coord: (Nx3) ndarray
        Coordinates of the nodes

    nr: property
        Number of nodes

    Examples
    -----------------------------------
     >>> nodes = Nodes(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
     >>> nodes.node_number
     array([1, 2, 3], dtype=int32)
     >>> nodes[1]
     array([1, 0, 0])
     >>> nodes[np.array([True, False, True])]
     array([[1, 0, 0], [0, 0, 1]])
    """

    def __init__(self, node_coord=None):
        self.node_coord = np.array([], dtype="float64")
        if node_coord is not None:
            self.node_coord = node_coord

    @property
    def nr(self):
        """Number of nodes"""
        return self.node_coord.shape[0]

    @property
    def node_number(self):
        """Node numbers (1, ..., nr)"""
        return np.arange(1, self.nr + 1, dtype="int32")

    def __getitem__(self, index):
        return _getitem_one_indexed(self.node_coord, index)


class Elements:
    """Mesh elements.

    Parameters
    --------------------------
    triangles (optional): (Nx3) ndarray
        Nodes of each triangle
    tetrahedra (optional): (Nx4) ndarray
        Nodes of each tetrahedron
    points (optional): (N,) ndarray
        Node of each point element
    lines (optional): (Nx2) ndarray
        Nodes of each line

    Attributes
    ----------------------------------
    elm_number: (N,) ndarray
        Element numbers (1, ..., nr)
    elm_type: (N,) ndarray
        Element type (2=triangle, 4=tetrahedron, etc)
    tag1: (N,) ndarray
        First tag of each element
    tag2: (N,) ndarray
        Second tag of each element
    node_number_list: (Nx4) ndarray
        Nodes of each element, padded with -1 for elements with fewer than 4 nodes
    nr: int
        Number of elements

    Notes
    -------------------------
    Node and element numbers start at 1.
    """

    def __init__(self, triangles=None, tetrahedra=None, points=None, lines=None):
        self.elm_type = np.zeros(0, "int8")
        self.tag1 = np.zeros(0, dtype="int16")
        self.tag2 = np.zeros(0, dtype="int16")
        self.node_number_list = np.zeros((0, 4), dtype="int32")

        if points is not None:
            assert len(points.shape) == 1
            assert np.all(points > 0), "Node count should start at 1"
            self.node_number_list = np.zeros((points.shape[0], 4), dtype="int32")
            self.node_number_list[:, 0] = points.astype("int32")
            self.node_number_list[:, 1:] = -1
            self.elm_type = np.full((self.nr,), ElementTypes.POINT, dtype=np.int32)

        if lines is not None:
            assert lines.shape[1] == 2
            assert np.all(lines > 0), "Node count should start at 1"
            self.node_number_list = np.zeros((lines.shape[0], 4), dtype="int32")
            self.node_number_list[:, :2] = lines.astype("int32")
            self.node_number_list[:, 2:] = -1
            self.elm_type = np.full((self.nr,), ElementTypes.LINE, dtype=np.int32)

        if triangles is not None:
            assert triangles.shape[1] == 3
            assert np.all(triangles > 0), "Node count should start at 1"
            self.node_number_list = np.zeros((triangles.shape[0], 4), dtype="int32")
            self.node_number_list[:, :3] = triangles.astype("int32")
            self.node_number_list[:, 3] = -1
            self.elm_type = np.full((self.nr,), ElementTypes.TRIANGLE, dtype=np.int32)

        if tetrahedra is not None:
            assert tetrahedra.shape[1] == 4
            assert np.all(tetrahedra > 0), "Node count should start at 1"
            if len(self.node_number_list) == 0:
                self.node_number_list = tetrahedra.astype("int32")
                self.elm_type = np.full((self.nr,), ElementTypes.TETRAHEDRON, dtype=np.int32)
            else:
                self.node_number_list = np.vstack((self.node_number_list, tetrahedra.astype("int32")))
                self.elm_type = np.append(
                    self.elm_type,
                    np.full(len(tetrahedra), ElementTypes.TETRAHEDRON, dtype=np.int32),
                )

        if len(self.node_number_list) > 0:
            self.tag1 = np.ones((self.nr,), dtype="int32")
            self.tag2 = np.ones((self.nr,), dtype="int32")

    @property
    def nr(self):
        """Number of elements"""
        return self.node_number_list.shape[0]

    @property
    def triangles(self):
        """Triangle element numbers"""
        return self.elm_number[self.elm_type == ElementTypes.TRIANGLE]

    @property
    def tetrahedra(self):
        """Tetrahedra element numbers"""
        return self.elm_number[self.elm_type == ElementTypes.TETRAHEDRON]

    def get_tags(
        self,
        tags: int | list[int] | tuple | np.ndarray,
        invert: bool = False,
        return_element_numbers: bool = False,
        return_indices: bool = False,
    ):
        """Get the elements with the given tags, as a mask unless return_element_numbers
        or return_indices is set. Element numbers are 1-based, indices are 0-based.

        Parameters
        ----------
        tags : int | list | tuple | np.ndarray
            Tags to look up in `tag1`.
        invert : bool
            If true, return the elements *not* in `tags`.
        return_element_numbers : bool, optional
            If true, return the element numbers (1-based!) of the desired
            elements (default = False).
        return_indices : bool, optional
            If true, return the indices (0-based!) of the desired elements
            (default = False).

        Returns
        -------
        elements
            Indices or mask of elements.
        """
        assert not (return_element_numbers and return_indices), (
            "Can only return *either* element numbers (one-based) *or* indices (zero-based)"
        )
        m = np.isin(self.tag1, tags, invert=invert)
        if return_element_numbers:
            m = self.elm_number[m]
        elif return_indices:
            m = np.where(m)[0]
        return m

    def get_type(
        self,
        element_type: str,
        tags: int | list[int] | tuple | np.ndarray | None = None,
        invert_tags: bool = False,
        return_element_numbers: bool = False,
        return_indices: bool = False,
    ):
        """Get the elements of a type, as a mask unless return_element_numbers or
        return_indices is set. Element numbers are 1-based, indices are 0-based.

        Parameters
        ----------
        element_type : str
            Name of a member of charm.mesh.element_tags.ElementTypes, e.g.,
            TRIANGLE or TETRAHEDRON.
        tags : int | list | tuple | np.ndarray | None, optional
            Return only elements with these tags (default = None, i.e., all
            elements of the type).
        invert_tags : bool
            If true, return the elements *not* in `tags`.
        return_element_numbers : bool, optional
            If true, return the element numbers (1-based!) of the desired
            elements (default = False).
        return_indices : bool, optional
            If true, return the indices (0-based!) of the desired elements
            (default = False).

        Returns
        -------
        elements
            Mask, element numbers or indices of the elements.
        """
        assert not (return_element_numbers and return_indices), (
            "Can only return *either* element numbers (one-based) *or* indices (zero-based)"
        )
        m = self.elm_type == getattr(ElementTypes, element_type)
        m = m & self.get_tags(tags, invert=invert_tags) if tags is not None else m
        if return_element_numbers:
            m = self.elm_number[m]
        elif return_indices:
            m = np.where(m)[0]
        return m

    def get_triangles(self, *args, **kwargs):
        return self.get_type("TRIANGLE", *args, **kwargs)

    def get_tetrahedra(self, *args, **kwargs):
        return self.get_type("TETRAHEDRON", *args, **kwargs)

    @property
    def elm_number(self):
        """Element numbers (1, ..., nr)"""
        return np.arange(1, self.nr + 1, dtype="int32")

    def get_faces(self, tetrahedra_indexes=None):
        """The nodes of each face and the faces of each tetrahedron

        Parameters
        ----------------
        tetrahedra_indexes: np.ndarray
            Tetrahedra to get the faces of (default: all tetrahedra)

        Returns
        -------------
        faces: np.ndarray
            List of nodes in faces, in arbitrary order
        th_faces: np.ndarray
            List of faces in each tetrahedra, starts at 0, order=((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3))
        face_adjacency_list: np.ndarray
            List of tetrahedron adjacent to each face, filled with -1 if a face is in a
            single tetrahedron. Not in the normal element ordering, but only in the order
            the tetrahedra are presented
        """
        if tetrahedra_indexes is None:
            tetrahedra_indexes = self.tetrahedra
        th = self[tetrahedra_indexes]
        faces = th[:, [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]].reshape(-1, 3)

        # A face is in at most two tetrahedra, which a stable sort of the faces puts next to each other
        order, is_first = _sort_rows(np.sort(faces, axis=1))
        first = np.flatnonzero(is_first)
        count = np.diff(np.append(first, len(faces)))
        if np.any(count > 2):
            raise InvalidMeshError("Found a face with more than 2 adjacent tetrahedra!")
        inv = np.empty(len(faces), dtype=np.int64)
        inv[order] = np.cumsum(is_first) - 1
        idx = order[first]

        tets = order // 4
        face_adjacency_list = np.full((len(first), 2), -1, dtype=np.int64)
        face_adjacency_list[:, 0] = tets[first]
        is_shared = count == 2
        face_adjacency_list[is_shared, 1] = tets[first[is_shared] + 1]
        face_adjacency_list[is_shared] = np.sort(face_adjacency_list[is_shared], axis=1)
        return faces[idx], inv.reshape(-1, 4), face_adjacency_list

    def _get_tet_faces_and_adjacent_tets(self):
        """reconstructs the triangle faces of the tetrahedra for further use
            by Msh.reconstruct_unique_surface() and during mesh despiking

        Returns
        ---------
            faces: N_faces x 3 array
                node indices of all tet faces
            idx_tet_faces: N_tet x 4 array
                indices of the tet faces (indices into the faces array)
            adj_th: N_tet x 4 array
                indices of tet neighbors (-1 marks no adjacent; i.e, "air")

        Notes
        ------
            * All returned indices are 0-based !!
            * The mesh must contain only tetrahedra
        """
        assert np.all(self.get_tetrahedra())

        faces, th_faces, adjacency_list = self.get_faces()
        faces = np.sort(faces, axis=1) - 1

        adj_th = -np.ones((self.nr, 4), dtype=int)
        th_indexing = np.tile(np.arange(self.nr), (2, 1)).T
        for i in range(4):
            adj = adjacency_list[th_faces[:, i]]
            adj_th[:, i] = adj[adj != th_indexing]

        return faces, th_faces.astype(int), adj_th

    def connected_components(self, element_indexes=None):
        """Finds connected components

        Parameters
        ----------------
        element_indexes: np.ndarray (optional)
            1-based element numbers or a mask of the elements to search. Default: all elements

        Returns
        -------------
        connected_comp: list of np.ndarray
            List of arrays with the indices of the elements of each connected component
        """
        if element_indexes is None:
            elm = self[:]
            elm_type = self.elm_type
            elm_number = self.elm_number
        else:
            elm = self[element_indexes]
            elm_type = _getitem_one_indexed(self.elm_type, element_indexes)
            elm_number = _getitem_one_indexed(self.elm_number, element_indexes)

        # Relabel the nodes to be contiguous, so they index a graph of only the used nodes
        _, fixed_elm = np.unique(elm, return_inverse=True)
        fixed_elm = fixed_elm.reshape(-1, 4)
        if -1 in elm:
            fixed_elm -= 1
            fixed_elm[elm == -1] = -1
        elm = fixed_elm
        # All nodes of an element are in the same component, so linking each to the first is enough
        n_nodes = np.max(elm) + 1
        tr = elm_type == ElementTypes.TRIANGLE
        th = elm_type == ElementTypes.TETRAHEDRON
        first = np.concatenate((np.repeat(elm[tr, 0], 2), np.repeat(elm[th, 0], 3)))
        others = np.concatenate((elm[tr, 1:3].ravel(), elm[th, 1:].ravel()))
        M = scipy.sparse.coo_matrix((np.ones(first.size, dtype=bool), (first, others)), shape=(n_nodes, n_nodes))
        n_comp, labels = scipy.sparse.csgraph.connected_components(M, directed=False)
        in_volume = tr | th
        element_labels = labels[elm[in_volume, 0]]
        order = np.argsort(element_labels, kind="stable")
        counts = np.bincount(element_labels, minlength=n_comp)
        return np.split(elm_number[in_volume][order], np.cumsum(counts)[:-1])

    def add_triangles(self, triangles, tag):
        """Add triangles before the other elements, in place

        Parameters
        -----------
        triangles: (Nx3) array
            Nodes of the triangles
        tag: int or (N,) array
            Tag of the triangles
        """
        if triangles.shape[1] != 3:
            raise ValueError("triangles should be a Nx3 array")
        n_add = triangles.shape[0]
        try:
            n_tag = len(tag)
        except TypeError:
            tag = int(tag) * np.ones(n_add, dtype=int)
        else:
            if n_tag != n_add:
                raise ValueError("Number of tags show be the same as the number of triangles")
            tag = np.reshape(tag, n_add)
            tag = tag.astype(int)

        elm_type = 2 * np.ones(n_add, dtype=int)
        node_number_list = np.hstack((triangles, -np.ones((n_add, 1), dtype=int)))
        self.node_number_list = np.vstack((node_number_list, self.node_number_list))
        self.elm_type = np.hstack((elm_type, self.elm_type))
        self.tag1 = np.hstack((tag, self.tag1))
        self.tag2 = np.hstack((tag, self.tag2))

    def __getitem__(self, index):
        return _getitem_one_indexed(self.node_number_list, index)


def make_surface_mesh(vertices, faces, tag: int | None = None):
    """A triangle mesh of the given one-based faces, all tagged with tag if given."""
    m = Msh(Nodes(vertices), Elements(faces))
    if tag is not None:
        m.elm.tag1[:] = tag
        m.elm.tag2[:] = tag
    return m


class Msh:
    def __init__(
        self,
        nodes: Nodes | npt.NDArray | None = None,
        elements: Elements | dict[str, npt.NDArray] | None = None,
    ):
        """A mesh of points, lines, triangles and tetrahedra.

        Parameters
        ----------
        nodes: Nodes | npt.NDArray | None
            Node coordinates.
        elements: Elements | dict[str, npt.NDArray] | None
            An Elements, or the keyword arguments of one (e.g.,
            dict(triangles=..., tetrahedra=...)).

        Attributes
        ----------
        nodes: charm.mesh.msh.Nodes
        elm: charm.mesh.msh.Elements
        fn: str
            Name of the file the mesh was read from or written to
        """
        match nodes:
            case None:
                nodes = Nodes()
            case Nodes():
                pass
            case np.ndarray():
                nodes = Nodes(nodes)
            case _:
                raise ValueError()
        self.nodes = nodes

        match elements:
            case None:
                elements = Elements()
            case Elements():
                pass
            case dict():
                elements = Elements(**elements)
            case _:
                raise ValueError()
        self.elm = elements

        self.fn = ""

    def crop_mesh(self, tags=None, elm_type=None, nodes=None, elements=None):
        """A new mesh with only the selected elements and the nodes they use, renumbered

        Parameters
        ---------------------
        tags: (optional) int or list
            Tags of the elements to keep
        elm_type: (optional) list of int
            Types of the elements to keep (2 for triangles, 4 for tetrahedra)
        nodes: (optional) list of ints
            Keep the elements with at least one of these nodes
        elements: (optional) list of ints
            Numbers of the elements to keep

        Returns
        ---------------------
        charm.mesh.msh.Msh

        Raises
        -----------------------
        ValueError
            If no selection is given or nothing is selected

        Notes
        -----------
        An element is kept if any of tags, elm_type, nodes or elements selects it.
        """
        if tags is None and elm_type is None and nodes is None and elements is None:
            raise ValueError("At least one type of crop must be specified")

        elm_keep = np.zeros((self.elm.nr,), dtype=bool)

        if tags is not None:
            elm_keep += np.isin(self.elm.tag1, tags)

        if elm_type is not None:
            elm_keep += np.isin(self.elm.elm_type, elm_type)

        if nodes is not None:
            elm_keep += np.any(np.isin(self.elm.node_number_list, nodes).reshape(-1, 4), axis=1)

        if elements is not None:
            elm_keep += np.isin(self.elm.elm_number, elements)

        if not np.any(elm_keep):
            raise ValueError("Could not find any element to crop!")

        idx = np.where(elm_keep)[0]
        kept_nodes = self.elm.node_number_list[idx]
        is_used = np.zeros(self.nodes.nr + 1, dtype=bool)
        is_used[kept_nodes[kept_nodes > 0]] = True
        unique_nodes = np.flatnonzero(is_used)
        nr_unique = np.size(unique_nodes)

        nodes_dict = np.zeros(self.nodes.nr + 1, dtype="int")
        nodes_dict[unique_nodes] = np.arange(1, 1 + nr_unique, dtype="int32")

        node_number_list = nodes_dict[kept_nodes]
        node_number_list[kept_nodes == -1] = -1

        node_coord = self.nodes.node_coord[unique_nodes - 1]

        cropped = Msh()

        cropped.elm.tag1 = np.copy(self.elm.tag1[idx])
        cropped.elm.tag2 = np.copy(self.elm.tag2[idx])
        cropped.elm.elm_type = np.copy(self.elm.elm_type[idx])
        cropped.elm.node_number_list = node_number_list

        cropped.nodes.node_coord = np.copy(node_coord)

        return cropped

    def join_mesh(self, other):
        """Join the current mesh with another

        Parameters
        -----------
        other: charm.mesh.msh.Msh
            Mesh to be joined

        Returns
        --------
        joined: charm.mesh.msh.Msh
            Mesh with joined nodes and elements
        """
        if self.nodes.nr == 0:
            return copy.deepcopy(other)
        elif other.nodes.nr == 0:
            return copy.deepcopy(self)

        joined = copy.deepcopy(self)
        other = copy.deepcopy(other)

        joined.nodes.node_coord = np.vstack([joined.nodes.node_coord, other.nodes.node_coord])
        other_node_number_list = other.elm.node_number_list + self.nodes.nr
        joined.elm.node_number_list = np.vstack([joined.elm.node_number_list, other_node_number_list])
        joined.elm.tag1 = np.hstack([joined.elm.tag1, other.elm.tag1])
        joined.elm.tag2 = np.hstack([joined.elm.tag2, other.elm.tag2])
        joined.elm.elm_type = np.hstack([joined.elm.elm_type, other.elm.elm_type])
        lines = joined.elm.get_type("LINE", return_indices=True)
        points = joined.elm.get_type("POINT", return_indices=True)
        triangles = joined.elm.get_triangles(return_indices=True)
        tetrahedra = joined.elm.get_tetrahedra(return_indices=True)
        new_elm_order = np.hstack([points, lines, triangles, tetrahedra])
        joined.elm.node_number_list = joined.elm.node_number_list[new_elm_order]
        joined.elm.tag1 = joined.elm.tag1[new_elm_order]
        joined.elm.tag2 = joined.elm.tag2[new_elm_order]
        joined.elm.elm_type = joined.elm.elm_type[new_elm_order]
        joined.elm.node_number_list[joined.elm.get_triangles(), 3] = -1
        joined.elm.node_number_list[joined.elm.get_type("LINE"), 2:] = -1
        joined.elm.node_number_list[joined.elm.get_type("POINT"), 1:] = -1

        return joined

    def elements_baricenters(self):
        """Calculates the baricenter of the elements

        Returns
        ------------
        baricenters: ElementData
        """
        bar = ElementData(np.zeros((self.elm.nr, 3), dtype=float), "baricenter", mesh=self)
        bar.mesh = self
        th_indexes = self.elm.tetrahedra
        tr_indexes = self.elm.triangles

        if len(th_indexes) > 0:
            bar[th_indexes] = np.average(self.nodes[self.elm[th_indexes]], axis=1)

        if len(tr_indexes) > 0:
            bar[tr_indexes] = np.average(self.nodes[self.elm[tr_indexes][:, :3]], axis=1)

        return bar

    def elements_volumes_and_areas(self, return_signed_volume: bool = False):
        """Calculates the volumes of tetrahedra and areas of triangles, in the mesh's unit (normally mm)

        Parameters
        ----------
        return_signed_volume: bool
            If true, return the signed volume of tetrahedra.

        Returns
        ----------
        v: ElementData
            Volume/areas of tetrahedra/triangles
        """
        vol = ElementData(np.zeros(self.elm.nr, dtype=float), "volumes_and_areas")

        tr_indexes = self.elm.triangles
        node_tr = self.nodes[self.elm[tr_indexes, :3]]
        sideA = node_tr[:, 1] - node_tr[:, 0]
        sideB = node_tr[:, 2] - node_tr[:, 0]
        n = np.cross(sideA, sideB)
        vol[tr_indexes] = np.linalg.norm(n, axis=1) * 0.5

        th_indexes = self.elm.tetrahedra
        node_th = self.nodes[self.elm[th_indexes]]
        signed_vol = _compute_signed_tetrahedra_volume(node_th)
        vol[th_indexes] = signed_vol if return_signed_volume else np.abs(signed_vol)

        return vol

    def triangle_angles(self):
        """Calculates the angles of the mesh triangles

        Returns
        --------
        min_triangle_angles: ElementData
            The three angles of each triangle in degrees, nan at other elements
        """
        tr_indexes = self.elm.triangles
        node_tr = self.nodes[self.elm[tr_indexes, :3]]
        a = np.linalg.norm(node_tr[:, 1] - node_tr[:, 0], axis=1)
        b = np.linalg.norm(node_tr[:, 2] - node_tr[:, 0], axis=1)
        c = np.linalg.norm(node_tr[:, 2] - node_tr[:, 1], axis=1)
        cos_angles = np.vstack(
            [
                (a**2 + b**2 - c**2) / (2 * a * b),
                (a**2 + c**2 - b**2) / (2 * a * c),
                (b**2 + c**2 - a**2) / (2 * b * c),
            ]
        ).T
        angles = np.rad2deg(np.arccos(cos_angles))

        angle_field = ElementData(np.nan * np.zeros((self.elm.nr, 3)), "min_triangle_angles")
        angle_field[tr_indexes] = angles
        return angle_field

    def triangle_normals(self, smooth=False):
        """Calculates the normals of triangles

        Parameters
        ------------
        smooth (optional): int
            Number of smoothing steps to perform. Default: 0

        Returns
        --------
        normals: ElementData
            normals of triangles, zero at the tetrahedra
        """
        normals = ElementData(np.zeros((self.elm.nr, 3), dtype=float), "normals")
        tr_indexes = self.elm.triangles
        node_tr = self.nodes[self.elm[tr_indexes, :3]]
        sideA = node_tr[:, 1] - node_tr[:, 0]
        sideB = node_tr[:, 2] - node_tr[:, 0]
        n = np.cross(sideA, sideB)
        normals[tr_indexes] = n / np.linalg.norm(n, axis=1)[:, None]
        if smooth > 0:
            normals_nodes = self.nodes_normals(smooth=smooth)
            tr = self.elm[tr_indexes, :3]
            n = np.mean(normals_nodes[tr - 1], axis=1)
            normals[tr_indexes] = n / np.linalg.norm(n, axis=1)[:, None]
        elif smooth < 0:
            raise ValueError("smooth parameter must be >= 0")
        return normals

    def nodes_areas(self):
        """Areas for all nodes in a surface

        Returns
        ---------
        nd: ndarray
            Area of each node
        """
        areas = self.elements_volumes_and_areas()[self.elm.triangles]
        triangle_nodes = self.elm[self.elm.triangles, :3] - 1
        nd = np.bincount(triangle_nodes.reshape(-1), np.repeat(areas / 3.0, 3), self.nodes.nr)

        return nd

    def nodes_normals(self, triangles=None, smooth=0):
        """Normals for all nodes in a surface

        Parameters
        ------------
        triangles: list of ints
            List of triangles to be taken into consideration for calculating the normals

        smooth: int (optional)
            Number of smoothing cycles to perform. Default: 0

        Returns
        ---------
        nd: ndarray
            Normal of each surface node
        """
        if triangles is None:
            elements = self.elm.triangles - 1
        else:
            elements = triangles - 1
        triangle_nodes = self.elm.node_number_list[elements, :3] - 1

        node_tr = self.nodes.node_coord[triangle_nodes]
        sideA = node_tr[:, 1] - node_tr[:, 0]
        sideB = node_tr[:, 2] - node_tr[:, 0]
        normals = np.cross(sideA, sideB)

        nd = np.zeros((self.nodes.nr, 3))
        for s in range(smooth + 1):
            for i in range(3):
                nd[:, i] = np.bincount(
                    triangle_nodes.reshape(-1),
                    np.repeat(normals[:, i], 3),
                    self.nodes.nr,
                )

            normals = np.sum(nd[triangle_nodes], axis=1)
            normals /= np.linalg.norm(normals, axis=1)[:, None]

        nodes = np.unique(triangle_nodes)
        nd[nodes] = nd[nodes] / np.linalg.norm(nd[nodes], axis=1)[:, None]

        return nd

    def gaussian_curvature(self):
        """Calculates the Gaussian curvature at each node

        Returns
        ---------
        nd: ndarray
            Gaussian curvature of each surface node

        References
        -------------
        https://computergraphics.stackexchange.com/a/1721
        """
        nodes_areas = self.nodes_areas()
        triangles_angles = np.deg2rad(np.nan_to_num(self.triangle_angles()[self.elm.triangles]))
        triangle_nodes = self.elm[self.elm.triangles] - 1
        triangle_nodes = triangle_nodes[:, :3]
        node_angles = np.bincount(triangle_nodes.reshape(-1), triangles_angles.reshape(-1), self.nodes.nr)
        idx = nodes_areas == 0
        nodes_areas[idx] = 0.1
        gaussian_curvature = (2 * np.pi - node_angles) / nodes_areas
        gaussian_curvature[idx] = 0
        return gaussian_curvature

    def fix_th_node_ordering(self):
        """Fixes the node ordering of tetrahedra in-place"""
        th = self.nodes[self.elm.node_number_list[self.elm.get_tetrahedra(), :]]
        switch = self.elm.get_tetrahedra()
        switch[switch] = _compute_signed_tetrahedra_volume(th) < 0
        self.elm.node_number_list[switch, :2] = self.elm.node_number_list[switch][:, [1, 0]]

    def fix_tr_node_ordering(self):
        """Fixes the node ordering of the triangles in-place"""
        corresponding = self.find_corresponding_tetrahedra()
        triangles = self.elm.get_triangles(return_indices=True)

        triangles = triangles[corresponding != -1]
        corresponding = corresponding[corresponding != -1]

        normals = self.triangle_normals().value[triangles]
        baricenters = self.elements_baricenters().value
        pos_bar = baricenters[corresponding - 1] - baricenters[triangles]

        dotp = np.einsum("ij, ij -> i", normals, pos_bar)
        switch = triangles[dotp > 0]
        self.elm.node_number_list[switch, :2] = self.elm.node_number_list[switch][:, [1, 0]]

    def compact_ordering(self, node_number):
        """Renumber the nodes of the elements from node_number to 1, ..., nr_nodes

        Parameters
        --------------
        node_number: (N_nodes,) ndarray of int
            Node numbers in the original mesh
        """
        rel = -9999 * np.ones((np.max(node_number) + 1), dtype=int)
        rel[node_number] = np.arange(1, self.nodes.nr + 1, dtype=int)
        is_padding = self.elm.node_number_list == -1
        self.elm.node_number_list = rel[self.elm.node_number_list]
        self.elm.node_number_list[is_padding] = -1
        self.elm.node_number_list[self.elm.get_triangles(), 3] = -1

    def find_corresponding_tetrahedra(self):
        """Finds the tetrahedra corresponding to each triangle

        Returns
        ---------
        corresponding_th_indices: ndarray of ints
           Element number (1-based) of the tetrahedron of each triangle, -1 if there is none
        """
        tr_tags = np.unique(self.elm.tag1[self.elm.get_triangles()])
        corresponding_th_indices = -np.ones(len(self.elm.triangles), dtype=int)
        for t in tr_tags:
            to_crop = [t]
            to_crop.append(t - ElementTags.TH_SURFACE_START)

            th_of_interest = self.elm.get_tetrahedra(to_crop, return_indices=True)
            if len(th_of_interest) == 0:
                continue
            tr_of_interest = self.elm.get_triangles(t, return_indices=True)

            th = self.elm.node_number_list[th_of_interest]
            faces = th[:, [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]].reshape(-1, 3)
            faces = faces.reshape(-1, 3)
            faces_hash_array = _hash_rows(faces)

            tr = self.elm.node_number_list[tr_of_interest, :3]
            tr_hash_array = _hash_rows(tr)
            faces_argsort = faces_hash_array.argsort()
            faces_hash_array = faces_hash_array[faces_argsort]
            tr_search = np.searchsorted(faces_hash_array, tr_hash_array)
            tr_search[tr_search == len(faces_hash_array)] = 0
            has_tetra = faces_hash_array[tr_search] == tr_hash_array
            index = faces_argsort[tr_search[has_tetra]] // 4

            position = np.searchsorted(self.elm.triangles, self.elm.elm_number[tr_of_interest[has_tetra]])
            corresponding_th_indices[position] = self.elm.elm_number[th_of_interest[index]]

        if np.any(corresponding_th_indices == -1):
            # Match the rest to outer boundary faces of any tag, leaving out "air" tetrahedra (tag1 = -1)
            idx_th_all = self.elm.get_tetrahedra(-1, invert_tags=True, return_indices=True)
            th = self.elm.node_number_list[idx_th_all]
            faces = th[:, [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]]
            faces = faces.reshape(-1, 3)
            faces_hash_array = _hash_rows(faces)
            # Faces in a single tetrahedron are on the outer boundary
            [faces_hash_array, idx_fc, counts] = np.unique(faces_hash_array, return_index=True, return_counts=True)
            faces_hash_array = faces_hash_array[counts == 1]
            idx_fc = idx_fc[counts == 1]

            tr_of_interest = self.elm.triangles[corresponding_th_indices == -1] - 1
            tr = self.elm.node_number_list[tr_of_interest, :3]
            tr_hash_array = _hash_rows(tr)
            _, idx_tr, idx_th = np.intersect1d(tr_hash_array, faces_hash_array, return_indices=True)
            idx_tr = tr_of_interest[idx_tr]
            idx_th = idx_th_all[idx_fc[idx_th] // 4] + 1
            corresponding_th_indices[idx_tr] = idx_th

        return corresponding_th_indices

    def intersect_segment(self, near, far):
        """Finds the triangle (if any) that intersects a line segment

        Parameters
        ------------
        near: (N, 3) array
            Start of the line segment
        far: (N, 3) array
            end of the line segment

        Returns
        --------
        indices: (M, 2) array
            Pairs of indices with the line segment index and the triangle index
        intercpt_pos (M, 3) array:
            Positions where the interceptions occur
        """
        if near.ndim == 1:
            near = near[None, :]
        if far.ndim == 1:
            far = far[None, :]
        if not (near.shape[1] == 3 and far.shape[1] == 3):
            raise ValueError("near and far poins should be arrays of size (N, 3)")

        indices, points = cgal.segment_triangle_intersection(
            self.nodes[:], self.elm[self.elm.get_triangles(), :3] - 1, near, far
        )
        if len(indices) > 0:
            indices[:, 1] = self.elm.triangles[indices[:, 1]]

        return indices, points

    def intersect_ray(self, points, directions, AABBTree=None):
        """Finds the triangle (if any) that intersects with the rays starting
        at points and pointing into directions

        Parameters
        ------------
        points: (N, 3) array
            start points
        directions: (N, 3) array
            direction vectors
        AABBTree: (optional)
            Precalculated tree with an intersection(start, end) method

        Returns
        --------
        indices: (M, 2) array
            Pairs of indices with the points index and the triangle index
            NOTE: points indices are 0-based, triangle indices are 1-based!
        intercpt_pos (M, 3) array:
            Positions where the interceptions occur
        """
        if points.ndim == 1:
            points = points[None, :]
        if directions.ndim == 1:
            directions = directions[None, :]
        if not (points.shape[1] == 3 and directions.shape[1] == 3):
            raise ValueError("start points and directions should be arrays of size (N, 3)")

        idx, far = self._intersect_segment_getfarpoint(points, directions)

        if len(idx) > 0:
            if AABBTree is None:
                indices, intercpt_pos = cgal.segment_triangle_intersection(
                    self.nodes[:],
                    self.elm[self.elm.get_triangles(), :3] - 1,
                    points[idx, :],
                    far,
                )
            else:
                indices, intercpt_pos = AABBTree.intersection(points[idx, :], far)

            if len(indices) > 0:
                indices[:, 1] = self.elm.triangles[indices[:, 1]]
                indices[:, 0] = idx[indices[:, 0]]

        else:
            indices = []
            intercpt_pos = []

        return indices, intercpt_pos

    def _intersect_segment_getfarpoint(self, points, directions):
        """The indices of the rays that intersect the bounding box of the mesh,
        and where they leave it.

        Parameters
        ----------
        points : array_like Nx3
            start points
        directions : array_like Nx3
            direction vectors.

        Returns
        -------
        idx_vec : (N,) array
            indices of intersecting rays
        end_points : (N, 3) array
            end points of the intersecting rays
        """
        eps = 0.01
        ROI = [
            [
                np.min(self.nodes.node_coord[:, 0]) - eps,
                np.max(self.nodes.node_coord[:, 0]) + eps,
            ],
            [
                np.min(self.nodes.node_coord[:, 1]) - eps,
                np.max(self.nodes.node_coord[:, 1]) + eps,
            ],
            [
                np.min(self.nodes.node_coord[:, 2]) - eps,
                np.max(self.nodes.node_coord[:, 2]) + eps,
            ],
        ]

        has_far = np.zeros(points.shape[0], dtype=bool)
        scale = np.zeros(points.shape[0])

        plane_idx = [[1, 2], [0, 2], [0, 1]]
        for k in range(3):
            idx = np.logical_and(directions[:, k] < 0, points[:, k] > ROI[k][0])
            s = (ROI[k][0] - points[idx, k]) / directions[idx, k]
            p = points[idx, :] + s.reshape(-1, 1) * directions[idx, :]

            inside_rect = (
                (p[:, plane_idx[k][0]] >= ROI[plane_idx[k][0]][0])
                * (p[:, plane_idx[k][0]] <= ROI[plane_idx[k][0]][1])
                * (p[:, plane_idx[k][1]] >= ROI[plane_idx[k][1]][0])
                * (p[:, plane_idx[k][1]] <= ROI[plane_idx[k][1]][1])
            )
            idx[idx] = inside_rect

            has_far[idx] = True
            scale[idx] = s[inside_rect]

            idx = np.logical_and(directions[:, k] > 0, points[:, k] < ROI[k][1])
            s = (ROI[k][1] - points[idx, k]) / directions[idx, k]
            p = points[idx, :] + s.reshape(-1, 1) * directions[idx, :]

            inside_rect = (
                (p[:, plane_idx[k][0]] >= ROI[plane_idx[k][0]][0])
                * (p[:, plane_idx[k][0]] <= ROI[plane_idx[k][0]][1])
                * (p[:, plane_idx[k][1]] >= ROI[plane_idx[k][1]][0])
                * (p[:, plane_idx[k][1]] <= ROI[plane_idx[k][1]][1])
            )

            idx[idx] = inside_rect
            has_far[idx] = True
            scale[idx] = s[inside_rect]

        far = points[has_far] + scale[has_far].reshape(-1, 1) * directions[has_far]

        return np.where(has_far)[0], far

    def partition_skin_surface(
        self,
        label_skin=ElementTags.SCALP_TH_SURFACE,
        assume_single_outside_component=True,
        tol: float = 1e-3,
    ):
        """Return the vertices and faces estimated to be on the inner skin surface
        (nasal cavities, ear canals, etc.) and the outer one (scalp, nose, etc.).

        A vertex is outside if the ray along its normal does not intersect the
        mesh. This is not perfect but does a reasonable job. A face is outside
        if two or more of its vertices are.

        Parameters
        ----------
        label_skin : int, optional, default: ElementTags.SCALP_TH_SURFACE
            Label of skin surface (e.g. 1005)
        assume_single_outside_component: bool
            If true, keep only the largest connected component of outside faces
            and relabel the rest as inside. This removes small spurious outside
            clusters, e.g., in the ear canal.
        tol : float
            Tolerance for avoiding self-intersections.

        Returns
        -------
        v_in : ndarray
            Indices (one-based!) of inner skin vertices.
        v_out : ndarray
            Indices (one-based!) of outer skin vertices.
        f_in : ndarray
            Indices (one-based!) of inner skin faces.
        f_out : ndarray
            Indices (one-based!) of outer skin faces.
        """
        assert tol > 0

        f_orig_id = self.elm.elm_number[self.elm.get_tags(label_skin)]
        faces = self.elm[f_orig_id, :3]
        verts = np.unique(faces)
        m = Msh(Nodes(self.nodes.node_coord), Elements(faces))

        is_subset = len(verts) < m.nodes.nr

        n = m.nodes_normals()
        n = n[verts - 1] if is_subset else n

        coords = m.nodes.node_coord
        coords = coords[verts - 1] if is_subset else coords

        # Avoid self-intersections by moving each point slightly along the test direction
        idx = np.unique(m.intersect_ray(coords + tol * n, n)[0][:, 0])

        v_in = verts[idx] if is_subset else idx  # idx is zero-based!
        v_out = np.setdiff1d(verts, v_in, assume_unique=True)

        f_out_m = np.isin(faces, v_out).sum(1) > 1

        if assume_single_outside_component:
            cc = m.elm.connected_components(f_out_m)
            cc_sorted_idx = np.argsort(list(map(len, cc)))[::-1]
            # cc is one-based!
            for c in cc_sorted_idx[1:]:
                f_out_m[cc[c] - 1] = False

            v_out = np.intersect1d(faces[f_out_m], v_out)
            v_in = np.setdiff1d(verts, v_out)

        f_in = f_orig_id[~f_out_m]
        f_out = f_orig_id[f_out_m]

        # the indices are into self
        return v_in, v_out, f_in, f_out

    def relabel_internal_air(
        self,
        label_skin=ElementTags.SCALP_TH_SURFACE,
        label_new=ElementTags.INTERNAL_AIR_TH_SURFACE,
    ):
        """Relabel the skin in internal air cavities, which marks the valid skin region for TES optimization.

        Parameters
        ----------
        label_skin : int, optional, default: ElementTags.SCALP_TH_SURFACE
            Label of skin surface (e.g. 1005) with internal air.
        label_new : int, optional, default: ElementTags.INTERNAL_AIR_TH_SURFACE
            Label of internal skin surface.

        Returns
        -------
        m : Msh
            Mesh with relabeled internal air.
        """
        m = copy.copy(self)
        _, _, f_in, _ = self.partition_skin_surface(label_skin)
        m.elm.tag1[f_in - 1] = label_new
        m.elm.tag2[:] = m.elm.tag1
        return m

    def project_points_on_surface(
        self,
        points,
        surface_tags: int | list[int] | tuple | np.ndarray | None = None,
        distance: float | np.ndarray = 0.0,
        return_element_index: bool | np.ndarray = False,
    ):
        """Find the position on the surface closest to each coordinate

        Parameters
        -------
        points: nx3 ndarray
            Points which will be projected on the surface.
        surface_tags: (optional)
            Tag of the target surface. Default: None (positions will be
            projected on closest surface)
        distance: float or nx1 ndarray (optional)
            Distance (normal) to the surface to be enforced. Default: 0
            Note: negative values will move the point inside, positive values
            outside the volume defined by the surface
        return_element_index: bool (optional)
            whether to return the elm index (one-based) of the triangle
            containing the closest surface point

        Returns
        ------
        coords_projected: nx3 ndarray
            coordinates projected on the surface
        elm_index: (n,) ndarray
            Only if return_element_index is set
        """
        points = np.array(points)
        if points.ndim == 1:
            points = points.reshape((1, len(points)))

        tri_mask = self.elm.get_triangles(surface_tags)
        tri_node_list = self.elm.node_number_list[tri_mask, :3] - 1
        tri_nodes = np.unique(tri_node_list)

        vertices = self.nodes.node_coord[tri_nodes]

        old2new = np.zeros(tri_nodes[-1] + 1, dtype=np.int32)
        old2new[tri_nodes] = np.arange(len(tri_nodes))
        faces = old2new[tri_node_list]

        # A real skin surface has pinched vertices, and projecting does not care.
        surface = cortech.Surface(vertices, faces, check_topology=False)
        tris, _, projs, _ = surface.project_points(points)

        if not np.all(np.isclose(distance, 0.0)):
            distance = np.array(distance)
            if distance.ndim == 0:
                distance = distance.reshape(1)

            tri_pts = surface.as_mesh()[tris]

            sideA = tri_pts[:, 1] - tri_pts[:, 0]
            sideB = tri_pts[:, 2] - tri_pts[:, 0]
            n = np.cross(sideA, sideB)
            n /= np.linalg.norm(n, axis=1)[:, None]

            projs += distance[:, None] * n

        if return_element_index:
            return projs, np.flatnonzero(tri_mask)[tris] + 1
        else:
            return projs

    def reconstruct_unique_surface(
        self,
        hierarchy=None,
        add_outer_as=None,
        faces=None,
        idx_tet_faces=None,
        adj_tets=None,
    ):
        """Reconstructs the mesh surfaces from the tetrahedra.

        Triangles are added between tetrahedra of different labels and at
        faces towards "air". The hierarchy decides the label of each triangle.

        Parameters
        ------------
        hierarchy: list of ints or None (optional)
            Hierarchy of triangle labels that determines which label will
            be chosen
            Default: (1001, 1002, 1009, 1003, 1004, 1008, 1007, 1006, 1010, 1005)
            WM (1001) has the highest priority, GM (1002) comes next, ...
        add_outer_as: int (optional)
            add outer boundary to mesh using given index. NOTE: This boundary
            has highest priority, and will be placed "first" (Default: None)
        faces (Default: None), idx_tet_faces (Default: None), adj_tets (Default: None):
            Output of Elements._get_tet_faces_and_adjacent_tets().
            Can be passed here to speed up surface reco.

        Note
        ------
        * The mesh must contain only tetrahedra
        * This function acts in-place
        * Will not fix any element_data that might be associated with this mesh
        * will not add triangles to "air" tetrahedra (having label -1)
        """
        assert np.all(self.elm.get_tetrahedra())

        if hierarchy is None:
            hierarchy = (
                ElementTags.WM,
                ElementTags.GM,
                ElementTags.BLOOD,
                ElementTags.CSF,
                ElementTags.BONE,
                ElementTags.SPONGY_BONE,
                ElementTags.COMPACT_BONE,
                ElementTags.EYE_BALLS,
                ElementTags.MUSCLE,
                ElementTags.SCALP,
            )
        hierarchy = np.asarray(hierarchy)
        if np.any(hierarchy > ElementTags.TH_SURFACE_START):
            hierarchy -= ElementTags.TH_SURFACE_START

        if (add_outer_as is not None) and (add_outer_as > ElementTags.TH_SURFACE_START):
            add_outer_as -= ElementTags.TH_SURFACE_START

        if (faces is None) or (idx_tet_faces is None) or (adj_tets is None):
            faces, idx_tet_faces, adj_tets = self.elm._get_tet_faces_and_adjacent_tets()

        tag = copy.copy(self.elm.tag1)

        adj_labels = tag[adj_tets]
        adj_labels[adj_tets == -1] = -1  # -1 indicates "air"
        adj_diff = adj_labels - tag.reshape((len(tag), 1)) != 0
        adj_diff[tag == -1] = False  # do not add triangles for "air" tetrahedra

        # Renumber the tags by hierarchy so that the smallest one wins on each face
        hierarchy = np.asarray(hierarchy)
        if add_outer_as is not None:
            hierarchy = np.append(-1, hierarchy)
        hierarchy = np.append(hierarchy, np.setdiff1d(np.unique(tag), hierarchy))

        map_old_new = np.zeros(np.max(hierarchy) + 2, dtype=int)
        map_old_new[hierarchy] = np.arange(len(hierarchy)) + 1
        tag = map_old_new[tag]

        idx_tri = idx_tet_faces[adj_diff]
        tag_tri = np.tile(tag, (4, 1)).T
        if add_outer_as is not None:
            tag_tri[adj_labels == -1] = 1  # because map_old_new[-1] == 1
        tag_tri = tag_tri[adj_diff]

        # The lowest tag of each face comes first when sorting on face * n_tags + tag
        n_tags = len(hierarchy) + 2
        keys = np.sort(idx_tri.astype(np.int64) * n_tags + tag_tri)
        idx_tri = keys // n_tags
        is_first = np.hstack((True, idx_tri[1:] != idx_tri[:-1]))
        idx_tri = idx_tri[is_first]

        tag_tri = hierarchy[keys[is_first] % n_tags - 1]
        if add_outer_as is not None:
            tag_tri[tag_tri == -1] = add_outer_as

        self.elm.add_triangles(faces[idx_tri, :] + 1, ElementTags.TH_SURFACE_START + tag_tri)
        self.fix_tr_node_ordering()

    def compute_normalized_gamma(
        self,
        tags: int | list[int] | tuple | np.ndarray | None = None,
    ) -> npt.NDArray:
        """Calculate the (normalized) gamma quality metric of tetrahedra. The
        equilateral tetrahedron has a value of 1. Other tetrahedra are >1.

        Parameters
        ----------
        tags : int | list | tuple | np.ndarray | None
            Tags of the tetrahedra. Default: all tetrahedra.

        Returns
        -------
        gamma : npt.NDArray
            Array of size (N,) containing the gamma metric. (Where the volume
            of tetrahedra is negative, this value is nan.)

        References
        ----------
        Parthasarathy, V. N., C. M. Graichen, and A. F. Hathaway. "A comparison of
            tetrahedron quality measures" Finite Elements in Analysis and Design
            15.3 (1994): 255-261.
        """
        equilateral_tetrahedron_value = 8.479670

        mask = self.elm.get_tetrahedra(tags)
        gamma = _kernels.tetrahedra_gamma(
            np.ascontiguousarray(self.nodes.node_coord, dtype=np.float64),
            np.ascontiguousarray(self.elm.node_number_list[mask], dtype=np.int32),
        )
        return np.divide(gamma, equilateral_tetrahedron_value, out=gamma)

    def smooth_surfaces(self, n_steps, step_size=0.3, tags=None, max_gamma=5):
        """In-place smoothing of the mesh surfaces using Taubin smoothing,
            ensures that the tetrahedra quality does not fall below
            a minimal level. Can still decrease overall tetrahedra quality, though.

        The mesh must have inner triangles.

        Parameters
        ------------
        n_steps: int
            number of smoothing steps to perform
        step_size (optional): float 0<step_size<1
            Size of each smoothing step. Default: 0.3
        tags: (optional) int or list
            list of tags to be smoothed. Default: all
        max_gamma: (optional) float
            Maximum gamma tetrahedron quality metric. Surface nodes are
            excluded from smoothing when gamma of neighboring tetrahedra
            would exceed max_gamma otherwise. Default: 5
            (for gamma metric see Parthasarathy et al., Finite Elements in
             Analysis and Design, 1994)
        """
        assert step_size > 0 and step_size < 1
        idx = self.elm.get_triangles(tags)
        surf_nodes = np.unique(self.elm.node_number_list[idx, :3]) - 1
        nodes_mask = np.zeros(self.nodes.nr, dtype=bool)
        nodes_mask[surf_nodes] = True

        tr = self.elm[self.elm.triangles, :3] - 1
        rows = tr[:, [0, 0, 1, 1, 2, 2]].ravel()
        columns = tr[:, [1, 2, 0, 2, 0, 1]].ravel()
        adj_tr = scipy.sparse.csr_matrix(
            (np.ones(rows.size, dtype=bool), (rows, columns)), shape=(self.nodes.nr, self.nodes.nr)
        )
        adj_tr.sum_duplicates()
        adj_indices = np.ascontiguousarray(adj_tr.indices, np.uint64)
        adj_indptr = np.ascontiguousarray(adj_tr.indptr, np.uint64)
        surf_nodes = surf_nodes.astype(np.uint64)

        th = self.elm[self.elm.tetrahedra] - 1
        th = th[nodes_mask[th].any(axis=1)]

        def _get_bad_gamma(gamma, max_gamma):
            return np.isnan(gamma) + (gamma > max_gamma)

        nodes_coords = np.ascontiguousarray(self.nodes.node_coord, float)
        smooth_mesh = Msh(nodes_coords, {"tetrahedra": th + 1})
        gamma = smooth_mesh.compute_normalized_gamma()
        n_badgamma = _get_bad_gamma(gamma, max_gamma).sum()
        for i in range(n_steps):
            nc_before = nodes_coords.copy()
            _kernels.gauss_smooth_simple(surf_nodes, nodes_coords, adj_indices, adj_indptr, float(step_size))
            # Taubin's inflating step, which counters the shrinking of the first
            _kernels.gauss_smooth_simple(surf_nodes, nodes_coords, adj_indices, adj_indptr, -1.05 * float(step_size))

            smooth_mesh.nodes.node_coord = nodes_coords
            gamma = smooth_mesh.compute_normalized_gamma()

            # Reverting the bad tetrahedra takes at most 4 iterations. The limit ensures stability.
            idx_badtet = _get_bad_gamma(gamma, max_gamma)
            for _ in range(4):
                if np.sum(idx_badtet) <= n_badgamma:
                    break

                idx_badnodes = np.zeros(self.nodes.nr, dtype=bool)
                idx_badnodes[th[idx_badtet]] = True
                nodes_coords[idx_badnodes] = nc_before[idx_badnodes]

                smooth_mesh.nodes.node_coord = nodes_coords
                gamma = smooth_mesh.compute_normalized_gamma()

                idx_badtet = _get_bad_gamma(gamma, max_gamma)

            n_badgamma = idx_badtet.sum()

        self.nodes.node_coord = smooth_mesh.nodes.node_coord

    def split_tets_along_line(self, idx_n1, idx_n2, do_checks=True, return_tetindices=False):
        """Add a node halfway between the two given nodes and split all
        tetrahedra on the line between them. Works in-place.

        Parameters
        ----------
        idx_n1 : int
            index of first node
        idx_n2 : int
            index of second node
        do_checks : bool, optional
            The mesh must only contain tets and no data. The corresponding
            checks can be disabled to gain a bit of speed. The default is True.
        return_tetindices : bool, optional
            The indices of the new tetrahedra are returned if set to True.
            The default is False.

        Returns
        -------
        tets1, tets2: lists of ints (optional)
            The tetrahedra connected to nodes n1 and n2, when return_tetindices is
            set. The new node is added at the end, so its index equals the number
            of nodes in the mesh. All indices are 1-based.
        """
        split = self.split_tets_along_lines([(idx_n1, idx_n2)], do_checks)[0]
        if return_tetindices:
            return split

    def split_tets_along_lines(self, lines, do_checks=True):
        """Splits along each of the given (idx_n1, idx_n2) in turn, as if by repeated calls
        to split_tets_along_line, returning their tets1, tets2. Works in-place.

        A split only touches the few tets around a line, so they are looked up through the
        tets of each node and the arrays of the mesh are only extended at the end.
        """
        if do_checks and not np.all(self.elm.get_tetrahedra()):
            raise TypeError("The mesh must only contain tetrahedra")

        nr_nodes = self.nodes.nr
        nr_elements = self.elm.nr
        line_nodes = {int(node) for line in lines for node in line}
        ptr, tets = _node_to_elements(self.elm.node_number_list - 1, nr_nodes, np.array(sorted(line_nodes)) - 1)
        tets_of_node = {}

        def node_tets(node):
            if node not in tets_of_node:
                tets_of_node[node] = set(tets[ptr[node - 1] : ptr[node]].tolist())
            return tets_of_node[node]

        element_nodes = {}

        def nodes_of(tet):
            if tet in element_nodes:
                return element_nodes[tet]
            return self.elm.node_number_list[tet].tolist()

        new_coords = []
        all_coords = self.nodes.node_coord

        def coords_of(node):
            return new_coords[node - nr_nodes - 1] if node > nr_nodes else all_coords[node - 1]

        n_added_elements = 0
        new_element_sources = []
        result = []
        for idx_n1, idx_n2 in lines:
            idx_n1, idx_n2 = int(idx_n1), int(idx_n2)
            orgtets = sorted(node_tets(idx_n1) & node_tets(idx_n2))
            if len(orgtets) == 0:
                raise ValueError("The two nodes are not connected!")

            new_coords.append((coords_of(idx_n1) + coords_of(idx_n2)) / 2)
            idx_newnode = nr_nodes + len(new_coords)

            newtets = []
            for tet in orgtets:
                nodes = nodes_of(tet)
                new_tet = nr_elements + n_added_elements
                n_added_elements += 1
                newtets.append(new_tet)
                new_element_sources.append(tet)

                # new tets connect to the new node and node n2, old tets to the new node and node n1
                element_nodes[new_tet] = [idx_newnode if n == idx_n1 else n for n in nodes]
                element_nodes[tet] = [idx_newnode if n == idx_n2 else n for n in nodes]

                for node in nodes:
                    if node != idx_n1 and (node in line_nodes or node > nr_nodes):
                        node_tets(node).add(new_tet)
            tets_of_node[idx_n2].difference_update(orgtets)
            tets_of_node[idx_newnode] = set(orgtets) | set(newtets)
            result.append((np.array(orgtets) + 1, np.array(newtets) + 1))

        if new_coords:
            self.nodes.node_coord = np.vstack((all_coords, new_coords))
        if n_added_elements:
            sources = np.array(new_element_sources)
            # Sources may themselves be new elements, whose tags are only known once all of the splits are through
            original = sources.copy()
            while np.any(original >= nr_elements):
                is_new = original >= nr_elements
                original[is_new] = sources[original[is_new] - nr_elements]
            self.elm.node_number_list = np.vstack(
                (self.elm.node_number_list, np.zeros((n_added_elements, 4), dtype=self.elm.node_number_list.dtype))
            )
            self.elm.tag1 = np.hstack((self.elm.tag1, self.elm.tag1[original]))
            self.elm.tag2 = np.hstack((self.elm.tag2, self.elm.tag2[original]))
            self.elm.elm_type = np.hstack((self.elm.elm_type, np.full(n_added_elements, 4, np.int32)))
        for tet, nodes in element_nodes.items():
            self.elm.node_number_list[tet] = nodes
        return result


class Data:
    """Store data in elements or nodes

    Parameters
    -----------------------
    value: np.ndarray
        Value of field in nodes

    field_name: str (optional)
        name of field. Default: empty

    mesh: charm.mesh.msh.Msh (optional)
        Mesh where the field is define. Required for several methods

    Attributes
    --------------
    value: ndarray
        Value of field in nodes
    field_name: str
        name of field
    nr: property
        number of data points
    nr_comp: property
        number of dimensions per data point (1 for scalars, 3 for vectors)

    """

    def __init__(self, value, name="", mesh=None):
        self.field_name = name
        self.value = value
        self.mesh = mesh

        if value.ndim > 2:
            raise ValueError("Can only handle 1 and 2 dimensional fields Tensors should be given as a Nx9 array")

        if self.nr_comp > self.nr:
            warnings.warn("Second axis larger than the first Field is probably transposed")

    @property
    def nr(self):
        """Number of data entries"""
        return self.value.shape[0]

    @property
    def nr_comp(self):
        """Number of field components"""
        try:
            return self.value.shape[1]
        except IndexError:
            return 1

    def __getitem__(self, index):
        return _getitem_one_indexed(self.value, index)

    def to_deformed_grid(
        self,
        warp,
        reference,
        out=None,
        out_original=None,
        order=1,
        inverse_warp=None,
        reference_original=None,
        binary=False,
        fix_boundary_zeros=True,
    ):
        """Interpolates field to a grid and apply non-linear interpolation

        We first interpolate to a grid and then apply the transformation in order to
        avoid problems from deformed triangles

        Parameters
        ------------
        warp: str
            Name of file with the transformation. Can either be a nifti file
            where each voxel corresponds to coordinates (x, y, z) in the original space
            or an affine transformation defined from the target space to the original
            space. In the later case, the name must finish in ".mat", and it will be read
            with the numpy loadtxt file
        reference: str
            Name of reference file. The output will be in the same space
            as the reference (same affine transfomation and same dimensions)
        out: str (optional)
            If not None, the result will be written to this file as a nifti
        out_original: str (optional)
            If not None, the volume in the original grid be written to this file as a nifti
        order: int (optional)
            Interpolation order to be used for transformation. Default: 1
        inverse_warp: str
            Name of nifti file with inverse the transformation. Used to rotate vectors to the
            target space in the case of non-liner transformations. If the transformation is
            linear, the inverse matrix is used.
        reference_original: str
            Name of nifti file with reference in the original space. Used to determine
            the dimensions and affine transformation for the initial griding
        binary: bool (optional)
            If True, thresholds the image at 0.5 and converts to np.uint8. Default: False
        fix_boundary_zeros: bool (Optional)
            Whether to replace zeros at boundaries of deformation field by values
            of clostest non-zero voxel; relevant only for non-linear transforms
            (Default: True)

        Returns
        --------
        img: nibabel.Nifti1Pair
            Nibabel image object with tranformed field
        """
        self._require_mesh()
        if reference_original is None:
            reference_nifti = nifti.load(reference)
        else:
            reference_nifti = nifti.load(reference_original)
        affine = reference_nifti.affine
        dimensions = reference_nifti.shape[:3]
        image = self.interpolate_to_grid(dimensions, affine)
        if len(image.shape) == 3:
            image = image[..., None]

        if out_original is not None:
            img = nibabel.Nifti1Pair(image, affine)
            img.header.set_xyzt_units("mm")
            img.set_qform(affine)
            if image.dtype == np.float64:
                img.set_data_dtype(np.float32)
            nifti.save(img, out_original)

        img = nifti_transform(
            (image, affine),
            warp,
            reference,
            out=out,
            order=order,
            inverse_warp=inverse_warp,
            binary=binary,
            fix_boundary_zeros=fix_boundary_zeros,
        )

        return img

    def __setitem__(self, index, item):
        index = _fix_indexing_one(index)
        self.value[index] = item

    def interpolate_to_grid(self, n_voxels, affine):
        raise NotImplementedError

    def _require_mesh(self) -> "Msh":
        if self.mesh is None:
            raise ValueError("Cannot evaluate function if .mesh property is not assigned")
        return self.mesh


class ElementData(Data):
    """
    Data (scalar, vector or tensor) defined in mesh elements.

    Parameters
    -----------------------
    value: ndarray
        Value of field in elements. Should have the shape
            - (n_elm,) or (n_elm, 1) for scalar fields
            - (n_elm, 3) for vector fields
            - (n_elm, 9) for tensors

    field_name: str (optional)
        name of field. Default: ''

    mesh: charm.mesh.msh.Msh (optional)
        Mesh structure where the field is defined. Required for many methods


    Attributes
    --------------------------
    value: ndarray
        Value of field in elements
    field_name: str
        name of field
    elm_number: ndarray
        index of elements
    nr: property
        number of data points
    nr_comp: property
        number of dimensions per data point (1 for scalars, 3 for vectors)
    """

    def __init__(self, value, name="", mesh=None):
        Data.__init__(self, value=value, name=name, mesh=mesh)

    def interpolate_to_grid(self, n_voxels, affine):
        """Interpolates the ElementData into a grid. Each voxel takes the value of
        the tetrahedron that contains it.

        Parameters
        -----------
        n_voxels: list or tuple
            number of voxels in x, y, and z directions
        affine: ndarray
            A 4x4 matrix specifying the transformation from voxels to xyz

        Returns
        --------
        image: ndarray
            An (n_voxels[0], n_voxels[1], n_voxels[2], nr_comp) matrix with
            interpolated values. If nr_comp == 1, the last dimension is squeezed out
        """
        msh = self._require_mesh()
        if self.nr != msh.elm.nr:
            raise ValueError("Invalid Mesh! Mesh should have the same number of elementsas the number of data points")
        if len(n_voxels) != 3:
            raise ValueError("n_voxels should have length = 3")
        if affine.shape != (4, 4):
            raise ValueError("Affine should be a 4x4 matrix")
        if len(msh.elm.tetrahedra) == 0:
            raise InvalidMeshError("Mesh has no volume elements")

        msh_th = msh.crop_mesh(elm_type=4)
        v = np.atleast_2d(self.value)
        if v.shape[0] < v.shape[1]:
            v = v.T
        v = v[msh.elm.get_tetrahedra()]

        nd = np.hstack([msh_th.nodes.node_coord, np.ones((msh_th.nodes.nr, 1))])
        inv_affine = np.linalg.inv(affine)
        nd = inv_affine.dot(nd.T).T[:, :3]

        image = np.zeros([n_voxels[0], n_voxels[1], n_voxels[2], self.nr_comp], dtype=float)
        field = v.astype(float)
        image = _kernels.interp_grid(
            np.array(n_voxels, dtype=int),
            field,
            nd.astype(float),
            (msh_th.elm.node_number_list - 1).astype(int),
        )
        image = image.astype(self.value.dtype)
        if self.nr_comp == 1:
            image = np.squeeze(image, axis=3)
        return image

    @classmethod
    def from_data_grid(cls, mesh, data_grid, affine, field_name="", **kwargs):
        """Defines an ElementData field from a mesh and gridded data

        Parameters
        ---------
        mesh: Msh()
            Mesh structure where the field is to be interpolated
        data_grid: ndarray
            Array of 3 or 4 dimensions with data
        affine: 4x4 ndarray
            Array describing the affine transformation from the data grid to the mesh
            space
        kwargs: see the scipy.ndimage.map_coordinates documentation
        """
        assert len(data_grid.shape) in [
            3,
            4,
        ], "The data grid must have 3 or 4 dimensions"
        coords = element_grid_coordinates(mesh, data_grid.shape, affine)

        f = partial(
            scipy.ndimage.map_coordinates,
            coordinates=coords,
            output=data_grid.dtype,
            **kwargs,
        )
        if len(data_grid.shape) == 4:
            indim = data_grid.shape[3]
            outdata = np.array([f(data_grid[..., i]) for i in range(indim)]).T
        elif len(data_grid.shape) == 3:
            outdata = f(data_grid)

        ed = cls(outdata, name=field_name, mesh=mesh)
        return ed


def element_grid_coordinates(mesh, shape, affine):
    """The coordinates (3, n_elements) of the baricenters of the elements in a grid of the
    given shape, for interpolating with scipy.ndimage.map_coordinates."""
    bar = mesh.elements_baricenters().value.T
    iM = np.linalg.inv(affine)
    coords = iM[:3, :3].dot(bar) + iM[:3, 3, None]
    # Points within half a voxel outside the grid take the edge voxel
    for i in range(3):
        s = shape[i]
        coords[i, (coords[i, :] > -0.5) * (coords[i, :] < 0)] = 0.0
        coords[i, (coords[i, :] > s - 1) * (coords[i, :] < s - 0.5)] = s - 1
    return coords


def _node_to_elements(node_list, nr_nodes, nodes=None):
    """For an element-by-node array of zero-based node numbers, the elements of each
    node in ascending order: those of node n are elements[ptr[n] : ptr[n + 1]].
    Restricted to the given nodes if any, as sorting all of them takes seconds."""
    flat = node_list.ravel()
    positions = None
    if nodes is not None:
        is_wanted = np.zeros(nr_nodes, dtype=bool)
        is_wanted[nodes] = True
        positions = np.flatnonzero(is_wanted[flat])
        flat = flat[positions]
    order = np.argsort(flat, kind="stable")
    ptr = np.zeros(nr_nodes + 1, dtype=np.int64)
    ptr[1:] = np.cumsum(np.bincount(flat, minlength=nr_nodes))
    if positions is not None:
        order = positions[order]
    return ptr, order // node_list.shape[1]


def _sort_rows(rows):
    """The stable lexicographic order of a (n, 3) array of positive integers, and
    whether each row in that order differs from the one before. Rows that fit a
    single integer sort much faster."""
    base = int(rows.max()) + 1
    if base**3 < 2**63:
        keys = (rows[:, 0].astype(np.int64) * base + rows[:, 1]) * base + rows[:, 2]
        order = np.argsort(keys, kind="stable")
        keys = keys[order]
        return order, np.concatenate(([True], keys[1:] != keys[:-1]))
    order = np.lexsort(rows.T[::-1])
    rows = rows[order]
    return order, np.concatenate(([True], np.any(rows[1:] != rows[:-1], axis=1)))


def _hash_rows(array, mult=1000003, dtype=np.uint64):
    # Based on CPython's tuple hash
    mult = dtype(mult)
    sorted = np.sort(array, axis=1).astype(dtype)
    hash_array = 0x345678 * np.ones(array.shape[0], dtype=dtype)
    for i in reversed(range(array.shape[1])):
        hash_array = np.bitwise_xor(hash_array, sorted[:, i]) * mult
        mult += dtype(82520 + i + i)
    hash_array += dtype(97531)
    hash_array[hash_array == -1] = np.array(-2).astype(dtype)
    return hash_array


def _fix_indexing_one(index):
    """Convert a one-based index into a zero-based one"""

    def fix_slice(slc):
        start = slc.start
        stop = slc.stop
        step = slc.step
        if start is not None:
            if start > 0:
                start -= 1
            elif start == 0:
                raise IndexError("Cant get item 0 in one-indexed array")
            else:
                raise IndexError("Cant get negative slices in one-indexed array")

        if stop is not None:
            if stop > 0:
                stop -= 1
            elif stop == 0:
                raise IndexError("Cant get item 0 in one-indexed array")
            else:
                raise IndexError("Cant get negative slices in one-indexed array")

        if step is not None and step < 0:
            raise IndexError("Cant get negative slices in one-indexed array")

        return slice(start, stop, step)

    def fix_index_array(idx_array):
        idx_array = np.array(idx_array)
        if idx_array.dtype == bool:
            return idx_array
        else:
            if np.any(idx_array == 0):
                raise IndexError("Cant get item 0 in one-indexed array")
            idx_array[idx_array > 0] -= 1
            return idx_array

    if isinstance(index, (int, np.integer)):
        if index > 0:
            return index - 1
        if index < 0:
            return index
        if index == 0:
            raise IndexError("Cant get item 0 in one-indexed array")

    elif isinstance(index, slice):
        return fix_slice(index)

    elif isinstance(index, (list, np.ndarray)):
        return fix_index_array(index)

    elif isinstance(index, tuple):
        index = list(index)
        if isinstance(index[0], (list, np.ndarray)):
            index[0] = fix_index_array(index[0])
        if isinstance(index[0], slice):
            index[0] = fix_slice(index[0])
        return tuple(index)

    else:
        return index


def _getitem_one_indexed(array, index):
    index = _fix_indexing_one(index)
    return array.__getitem__(index)


def _compute_signed_tetrahedra_volume(mesh: npt.NDArray):
    """Compute the signed volume of an array of tetrahedra

    Parameters
    ----------
    mesh : npt.NDArray
        Mesh of size (N, 4, 3) where N is the number of tetrahedra.

    Returns
    -------
    npt.NDArray
        Signed volume of each tetrahedron.
    """
    a = mesh[:, 1] - mesh[:, 0]
    b = mesh[:, 2] - mesh[:, 0]
    c = mesh[:, 3] - mesh[:, 0]
    return np.einsum("ij,ij->i", np.cross(a, b), c) / 6.0
