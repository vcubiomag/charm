import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import nibabel as nib
import numpy as np
import scipy.ndimage
import scipy.sparse
import scipy.sparse.csgraph

from charm.logging import logger
from charm.mesh import cgal
from charm.mesh._mmg3d import remesh
from charm.mesh.element_tags import ElementTags
from charm.mesh.io import read_medit, write_msh
from charm.mesh.msh import (
    Elements,
    Msh,
    Nodes,
    _node_to_elements,
    element_grid_coordinates,
)
from charm.utils import nifti, transformations
from charm.utils._thickness import calc_thickness
from charm.utils.morphology import dilate, dilate_cross, erosion
from charm.utils.timing import format_time
from charm.utils.transformations import get_vox_size


class MeshingError(ValueError):
    pass


def _write_inr(image, voxel_dims, fn_out):
    if image.ndim != 3:
        raise MeshingError("Nifti volume must have 3 dimensions")
    if len(voxel_dims) != 3:
        raise MeshingError("Voxel_dims must have 3 dimensions")

    if image.dtype in [np.uint8, np.uint16]:
        inria_dtype = "unsigned fixed"
    elif image.dtype in [np.float32, np.float64]:
        inria_dtype = "float"
    else:
        raise MeshingError(
            f"Image data type: {image.dtype} not compatible. compatible types are: uint8, uint16, float32, float64"
        )

    n_bits = 8 * image.itemsize
    with open(fn_out, "bw") as f:
        header = b"#INRIMAGE-4#{\n"
        header += "XDIM={}\nYDIM={}\nZDIM={}\nVDIM=1\n".format(*image.shape).encode()
        header += f"TYPE={inria_dtype}\nPIXSIZE={n_bits} bits\nCPU=decm\n".encode()
        header += "VX={}\nVY={}\nVZ={}\n".format(*voxel_dims).encode()
        # INR headers are padded to 256 bytes
        header += (252 - len(header)) * b"\n" + b"##}\n"
        f.write(header)
        f.write(image.tobytes(order="F"))


def _mesh_image(
    image,
    voxel_dims,
    facet_angle,
    facet_size,
    facet_distance,
    cell_radius_edge_ratio,
    cell_size,
    do_perturb,
    do_exude,
    do_lloyd,
):
    with tempfile.TemporaryDirectory() as tmpdir:
        fn_image = os.path.join(tmpdir, "image.inr")
        fn_mesh = os.path.join(tmpdir, "mesh.mesh")
        _write_inr(image, voxel_dims, fn_image)
        if type(cell_size) is np.ndarray:
            ret = cgal.mesh_image_sizing_field(
                fn_image,
                fn_mesh,
                facet_angle,
                facet_size,
                facet_distance,
                cell_radius_edge_ratio,
                cell_size,
                do_perturb,
                do_exude,
                do_lloyd,
            )
        else:
            ret = cgal.mesh_image(
                fn_image,
                fn_mesh,
                facet_angle,
                facet_size,
                facet_distance,
                cell_radius_edge_ratio,
                cell_size,
                do_perturb,
                do_exude,
                do_lloyd,
            )

        if ret != 0:
            raise MeshingError("There was an error while meshing the image")

        mesh = read_medit(fn_mesh)
    # In concurrent meshing there might be some spurious nodes
    used_nodes = np.unique(mesh.elm[:])[1:]
    mesh = mesh.crop_mesh(nodes=used_nodes)

    return mesh


def _decompose_affine(affine):
    """Decompose an affine transformation into A = RZS, with R a rotation, Z a
    scaling and S a shearing matrix
    """
    if np.isclose(np.linalg.det(affine), 0):
        raise ValueError("Affine matrix is singular!")
    rot, z = np.linalg.qr(affine[:3, :3])
    scaling = np.diagonal(z)
    rot *= np.sign(scaling)
    z *= np.sign(scaling)[:, None]
    scaling = np.abs(scaling)
    shearing = np.diag(1 / scaling).dot(z)
    return rot, scaling, shearing


def image2mesh(
    image,
    affine,
    facet_angle=30,
    facet_size=None,
    facet_distance=None,
    cell_radius_edge_ratio=3,
    cell_size=None,
    do_perturb=False,
    do_exude=False,
    do_lloyd=False,
):
    """Creates a tetrahedral mesh from a labeled 3D image with CGAL

    Parameters
    ------------
    image: ndarray
        Labeled 3D image of type uint8 or uint16

    affine: 4x4 ndarray
        Affine transformation from voxel space to world space. It must decompose into
        R*Z, with R a rotation and Z a diagonal scaling matrix (no shearing).

    facet_angle: float (optional)
        Lower bound for the angle (in degrees) of surface facets. When boundary surfaces
        are smooth, meshing is guaranteed to terminate if this is at most 30 degrees.
        Default: 30

    facet_size: float or ndarray with the shape of image (optional)
        Upper bound for the radii of the surface Delaunay balls, which circumscribe a
        surface facet and are centered on the surface patch.
        Default: minimum voxel size (very low!)

    facet_distance: float or ndarray with the shape of image (optional)
        Upper bound for the distance between the circumcenter of a surface facet and the
        center of its surface Delaunay ball. This controls the approximation error of the
        surfaces. Default: minimum voxel size

    cell_radius_edge_ratio: float (optional)
        Upper bound for the ratio between the circumradius of a tetrahedron and its
        shortest edge. It does not filter slivers. Delaunay refinement is guaranteed to
        terminate for values above 2. Default: 3

    cell_size: float or ndarray with the shape of image (optional)
        Upper bound for the circumradii of the tetrahedra.
        Default: minimum voxel size (very low!)

    do_perturb: bool
        Apply sliver perturbation in CGAL after meshing (default = False).
    do_exude: bool
        Apply exudation in CGAL after meshing (default = False).
    do_lloyd: bool
        Apply Lloyd optimization in CGAL after meshing (default = False).

    Returns
    ----------
    msh: charm.mesh.msh.Msh
        Mesh structure

    References
    -----------
    https://doc.cgal.org/latest/Mesh_3/index.html
    """
    if image.dtype not in [np.uint8, np.uint16]:
        raise MeshingError("Image must be of type uint8 or uint16")

    rot, voxel_dims, shearing = _decompose_affine(affine)

    if not np.allclose(shearing, np.eye(3), atol=1.0e-5):
        raise ValueError("Affine matrix has a shearing component")

    if facet_size is None:
        facet_size = min(voxel_dims)

    if facet_distance is None:
        facet_distance = min(voxel_dims)

    if cell_size is None:
        cell_size = min(voxel_dims)

    if type(cell_size) is np.ndarray or type(facet_size) is np.ndarray or type(facet_distance) is np.ndarray:
        if type(cell_size) is np.ndarray:
            assert cell_size.shape == image.shape
        else:
            cell_size = cell_size * np.ones_like(image, dtype=np.float32, order="F")

        if type(facet_size) is np.ndarray:
            assert facet_size.shape == facet_size.shape
        else:
            facet_size = facet_size * np.ones_like(image, dtype=np.float32, order="F")

        if type(facet_distance) is np.ndarray:
            assert facet_distance.shape == facet_distance.shape
        else:
            facet_distance = facet_distance * np.ones_like(image, dtype=np.float32, order="F")

    mesh = _mesh_image(
        image,
        voxel_dims,
        facet_angle,
        facet_size,
        facet_distance,
        cell_radius_edge_ratio,
        cell_size,
        do_perturb,
        do_exude,
        do_lloyd,
    )
    mesh.nodes.node_coord = rot.dot(mesh.nodes.node_coord.T).T
    mesh.nodes.node_coord += affine[:3, 3]
    # A reflection flips the orientation of the tetrahedra
    if np.linalg.det(rot) < 0:
        mesh.elm.node_number_list = mesh.elm.node_number_list[:, [1, 0, 2, 3]]
    return mesh


def _sizing_field_from_thickness(label, thickness, elem_sizes):
    """Calculates a sizing field from thickness,
    with the option to define label-wise sizing rules"""
    slope = elem_sizes["standard"]["slope"]
    ranges = elem_sizes["standard"]["range"]
    if ranges[0] > ranges[1]:
        raise ValueError("Ranges value should be in format (min, max)")
    field = np.clip(np.array(slope * thickness, dtype=np.float32, order="F"), *ranges)

    for tissue, sizes in elem_sizes.items():
        if tissue == "standard":
            continue
        slope = sizes["slope"]
        ranges = sizes["range"]
        if ranges[0] > ranges[1]:
            raise ValueError("Ranges value should be in format (min, max)")
        idx = label == int(tissue)
        field[idx] = np.clip(slope * thickness[idx], *ranges)
    return field


def _get_connectivity_matrix(faces, nr_nodes=None):
    """Connectivity matrix of the surface nodes of the triangles

    Parameters
    ----------
    faces:
        n_facesx3 ndarray of triangle nodes
    nr_nodes:
        total number of nodes (set to np.max(faces)+1 if left empty)

    Returns
    -------
    boolean csr sparse matrix (nr_nodes x nr_nodes)
    """
    if nr_nodes is None:
        nr_nodes = np.max(faces) + 1
    rows = faces[:, [0, 0, 1, 1, 2, 2]].ravel()
    columns = faces[:, [1, 2, 0, 2, 0, 1]].ravel()
    C = scipy.sparse.csr_matrix((np.ones(rows.size, dtype=bool), (rows, columns)), shape=(nr_nodes, nr_nodes))
    C.sum_duplicates()
    return C


def _row_modes(rows):
    """The most frequent value of each row, the lowest of them for a tie, and how often it occurs."""
    rows = np.sort(rows, axis=1)
    counts = (rows[:, :, None] == rows[:, None, :]).sum(axis=2)
    first = counts.argmax(axis=1)
    index = np.arange(len(rows))
    return rows[index, first], counts[index, first]


def _get_surfaces(faces, tet_faces, adj_tets, tag, nr_nodes):
    """Reconstructs the surfaces between tets of different labels

    Parameters
    ----------
    faces:
        n_facesx3 ndarray of triangle nodes
    tet_faces:
        n_tetsx4 ndarray of tet faces (indices into the faces array)
    adj_tets:
        n_tetx4 ndarray of tet neighbors (-1 in case of "air")
    tag:
        n_tetx1 ndarray of tet labels
    nr_nodes:
        total number of nodes

    Returns
    -------
    idx_surface_tri:
        ndarray of surface triangles (indices into the faces array)
    face_node_diff:
        nr_nodesx1 ndarray indicating difference between number of surface
        faces connected to a node and number of neigbhor surface nodes
    nneighb:
        nr_nodesx1 ndarray indicating number of neigbhor surface nodes
    conn_nodes:
        connectivity matrix of surface nodes (boolean csr sparse matrix,
        nr_nodes x nr_nodes)
    """
    adj_labels = tag[adj_tets]
    adj_labels[adj_tets == -1] = -1
    adj_diff = adj_labels - tag.reshape((len(tag), 1)) != 0
    adj_diff[tag == -1] = False  # no triangles for "air" tetrahedra

    idx_surface_tri = np.unique(tet_faces[adj_diff])
    conn_nodes = _get_connectivity_matrix(faces[idx_surface_tri], nr_nodes)
    nneighb = conn_nodes.getnnz(axis=1)
    nfaces = np.bincount(faces[idx_surface_tri].reshape(-1), minlength=nr_nodes)
    # face_node_diff > 0 indicates a more complex surface topology (T-junction, ...)
    face_node_diff = nfaces - nneighb

    return idx_surface_tri, face_node_diff, nneighb, conn_nodes


def _get_elm_and_new_tag(tag, adj_tets, nr_diff, return_diffmat=False):
    """Get all tets with nr_diff (2, 3 or 4) neighboring tets of different
    labels, and the new label from those neighbors

    Parameters
    ----------
    tag:
        n_tetx1 ndarray of tet labels
    adj_tets:
        n_tetx4 ndarray of tet neighbors (-1 in case of "air")
    nr_diff:
        required number of neighbors with different labels (2, 3 or 4)
    return_diffmat (optional; Default: False):
        whether to return a boolean matrix indicating neighbors with different labels

    Returns
    -------
    idx_elm:
        ndarray of selected tets
    new_tag:
        new tag based on the neighbor labels
    adj_diff (optional):
        n_tetx4 boolean ndarray indicating neighbor tets with different labels

    Notes
    -------
    * nr_diff == 4: the most frequent neighbor label is selected
    * nr_diff == 3: only tets are returned where at least 2 neighbors
                    have the same label; the most frequent neighbor label is
                    returned
    * nr_diff == 2: only tets are returned where the two neighbors
                    have the same label
    """
    adj_labels = tag[adj_tets]
    adj_labels[adj_tets == -1] = -1
    adj_diff = adj_labels - tag.reshape((len(tag), 1)) != 0

    idx_elm = np.where(np.sum(adj_diff, axis=1) == nr_diff)[0]
    adj_labels = adj_labels[idx_elm]
    adj_labels = adj_labels[adj_diff[idx_elm]].reshape((-1, nr_diff))

    if nr_diff == 4:
        new_tag = _row_modes(adj_labels)[0]
    elif nr_diff == 3:
        new_tag, n_occ = _row_modes(adj_labels)
        idx_relabel = n_occ > 1
        idx_elm = idx_elm[idx_relabel]
        new_tag = new_tag[idx_relabel]
    elif nr_diff == 2:
        idx_relabel = np.diff(adj_labels).flatten() == 0
        idx_elm = idx_elm[idx_relabel]
        new_tag = adj_labels[idx_relabel, 0]
    else:
        raise ValueError("nr_diff has to be 2,3 or 4")

    if return_diffmat:
        return idx_elm, new_tag, adj_diff
    return idx_elm, new_tag


def _get_test_nodes(
    faces,
    tet_faces,
    adj_tets,
    idx_surface_tri,
    face_node_diff,
    nneighb,
    node_coord,
    tag,
    node_number_list,
    fast_track=False,
):
    """Surface nodes that might be spikes and need testing

    Parameters
    ----------
    faces:
        n_facesx3 ndarray of triangle nodes
    tet_faces:
        n_tetsx4 ndarray of tet faces (indices into the faces array)
    adj_tets:
        n_tetx4 ndarray of tet neighbors (-1 in case of "air")
    idx_surface_tri:
        ndarray of surface triangles (indices into the faces array)
    face_node_diff:
        nr_nodesx1 ndarray indicating difference between number of surface
        faces connected to a node and number of neigbhor surface nodes
    nneighb:
        nr_nodesx1 ndarray indicating number of neigbhor surface nodes
    node_coord:
        n_nodesx3 ndarray of node positions
    tag:
        n_tetx1 ndarray of tet labels
    node_number_list:
            n_tetx4 ndarray of node indices
    fast_track: bool (standard: False)
            when set to True, only nodes connected to three different
            regions are detected as putative spike nodes. This leads
            to far less nodes that need to be tested, making the next
            steps faster. However, spikes, e.g. in GM sulci will then
            not be detected.

    Returns
    -------
    idx_test_nodes :
        ndarray of selected candidate nodes

    Notes
    -------
    * uses heuristics to lower the number of candidate nodes
      while not losing too many real spike nodes
    * required as _get_spikes_from_conn_matrix is rather slow
    """
    idx_test_nodes = np.zeros(len(nneighb), dtype=bool)

    if not fast_track:
        idx_elm, _, adj_diff = _get_elm_and_new_tag(tag, adj_tets, 3, return_diffmat=True)
        # The node shared by the three tet faces facing the different neighbor tets
        faces_elm = tet_faces[idx_elm]
        faces_elm = faces_elm[adj_diff[idx_elm]].reshape((-1, 3))
        idx_node = _row_modes(faces[faces_elm].reshape((-1, 9)))[0]
        if len(idx_node) > 0:
            idx_test_nodes[idx_node] = True

        idx_elm = _get_elm_and_new_tag(tag, adj_tets, 2)[0]
        # The 2 nodes shared by the two tet faces facing the different neighbor tets
        faces_elm = tet_faces[idx_elm]
        faces_elm = faces_elm[adj_diff[idx_elm]].reshape((-1, 2))
        idx_node = np.sort(faces[faces_elm].reshape((-1, 6)))
        idx_node = idx_node[:, :5][np.diff(idx_node) == 0]
        if len(idx_node) > 0:
            idx_test_nodes[idx_node] = True

        # The steps above add too many nodes, so exclude nodes with low surface
        # curvature even though a few true spikes get lost
        m_surf = Msh()
        m_surf.nodes.node_coord = node_coord
        m_surf.elm.add_triangles(faces[idx_surface_tri, :] + 1, np.ones(len(idx_surface_tri), dtype=int))
        idx_test_nodes *= np.abs(m_surf.gaussian_curvature()) > 0.1

    # Add nodes in tets of three different regions, not counting air tets
    n_tags = np.max(tag) + 1
    not_air = tag != -1
    node_tags = np.unique((node_number_list[not_air].astype(np.int64) - 1) * n_tags + tag[not_air, None])
    n_node_tags = np.bincount(node_tags // n_tags, minlength=len(nneighb))
    idx_test_nodes += n_node_tags == 3

    # A spike node needs at least 6 neighbor nodes
    idx_test_nodes *= nneighb > 5
    # Exclude more complex surface geometries (e.g. T-junctions)
    idx_test_nodes *= face_node_diff == 0
    # Exclude nodes on outer faces
    idx = tet_faces[adj_tets == -1]
    idx = np.unique(faces[idx])
    idx_test_nodes[idx] = False

    return idx_test_nodes


def _get_spikes_from_conn_matrix(conn_nodes, idx_test_nodes, nneighb):
    """Test which nodes in idx_test_nodes are spike nodes

    Parameters
    ----------
    conn_nodes: boolean sparse matrix (nr_nodes x nr_nodes)
        node connectivity matrix
    idx_test_nodes: ndarray
        mask of the candidate nodes
    nneighb: nr_nodesx1 ndarray
        number of neighbor surface nodes

    Returns
    -------
    ndarray of spike node indices

    Notes
    ------
    * nodes belonging to more complex surface configurations
     (i.e. face_node_diff > 0) must not be candidate nodes
    """
    C = conn_nodes.tocsr()
    C.sum_duplicates()
    n_nodes = C.shape[0]
    idx_test_nodes = np.flatnonzero(idx_test_nodes)

    # The neighbors of all test nodes side by side, with `rows` the test node and `neighbors` the neighbor
    neighbors_of_test = C[idx_test_nodes]
    rows = np.repeat(np.arange(len(idx_test_nodes)), neighbors_of_test.getnnz(axis=1))
    neighbors = neighbors_of_test.indices
    keys = rows.astype(np.int64) * n_nodes + neighbors

    # The edges between neighbors of the same test node, as positions in those arrays
    n_candidates = np.diff(C.indptr)[neighbors]
    source = np.repeat(np.arange(len(neighbors)), n_candidates)
    starts = np.repeat(C.indptr[neighbors], n_candidates)
    offsets = np.arange(n_candidates.sum()) - np.repeat(np.cumsum(n_candidates) - n_candidates, n_candidates)
    query = rows[source].astype(np.int64) * n_nodes + C.indices[starts + offsets]
    target = np.searchsorted(keys, query)
    target[target == len(keys)] = 0
    is_edge = keys[target] == query
    source = source[is_edge]
    target = target[is_edge]

    # Almost always all the neighbors have two neighbors themselves, which makes them a set of rings:
    # a spike if there is more than one. Connected components of all of them at once tell that.
    degree = np.bincount(source, minlength=len(neighbors))
    is_ring = np.bincount(rows[degree != 2], minlength=len(idx_test_nodes)) == 0
    _n_labels, labels = scipy.sparse.csgraph.connected_components(
        scipy.sparse.coo_matrix(
            (np.ones(source.size, dtype=bool), (source, target)),
            shape=(len(neighbors), len(neighbors)),
        ),
        directed=False,
    )
    _, first_of_component = np.unique(labels, return_index=True)
    n_components = np.bincount(rows[first_of_component], minlength=len(idx_test_nodes))
    is_spike = is_ring & (n_components > 1)

    for k in np.flatnonzero(~is_ring):
        idx = C.indices[C.indptr[idx_test_nodes[k]] : C.indptr[idx_test_nodes[k] + 1]]
        is_spike[k] = _neighbors_are_disconnected(C[idx][:, idx].toarray())

    return idx_test_nodes[is_spike]


def _neighbors_are_disconnected(c):
    """Test on the boolean connectivity matrix of the neighbors of a node with
    a topology more complex than a ring of them, whether they are not all connected."""
    # Resolve topologies where some nodes have more than 2 neighbors. A node with more than 2
    # neighbors, at least two of which have exactly 2, is cut from the nodes without exactly 2.
    nnz = c.sum(axis=0)
    while np.any(nnz > 2):
        idx_2 = nnz == 2
        idx_two2neighb = np.flatnonzero((c.astype(int) @ idx_2 > 1) & ~idx_2)
        if len(idx_two2neighb) == 0:
            break
        idx_not2 = np.flatnonzero(~idx_2)
        c[np.ix_(idx_two2neighb, idx_not2)] = False
        c[np.ix_(idx_not2, idx_two2neighb)] = False
        previous_nnz = nnz
        nnz = c.sum(axis=0)
        if np.array_equal(nnz, previous_nnz):
            break

    reached = c[:, 0].copy()
    n_nodes = len(reached)
    for _ in range(n_nodes // 2 - 1):
        reached |= c @ reached
    return n_nodes != reached.sum()


def _get_new_tag_for_spikes(idx_spike_nodes, adj_tets, elm, tag, node_nr):
    """Resolve spikes by relabeling some of the tetrahedra of each spike node.

    The tetrahedra of a spike node are sorted into groups that share faces
    and have the same label. The smallest group takes the label of the group
    it shares faces with.

    Parameters
    ----------
    idx_spike_nodes : ndarray
        indices of the spike nodes
    adj_tets:
        n_tetx4 ndarray of tet neighbors (-1 in case of "air")
    elm:
        n_tetx4 ndarray of node indices
    tag:
        n_tetx1 ndarray of tet labels
    node_nr:
        total number of nodes

    Returns
    -------
    tag:
        updated n_tetx1 ndarray of tet labels
    spike_data: list of tuples of the form
        (idx_spikenode (1-based), idx_spike_tets, tag_old, tag_new)

    Notes
    ------
    * elm must contain only tetrahedra
    """
    # Restrict to the tets around spike nodes for speed
    idx = np.zeros(node_nr, dtype=bool)
    idx[idx_spike_nodes] = True
    elm = np.copy(elm) - 1

    map_new_old = np.where(np.any(idx[elm], axis=1))[0]
    map_old_new = -1 * np.ones(len(elm), dtype=int)
    map_old_new[map_new_old] = np.arange(len(map_new_old))

    elm = elm[map_new_old]
    tag_spk = tag[map_new_old]
    adj_tets_spk = map_old_new[adj_tets[map_new_old]]

    tag_buff = -1 * np.ones_like(tag_spk)
    idx_spike_tets = np.empty(0, dtype=int)
    spike_data = []
    ptr, node_tets = _node_to_elements(elm, node_nr)
    for idx_node in idx_spike_nodes:
        idx_all_tets = node_tets[ptr[idx_node] : ptr[idx_node + 1]]
        idx_tets = np.copy(idx_all_tets)

        len_idx_spike_tets = 1000000
        while len(idx_tets) > 0:
            idx_group = [idx_tets[0]]
            tag_group = tag_spk[idx_group[0]]
            len_idx_group = 0
            while len(idx_group) > len_idx_group:
                len_idx_group = len(idx_group)
                idx_group = np.append(idx_group, adj_tets_spk[idx_group].flatten())
                idx_group = idx_group[(idx_group != -1) * (tag_spk[idx_group] == tag_group)]
                idx_group = np.intersect1d(idx_group, idx_tets)
            if len_idx_group < len_idx_spike_tets:
                idx_spike_tets = idx_group
                len_idx_spike_tets = len_idx_group
            idx_tets = np.setdiff1d(idx_tets, idx_group)

        idx_neigh = np.intersect1d(adj_tets_spk[idx_spike_tets], idx_all_tets)
        idx_neigh = np.setdiff1d(idx_neigh, idx_spike_tets)
        tag_neigh = tag_spk[idx_neigh]
        if np.any(np.diff(tag_neigh)):
            logger.warning("ambiguous new tag for node " + str(idx_node))
        tag_buff[idx_spike_tets] = tag_neigh[0]

        spike_data.append(
            (
                idx_node + 1,
                map_new_old[idx_spike_tets],
                tag_spk[idx_spike_tets[0]],
                tag_neigh[0],
            )
        )

    new_tag = -1 * np.ones_like(tag)
    new_tag[map_new_old] = tag_buff
    tag = np.copy(tag)
    tag[new_tag != -1] = new_tag[new_tag != -1]
    return tag, spike_data


def _get_candidates_for_splitting(sp_dat, node_number_list, idx_surf_nodes):
    """
    Determines all spikes that might be suited for splitting. The basic idea is
    that a spike can be split if there is a second node to which all tets of the
    spike are also connected. Then, the spike can be split by adding a new node
    in the middle of the line between the orignal spike node and the second node.

    Parameters
    ----------
    sp_dat : list of tuples of the form
        (idx_spikenode (1-based), idx_spike_tets, tag_old, tag_new)
    node_number_list :
        n_tetx4 ndarray of node indices
    idx_surf_nodes : boolean ndarray
        True for nodes at region boundaries

    Returns
    -------
    splittest : list of tuples of the form
        (idx_spikenode (1-based), idx_2nd_node, tag_old, tag_new)
    sp2_spike_tets : nx2 ndarray
        tet indices of spikes having 2 tets
        (see _combine_small_spikes for further information)
    sp2_unique_nodes : nx2 ndarray
        for the spikes with 2 tets, the nodes in only one of the two tets
    """
    splittest = []
    sp2_spike_tets = np.empty((0, 2), dtype="int32")
    sp2_unique_nodes = np.empty((0, 2), dtype="int32")

    for sp in sp_dat:
        idx_sp_node = sp[0]
        idx_sp_tets = sp[1]
        old_tag = sp[2]
        new_tag = sp[3]

        if len(idx_sp_tets) < 2:
            logger.info(" _get_candidates_for_splitting: spike with one tetrahedron should not occur at this stage")

        elif len(idx_sp_tets) == 2:
            un, cn = np.unique(node_number_list[idx_sp_tets], return_counts=True)
            idx = np.where(cn == 1)[0]
            if len(idx) == 2:
                sp2_unique_nodes = np.append(sp2_unique_nodes, un[idx].reshape((1, 2)), axis=0)
                sp2_spike_tets = np.append(sp2_spike_tets, idx_sp_tets.reshape((1, 2)), axis=0)
            else:
                logger.info(" _get_candidates_for_splitting: weird spike with two tets")
        else:
            # Look for a 2nd node to which all tets of the spike are connected
            un, cn = np.unique(node_number_list[idx_sp_tets], return_counts=True)
            idx = np.where((un != idx_sp_node) * (cn == len(idx_sp_tets)))[0]
            if len(idx) == 0:
                # Otherwise pick the one non-surface node, if there is exactly one
                idx2 = np.where(~idx_surf_nodes[un - 1])[0]
                if len(idx2) == 1:
                    splittest.append((idx_sp_node, un[idx2[0]], old_tag, new_tag))
                    # The tets of the spike not connected to the 2nd node are not
                    # tested and get the standard new tag
            elif len(idx) == 1:
                splittest.append((idx_sp_node, un[idx[0]], old_tag, new_tag))
            else:
                logger.info(" _get_candidates_for_splitting: weird spike with two fully connected 2nd nodes")

    return splittest, sp2_spike_tets, sp2_unique_nodes


def _select_splits_from_candidates(splittest, node_number_list, node_coord, tag_org):
    """
    Determines the spikes that will be split from the candidates.
    The spikes have to fulfill two criteria:
        * All tets connected to both the spike node and the 2nd node have to
          have the same tag.
        * When projected on the line between the spike node and the 2nd node,
          all other nodes of these tets have to fall approx. in the middle of
          the line.

    Parameters
    ----------
    splittest : list of tuples of the form
        (idx_spikenode (1-based), idx_2nd_node, tag_old, tag_new)
    node_number_list :
        n_tetx4 ndarray of node indices
    node_coord :
        n_nodesx3 ndarray of node positions.
    tag_org :
        n_tetx1 ndarray of tet labels BEFORE any spike removal

    Returns
    -------
    splitlist : list of tuples of the form
        (idx_spikenode (1-based), idx_2nd_node, tag_old, tag_new)
    """
    splitlist = []
    ptr, node_tets = _node_to_elements(
        node_number_list - 1,
        len(node_coord),
        np.array([sp[:2] for sp in splittest], dtype=int).reshape(-1) - 1,
    )
    for sp in splittest:
        idx_n1 = sp[0]
        idx_n2 = sp[1]
        idx_orgtets = np.intersect1d(
            node_tets[ptr[idx_n1 - 1] : ptr[idx_n1]],
            node_tets[ptr[idx_n2 - 1] : ptr[idx_n2]],
            assume_unique=True,
        )
        if len(idx_orgtets) == 0:
            raise ValueError(
                "The two nodes are not connected! This happens sporadically. Running the meshing again (charm subID --mesh) should solve the problem."
            )

        if np.max(tag_org[idx_orgtets]) != np.min(tag_org[idx_orgtets]):
            # Happens when the 2nd node is in the "ring" of surface nodes around the first node
            continue

        un = np.unique(node_number_list[idx_orgtets])
        idx_others = un[(un != idx_n1) * (un != idx_n2)]

        v1 = node_coord[idx_n1 - 1, :] - node_coord[idx_n2 - 1, :]
        v2 = node_coord[idx_n1 - 1, :] - node_coord[idx_others - 1, :]
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2, axis=1)
        cosalpha = np.sum(v1 * v2, axis=1) / (norm_v1 * norm_v2)

        if np.all(np.abs(cosalpha - 0.5) < 0.3):  # 0.3 is a magic number determined during initial testing
            splitlist.append(sp)

    return splitlist


def _combine_small_spikes(sp2_unique_nodes, sp2_spike_tets, adj_tets, node_number_list, tag_org, nr_nodes):
    """
    At thin interfaces, two spikes with each 2 tets can be directly next to each other.
    The function combines them to a common spike with 4 tets and returns them as list
    of spikes for splitting.

    Parameters
    ----------
    sp2_unique_nodes : nx2 ndarray
        for the spikes with 2 tets, the nodes in only one of the two tets
    sp2_spike_tets : nx2 ndarray
        tet indices of spikes having 2 tets
    adj_tets :
        n_tetx4 ndarray of tet neighbors (-1 in case of "air")
    node_number_list :
        n_tetx4 ndarray of node indices
    tag_org :
        n_tetx1 ndarray of tet labels BEFORE any spike removal
    nr_nodes :
        total number of nodes

    Returns
    -------
    splitlist : list of tuples of the form
        (idx_spikenode (1-based), idx_2nd_node, tag_old, tag_new)
    """
    sp2_unique_nodes = np.sort(np.copy(sp2_unique_nodes), axis=1)
    un, cn = np.unique(sp2_unique_nodes, axis=0, return_counts=True)
    un = un[cn > 1]

    splitlist = []
    for i in un:
        idx_spikes = np.where(np.all(i == sp2_unique_nodes, axis=1))[0]
        un2, cn2 = np.unique(node_number_list[sp2_spike_tets[idx_spikes]], return_counts=True)
        idx_sp_nodes = un2[cn2 == 4]

        if len(idx_sp_nodes) == 2:
            _, sp_dat_hlp = _get_new_tag_for_spikes(idx_sp_nodes - 1, adj_tets, node_number_list, tag_org, nr_nodes)
            splitlist.append((idx_sp_nodes[0], idx_sp_nodes[1], sp_dat_hlp[1][3], sp_dat_hlp[0][3]))

    return splitlist


def _split_spikes(m, splitlist):
    """Splits the spikes in the splitlist and updates the tags, in place

    Parameters
    ----------
    m : charm.mesh.msh.Msh
    splitlist : list of tuples of the form
        (idx_spikenode (1-based), idx_2nd_node, tag_old, tag_new)

    Returns
    -------
    idx_splittets :
        indices of the split tets (for visualization and debugging)
    """
    splitlist = np.array(splitlist)
    if len(splitlist) > 0:
        splitlist[:, :2] = np.sort(splitlist[:, :2], axis=1)
        idx = np.unique(splitlist[:, :2], return_index=True, axis=0)[1]
        splitlist = splitlist[idx]

    splits = m.split_tets_along_lines(splitlist[:, :2]) if len(splitlist) > 0 else []
    for (idx_tets1, idx_tets2), (_, _, old_tag, new_tag) in zip(splits, splitlist):
        m.elm.tag1[idx_tets1 - 1] = new_tag
        m.elm.tag1[idx_tets2 - 1] = old_tag

    m.elm.tag2[:] = m.elm.tag1
    return np.concatenate([tets for split in splits for tets in split]) if splits else np.empty(0, dtype="int32")


def update_tag_from_label_img(m, adj_tets, vol, affine, label_GM=None, label_CSF=None):
    """Relabels tetrahedra that both
        * more likely belong to another label, based on the label image
        * have more than one face to a neighbor with a different label

    Parameters
    ----------
    m: charm.mesh.msh.Msh
    adj_tets:
        n_tetx4 ndarray of tet neighbors (-1 in case of "air")
    vol : 3d ndarray
        label image
    affine : 4x4 ndarray
        affine transformation from voxel indices to world mm coordinates
    label_GM : int, optional
        label used for GM tets. The default is None.
    label_CSF : int, optional
        label used for CSF tets. The default is None.

    Returns
    -------
    m:
        mesh with updated m.elm.tag1 and m.elm.tag2

    Notes
    ------
    * The mesh must contain only tetrahedra
    * done repeatedly until no tets are relabeled anymore (max 20 times)
    * when labels for GM and CSF are given, CSF is not relabeled to GM (avoids
      a few GM spikes)
    * updates tag1 and tag2 of m.elm
    """
    best_tag = np.zeros_like(m.elm.tag1)
    best_tag_p = np.zeros_like(m.elm.tag1, dtype=np.float32)
    coords = element_grid_coordinates(m, vol.shape, affine)
    tags = np.unique(m.elm.tag1)

    def interpolate(tag):
        return scipy.ndimage.map_coordinates((vol == tag).astype(np.float32), coords, output=np.float32, order=1)

    with ThreadPoolExecutor() as executor:
        for i, p in zip(tags, executor.map(interpolate, tags)):
            idx = p > best_tag_p
            best_tag[idx] = i
            best_tag_p[idx] = p[idx]

    def get_nr_diff_tag(tag, adj_tets):
        adj_labels = tag[adj_tets]
        adj_labels[adj_tets == -1] = -1
        adj_diff = adj_labels - m.elm.tag1.reshape((len(tag), 1)) != 0
        return np.sum(adj_diff, axis=1)

    m.elm.tag2 = np.copy(m.elm.tag1)
    for i in range(20):
        nr_diff_pre = get_nr_diff_tag(m.elm.tag1, adj_tets)
        idx_relabel = (nr_diff_pre > 1) * (best_tag != m.elm.tag1)
        if label_GM is not None:
            idx_relabel[(m.elm.get_tags(label_CSF)) & (best_tag == label_GM)] = False
        m.elm.tag1[idx_relabel] = best_tag[idx_relabel]

        # Undo the relabeling of tets that got more "spiky"
        nr_diff_post = get_nr_diff_tag(m.elm.tag1, adj_tets)
        diff_pre_post = nr_diff_pre - nr_diff_post
        idx_undo = idx_relabel * (diff_pre_post < 0)
        m.elm.tag1[idx_undo] = m.elm.tag2[idx_undo]
        if np.sum(idx_relabel) == np.sum(idx_undo):
            break

    # Tets relabeled to 0 get their original label back
    idx_relabel = m.elm.get_tags(0)
    m.elm.tag1[idx_relabel] = m.elm.tag2[idx_relabel]
    logger.info("   Relabled " + str(np.sum(m.elm.tag1 != m.elm.tag2)) + " tets")
    m.elm.tag2[:] = m.elm.tag1
    return m


def update_tag_from_tet_neighbors(m, faces, tet_faces, adj_tets, nr_iter=12):
    """Relabels tetrahedra that have
        * 4 neighbors all with a different label
        * 3 neighbors with a different label, 2 of which share a label (e.g. a
          tet with label 2 surrounded by 2, 3, 4, 4 is relabeled to 4)
        * 2 neighbors with the same different label, where a simple surface
          analysis indicates a surface defect

    This is repeated nr_iter times. Most tetrahedra that get relabeled
    repeatedly are blocked from further relabeling. Full convergence is not
    guaranteed.

    Parameters
    ----------
    m: charm.mesh.msh.Msh
    faces:
        n_facesx3 ndarray of triangle nodes
    tet_faces:
        n_tetsx4 ndarray of tet faces (indices into the faces array)
    adj_tets:
        n_tetx4 ndarray of tet neighbors (-1 in case of "air")
    nr_iter: int, optional
        number of iterations. The default is 12.

    Returns
    -------
    m:
        mesh with updated m.elm.tag1 and m.elm.tag2

    Notes
    ------
    * The mesh must contain only tetrahedra
    * updates tag1 and tag2 of m.elm
    """
    tag = np.copy(m.elm.tag1)
    tag_buffer = np.copy(m.elm.tag1)
    relabeling_allowed = np.ones_like(m.elm.tag1, dtype=bool)
    just_relabelled = np.zeros_like(m.elm.tag1, dtype=bool)
    for i in range(nr_iter):
        just_relabelled[:] = False

        for k in (4, 3):
            idx_elm, new_tag = _get_elm_and_new_tag(tag, adj_tets, k)
            tag[idx_elm] = new_tag
            just_relabelled[idx_elm] = True

        idx_elm, new_tag, adj_diff = _get_elm_and_new_tag(tag, adj_tets, 2, return_diffmat=True)
        # Exclude tets at the outer surface and those blocked from relabeling
        idx = new_tag > -1
        idx *= np.isin(idx_elm, np.where(relabeling_allowed)[0])
        idx_elm = idx_elm[idx]
        new_tag = new_tag[idx]
        # The 2 nodes shared by the two tet faces facing the different neighbor tets
        faces_elm = tet_faces[idx_elm]
        faces_elm = faces_elm[adj_diff[idx_elm]].reshape((-1, 2))
        facenodes_elm = np.sort(faces[faces_elm].reshape((-1, 6)))
        facenodes_elm = facenodes_elm[:, :5][np.diff(facenodes_elm) == 0].reshape((-1, 2))
        # The two different neighbors have the same label, so face_node_diff > 0 reveals
        # a surface defect rather than a T-junction
        _idx_surface_tri, face_node_diff = _get_surfaces(faces, tet_faces, adj_tets, tag, m.nodes.nr)[:2]
        idx = np.max(face_node_diff[facenodes_elm], axis=1) > 0
        idx_elm = idx_elm[idx]
        new_tag = new_tag[idx]
        tag[idx_elm] = new_tag

        if i > 2:
            # Stop flip-flopping between the original and a second label
            idx = (tag == m.elm.tag1) * (tag != tag_buffer)
            relabeling_allowed[idx] = False
            # Keep the relabeling of tets with two different neighbors from giving them 3 or 4 again
            idx = (tag == tag_buffer) * just_relabelled
            relabeling_allowed[idx] = False
        logger.info("     It. " + str(i) + ": relabled " + str(np.sum(tag_buffer != tag)) + " tets")
        tag_buffer[:] = tag

    logger.info("   Relabeled " + str(np.sum(m.elm.tag1 != tag)) + " tets")
    m.elm.tag1 = tag
    m.elm.tag2[:] = m.elm.tag1
    return m


def update_tag_from_surface(m, faces, tet_faces, adj_tets, do_splits=False, fast_track=False):
    """Relabels tetrahedra in localized spikes, found by analysis of the
    surface topology

    Parameters
    ----------
    m: charm.mesh.msh.Msh
    faces:
        n_facesx3 ndarray of triangle nodes
    tet_faces:
        n_tetsx4 ndarray of tet faces (indices into the faces array)
    adj_tets:
        n_tetx4 ndarray of tet neighbors (-1 in case of "air")
    do_splits: bool
        whether to split spikes that reach deep into two regions
        (standard: False)
    fast_track: bool
        whether to test only nodes connected to three different regions.
        This leads to far less nodes that need to be tested. However,
        spikes, e.g. in GM sulci will then not be removed.
        (standard: False)

    Returns
    -------
    m:
        mesh with updated m.elm.tag1 and m.elm.tag2

    Notes
    ------
    * The mesh must contain only tetrahedra
    * updates tag1 and tag2 of m.elm
    * for do_splits=True: variables faces, tet_faces, adj_tets will
      be outdated after the call, as the function adds new tetrahedra
    """
    idx_surface_tri, face_node_diff, nneighb, conn_nodes = _get_surfaces(
        faces, tet_faces, adj_tets, m.elm.tag1, m.nodes.nr
    )

    idx_test_nodes = _get_test_nodes(
        faces,
        tet_faces,
        adj_tets,
        idx_surface_tri,
        face_node_diff,
        nneighb,
        m.nodes[:],
        m.elm.tag1,
        m.elm.node_number_list,
        fast_track=fast_track,
    )

    logger.info(
        "     Testing "
        + str(np.sum(idx_test_nodes))
        + " nodes (matrix: "
        + str(conn_nodes.shape)
        + ", "
        + str(conn_nodes.nnz)
        + " entries)"
    )
    idx_spike_nodes = _get_spikes_from_conn_matrix(conn_nodes, idx_test_nodes, nneighb)

    tag_buff = m.elm.tag1
    m.elm.tag1, sp_dat = _get_new_tag_for_spikes(idx_spike_nodes, adj_tets, m.elm[:], m.elm.tag1, m.nodes.nr)
    logger.info("   Relabled " + str(np.sum(m.elm.tag1 != m.elm.tag2)) + " tets")
    m.elm.tag2[:] = m.elm.tag1

    if do_splits:
        idx_surf_nodes = conn_nodes.getnnz(axis=0) > 0
        splittest, sp2_tets, sp2_uniquenodes = _get_candidates_for_splitting(
            sp_dat, m.elm.node_number_list, idx_surf_nodes
        )
        splitlist = _select_splits_from_candidates(splittest, m.elm.node_number_list, m.nodes.node_coord, tag_buff)

        # At thin interfaces, two spikes of 2 tets each can be next to each other.
        # They are combined into one spike of 4 tets to split.
        splitlist += _combine_small_spikes(
            sp2_uniquenodes,
            sp2_tets,
            adj_tets,
            m.elm.node_number_list,
            tag_buff,
            m.nodes.nr,
        )

        n_tet_pre = m.elm.nr
        _split_spikes(m, splitlist)
        logger.info("   Split " + str(m.elm.nr - n_tet_pre) + " tets")
        m.elm.tag2 = m.elm.tag1.copy()

    return m


def _remove_spikes(m, label_img, affine, label_GM=2, label_CSF=3):
    """Runs the three spike removal steps

    Parameters
    ----------
    m : charm.mesh.msh.Msh
        mesh from cgal, has to be without surfaces!
    label_img: 3D np.ndarray in uint8 format
        Labeled image from segmentation
    affine: 4x4 np.ndarray
        Affine transformation from voxel coordinates to world coordinates
    label_GM : int, optional
        label for GM volume. The default is 2.
    label_CSF : int, optional
        label for CSF volume. The default is 3.

    Returns
    -------
    msh: charm.mesh.msh.Msh
        Mesh structure
    """
    logger.info("Removing Spikes")
    logger.info(" Step 1: Update tags from label image")
    faces, tet_faces, adj_tets = m.elm._get_tet_faces_and_adjacent_tets()
    tag_buff = m.elm.tag1.copy()
    m = update_tag_from_label_img(m, adj_tets, label_img, affine, label_GM=label_GM, label_CSF=label_CSF)

    logger.info(" Step 2: Update tags from tet neighbors")
    m = update_tag_from_tet_neighbors(m, faces, tet_faces, adj_tets)

    logger.info(" Step 3: Resolve remaining localized spikes ")
    m = update_tag_from_surface(m, faces, tet_faces, adj_tets, do_splits=True)

    logger.info(
        "Done Removing Spikes: Total number of relabled tets: "
        + str(np.sum(m.elm.tag1[: len(tag_buff)] != tag_buff))
        + "; Number of split tets: "
        + str(len(m.elm.tag1) - len(tag_buff))
    )

    idx_keep = np.where(m.elm.tag1 != -1)[0] + 1
    m = m.crop_mesh(elements=idx_keep)

    return m


def _fix_labels(m, label_img):
    """Assign the right labels to the mesh as CGAL modifies them"""
    indices_seg, label_counts = np.unique(label_img, return_counts=True)
    indices_seg = indices_seg[1:]
    label_counts = label_counts[1:]
    indices_cgal = np.unique(m.elm.tag1)
    n_dropped = len(indices_seg) - len(indices_cgal)
    if n_dropped:
        idx_keep = np.argsort(label_counts)[::-1]
        idx_keep = idx_keep[:-n_dropped]
        indices_seg = np.sort(indices_seg[idx_keep])
        logger.warning(f"{n_dropped} small region(s) dropped during meshing. Check label numbers in mesh!")
    new_tags = np.copy(m.elm.tag1)
    for i, t in enumerate(indices_seg):
        new_tags[m.elm.get_tags(i + 1)] = t
    m.elm.tag1 = new_tags
    m.elm.tag2 = new_tags.copy()
    return m


def _relabel_microtets(m, el_max=0.0001):
    """CGAL can create spurious groups of microscopic tetrahedra at region
    boundaries. Relabeling each group to its most common tag moves the
    boundary away, so that mmg fixes them even with -nosurf.

    Parameters
    ----------
    m : charm.mesh.msh.Msh
        Mesh structure.
    el_max : float, optional
        maximal edge length. Tetrahedra will be relabeled
        when all edges are shorter than el_max.
        The default is 0.0001.

    Returns
    -------
    m : charm.mesh.msh.Msh
        Mesh structure.
    """
    M = m.nodes.node_coord[m.elm.node_number_list - 1]
    # the tets with all edges short have all nodes within el_max along each axis
    near = np.flatnonzero((np.ptp(M, axis=1) < el_max).all(axis=1))
    M = M[near]
    E = M[:, [0, 0, 0, 1, 1, 2]] - M[:, [1, 2, 3, 2, 3, 3]]
    idx = near[np.max(np.linalg.norm(E, axis=2), axis=1) < el_max]

    if len(idx) == 0:
        return m

    idx_c = m.elm.connected_components(idx + 1)

    for i in idx_c:
        logger.debug(f"Relabeling group of {len(i)} micro-tets")
        m.elm.tag1[i - 1] = np.argmax(np.bincount(m.elm.tag1[i - 1]))
    m.elm.tag2[:] = m.elm.tag1
    return m


def _morton_code(points):
    """Position of each point along a Z-order curve over its bounding box."""

    def spread_bits(v):
        v = (v | (v << 16)) & 0x030000FF
        v = (v | (v << 8)) & 0x0300F00F
        v = (v | (v << 4)) & 0x030C30C3
        return (v | (v << 2)) & 0x09249249

    low = points.min(axis=0)
    cells = (points - low) / (points.max(axis=0) - low).max() * (2**10 - 1)
    x, y, z = cells.astype(np.int64).T
    return spread_bits(x) | (spread_bits(y) << 1) | (spread_bits(z) << 2)


def _spatially_ordered(nodes, tetrahedra, tetrahedra_tags, triangles):
    # CGAL's concurrent meshing leaves neighbors scattered in memory, and mmg
    # visits elements through their neighbors. Sorting more than halves its run
    # time on a 4M tetrahedra head mesh, ahead of mmg's own Scotch renumbering.
    node_order = np.argsort(_morton_code(nodes), kind="stable")
    new_number = np.empty_like(node_order)
    new_number[node_order] = np.arange(1, len(node_order) + 1)

    nodes = nodes[node_order]
    tetrahedra = new_number[tetrahedra - 1]
    tet_order = np.argsort(_morton_code(nodes[tetrahedra - 1].mean(1)), kind="stable")
    return (
        nodes,
        tetrahedra[tet_order],
        tetrahedra_tags[tet_order],
        new_number[triangles - 1],
    )


def _run_mmg(m, repeats=2, mmg_noinsert=True):
    """Improve mesh quality using mmg3d.

    Parameters
    ----------
    m : charm.mesh.msh.Msh
        Mesh structure, with triangles and tetrahedra.
    repeats : int
        Number of mmg runs.
    mmg_noinsert : bool
        Do not insert additional points.

    Returns
    -------
    m : charm.mesh.msh.Msh
        Mesh structure.
    """
    logger.info("Improving Mesh Quality")

    for i in range(repeats):
        is_tri = m.elm.elm_type == 2
        is_tet = m.elm.elm_type == 4
        nodes, tets, tet_tags, tris = _spatially_ordered(
            m.nodes.node_coord.astype(np.float64),
            m.elm.node_number_list[is_tet].astype(np.int64),
            m.elm.tag1[is_tet].astype(np.int64),
            m.elm.node_number_list[is_tri, :3].astype(np.int64),
        )
        nodes, tets, tet_tags, tris, tri_tags = remesh(
            np.ascontiguousarray(nodes),
            np.ascontiguousarray(tets),
            np.ascontiguousarray(tet_tags),
            np.ascontiguousarray(tris),
            np.ascontiguousarray(m.elm.tag1[is_tri], dtype=np.int64),
            noinsert=mmg_noinsert,
            hsiz=None if mmg_noinsert else 100.0,
            hmin=None if mmg_noinsert else 1.3,
        )
        elements = Elements(triangles=tris, tetrahedra=tets)
        elements.tag1 = np.concatenate((tri_tags, tet_tags)).astype(elements.tag1.dtype)
        elements.tag2 = elements.tag1.copy()
        m = Msh(Nodes(nodes), elements)

        logger.info(f"Tetraedras after remeshing run {i + 1}: {len(m.elm.tetrahedra)}")

    return m


def create_mesh(
    label_img,
    affine,
    elem_sizes=None,
    smooth_size_field=2,
    skin_facet_size=2.0,
    facet_distances=None,
    optimize=False,
    remove_spikes=True,
    skin_tag=1005,
    hierarchy=None,
    apply_cream=True,
    smooth_steps=5,
    skin_care=20,
    mmg_noinsert=False,
    debug=False,
    debug_path="",
):
    """Create a mesh from a labeled image.

    The maximum element sizes (CGAL facet_size and cell_size) are controlled
    by elem_sizes:
        size = clip(slope * thickness, *range)
    where "thickness" is the local tissue thickness and "range" is the size
    range of the label in elem_sizes, or the "standard" one. The CGAL
    facet_distance is calculated the same way from facet_distances. This lets
    the meshing adjust sizes to local needs.

    Parameters
    ----------
    label_img: 3D np.ndarray in uint8 or uint16 format
        Labeled image from segmentation
    affine: 4x4 np.ndarray
        Affine transformation from voxel coordinates to world coordinates
    elem_sizes: dictionary (optional)
        Lists the relationship between thickness and elem_sizes.
        Label-specific relationships can be added if needed, e.g. for label 2:
            {"standard": {"range": [1, 5], "slope": 1.0},
                    "2": {"range": [1, 2], "slope": 0.7}}
        "range" determines the minimum and maximum values for element sizes.
        "slope" determines relationship between thickness and element sizes.
        Note: Label indices are used as keys, and need to be provided as string
        Default: {"standard": {"range": [1, 5], "slope": 1.0}}
    smooth_size_field: int (optional)
        Defines the size of a triangular kernel to smooth the size field. A bit
        of smoothing helps to remove the effect of a few outliers in the
        thickness estimates on the size field. Set to 0 to disable.
        The kernel size is 2*smooth_size_field+1 in voxels. Default:  2
    skin_facet_size: float (optional)
        Maximum size for the triangles of the outermost surface. If set to None,
        the elements of the outermost surface will be scaled the same way as
        the other surfaces. Default:  2.0
    facet_distances: dictionary (optional)
        Relationship between thickness and facet_distance. For small
        facet_distance values, the meshing will follow the label boundaries
        in the original image more strictly. This also means more elements.
        Label-specific relationships can be added if needed.
        "range": Minimum and maximum values for facet_distance
        "slope": Steepness of relationship between thickness and facet_distance.
        Default: {"standard": {"range": [0.1, 3], "slope": 0.5}}
    optimize: bool (optional)
        Whether to apply sliver perturbation, exudation, and lloyd optimization
        to the mesh with CGAL. Default: False
    remove_spikes: bool (optional)
        Whether to remove spikes to create smoother meshes. Default: True
    skin_tag: float (optional)
        1) Restrict effects of skin_facet_size to the outer boundary of the
           region with label skin_tag-1000 (i.e. 5 for 1005)
        2) Add outer surface to mesh using given tag. Set to None to disable.
        NOTE: This surface will replace any other surface with the same tag.
        Default: 1005
    hierarchy: list of ints or None (optional)
        Tags in order of priority for the label of the triangles between two
        regions. Default: (1, 2, 9, 3, 4, 8, 7, 6, 11, 10, 12, 5), i.e. WM (1)
        has highest priority, GM (2) comes next, etc.
    apply_cream: bool (optional)
        Whether to apply a cream layer around the head, which removes spikes on its outside
    smooth_steps: int (optional)
        Number of smoothing steps applied to the final mesh surfaces. Default: 5
    skin_care: int (optional)
        Number of additional smoothing steps applied to the skin. Default: 20
    mmg_noinsert : bool, optional, default: False
        Keep MMG from inserting points, so the number of elements does not grow (not recommended).
    debug : bool, optional, default: False
        Whether to save intermediate mesh results
    debug_path : string, optional, default: ""
        The path to use to save the debug output

    Returns
    -------
    msh: charm.mesh.msh.Msh
        Mesh structure
    """
    if hierarchy is None:
        hierarchy = (1, 2, 9, 3, 4, 8, 7, 6, 11, 10, 12, 5)
    if elem_sizes is None:
        elem_sizes = {"standard": {"range": [1, 5], "slope": 1.0}}
    if facet_distances is None:
        facet_distances = {"standard": {"range": [0.1, 3], "slope": 0.5}}
    if "standard" not in elem_sizes:
        raise ValueError('elem_sizes needs a "standard" entry')
    if "standard" not in facet_distances:
        raise ValueError('facet_distances needs a "standard" entry')

    if label_img.dtype not in [np.uint8, np.uint16]:
        raise MeshingError("Image must be of type uint8 or uint16")

    label_img = label_img.astype(np.uint16)

    if apply_cream:
        label_img, affine = transformations.pad_vol(label_img, affine, 30)

    logger.info("Calculating tissue thickness")
    start = time.time()
    thickness = calc_thickness(label_img)
    thickness[thickness < 0.5] = 100  # background
    voxel_size = get_vox_size(affine)
    if not np.allclose(np.diff(voxel_size), 0):
        logger.warning("Anisotropic image, meshing may contain extra artifacts")
    thickness *= np.average(voxel_size)

    logger.info("Calculating sizing fields")
    size_field = _sizing_field_from_thickness(label_img, thickness, elem_sizes)
    distance_field = _sizing_field_from_thickness(label_img, thickness, facet_distances)
    del thickness

    if apply_cream:
        logger.info("Applying cream mask")
        apply_cream_layer(label_img, size_field, distance_field, 20)

    # Smoothing reduces the effect of outliers in the thickness map, which show up as
    # localized small thickness values at some of the tissue boundaries
    if smooth_size_field:
        size_field = size_field ** (1 / 3)  # compress high values to preserve edges better
        kernel = smooth_size_field + 1 - np.abs(np.arange(-smooth_size_field, smooth_size_field + 1, 1))
        kernel = kernel / np.sum(kernel)
        for i in range(3):
            size_field = scipy.ndimage.convolve1d(size_field, kernel, axis=i, mode="constant", cval=0.0, origin=0)
        size_field = size_field**3

    # Limit the triangle size of the outer surface so that electrode meshing works
    if skin_facet_size is not None:
        boundary = ((label_img > 0) & (label_img != ElementTags.CREAM)).astype("int8")
        boundary = boundary - erosion(boundary, 1)
        if skin_tag is not None:
            # Only the boundary of label skin_tag-1000, to save tetrahedra
            skin_tet_tag = skin_tag - ElementTags.TH_SURFACE_START
            boundary *= label_img == skin_tet_tag
            boundary = dilate(boundary, 1)
            boundary *= label_img == skin_tet_tag
        else:
            boundary = dilate(boundary, 1)
        size_field[boundary] = skin_facet_size
        size_field[label_img == ElementTags.CREAM] = skin_facet_size
        del boundary

    logger.info("Time to prepare meshing: " + format_time(time.time() - start))

    if debug:
        tmp_nii = nib.Nifti1Image(size_field, affine)
        nifti.save(tmp_nii, os.path.join(debug_path, "size_field.nii.gz"))
        tmp_nii = nib.Nifti1Image(distance_field, affine)
        nifti.save(tmp_nii, os.path.join(debug_path, "distance_field.nii.gz"))
        tmp_nii = nib.Nifti1Image(label_img, affine)
        nifti.save(tmp_nii, os.path.join(debug_path, "label_img.nii.gz"))
        del tmp_nii

    start = time.time()
    m = image2mesh(
        label_img,
        affine,
        facet_size=size_field,
        facet_distance=distance_field,
        cell_size=size_field,
        do_perturb=optimize,
        do_exude=optimize,
        do_lloyd=optimize,
    )

    del size_field
    del distance_field

    logger.info("Time to mesh: " + format_time(time.time() - start))

    # The surfaces are reconstructed later
    start = time.time()
    m = m.crop_mesh(elm_type=4)

    m = _fix_labels(m, label_img)
    m = _relabel_microtets(m)

    if debug:
        write_msh(m, os.path.join(debug_path, "before_despike.msh"))

    if remove_spikes:
        m = _remove_spikes(m, label_img, affine, label_GM=2, label_CSF=3)

    if apply_cream:
        logger.info("Remove cream mask")
        tags_keep = np.setdiff1d(np.unique(m.elm.tag1), ElementTags.CREAM)
        m = m.crop_mesh(tags=tags_keep)

    idx = m.elm.connected_components()
    m = m.crop_mesh(elements=max(idx, key=np.size))

    logger.info("Reconstructing Surfaces")
    m.fix_th_node_ordering()
    m.reconstruct_unique_surface(hierarchy=hierarchy, add_outer_as=skin_tag)

    if debug:
        write_msh(m, os.path.join(debug_path, "before_smooth.msh"))

    m = _run_mmg(m, 1, mmg_noinsert)

    if smooth_steps > 0:
        logger.info("Smoothing Mesh Surfaces")
        m.smooth_surfaces(smooth_steps, step_size=0.3, max_gamma=10)

    if skin_care > 0:
        logger.info("Extra Skin Care")
        m.smooth_surfaces(skin_care, step_size=0.3, tags=skin_tag, max_gamma=10)

    if debug:
        write_msh(m, os.path.join(debug_path, "before_mmg.msh"))

    m = _run_mmg(m, 2, mmg_noinsert)

    logger.info("Time to post-process mesh: " + format_time(time.time() - start))
    return m


def apply_cream_layer(label_img, size_field, distance_field, cream_thickness):
    # Each layer only depends on the one before it, so only those voxels need looking at. A voxel on the
    # image boundary is its own neighbor outside of the image, as in grey_dilation's reflective boundary.
    shape = np.array(label_img.shape)
    layer = np.nonzero(dilate_cross(label_img > 0) & (label_img == 0))
    for i in range(cream_thickness):
        coords = np.array(layer)
        if coords.shape[1] == 0:
            break
        neighbors = np.repeat(coords[:, None], 6, axis=1)
        for axis in range(3):
            neighbors[axis, 2 * axis] = np.maximum(coords[axis] - 1, 0)
            neighbors[axis, 2 * axis + 1] = np.minimum(coords[axis] + 1, shape[axis] - 1)
        neighbor_labels = label_img[tuple(neighbors)]
        is_assigned = neighbor_labels > 0

        for field in (size_field, distance_field):
            values = np.where(is_assigned, field[tuple(neighbors)], -1)
            field[layer] = values.max(axis=0)
        label_img[layer] = ElementTags.CREAM

        # The next layer is what's still background around this one
        is_next = ~is_assigned
        candidates = neighbors[:, is_next]
        candidates = candidates[:, label_img[tuple(candidates)] == 0]
        layer = np.unravel_index(
            np.unique(np.ravel_multi_index(tuple(candidates), label_img.shape)),
            label_img.shape,
        )
