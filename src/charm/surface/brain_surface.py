import numpy as np

from charm.logging import logger
from charm.mesh.msh import Elements, Msh, Nodes


def segment_triangle_intersect(vertices, faces, segment_start, segment_end):
    """Intersect line segments with a triangulated surface.

    Parameters
    -----------
    vertices: ndarray
        Vertex positions.
    faces: ndarray
        0-indexed triangles.
    segment_start: ndarray
        N_lines x 3 array with the start of the line segments
    segment_end: ndarray
        N_lines x 3 array with the end of the line segments

    Returns
    --------
    indices_pairs: ndarray
        Nx2 array of ints with the pair (segment index, face index) for each intersection
    positions: ndarray
        Nx3 array of floats with the position of the intersections
    """
    m = Msh(nodes=Nodes(vertices), elements=Elements(faces + 1))
    indices_pairs, positions = m.intersect_segment(segment_start, segment_end)
    # Msh elements are 1-indexed.
    indices_pairs[:, 1] -= 1
    return indices_pairs, positions


def _rasterize_surface(vertices, faces, affine, shape, axis="z"):
    inv_affine = np.linalg.inv(affine)
    vertices_trafo = inv_affine[:3, :3].dot(vertices.T).T + inv_affine[:3, 3].T

    # Permute the axes so the rays run along the last one.
    if axis == "z":
        out_shape = shape
    elif axis == "y":
        vertices_trafo = vertices_trafo[:, [0, 2, 1]]
        out_shape = np.array(shape, dtype=int)[[0, 2, 1]]
    elif axis == "x":
        vertices_trafo = vertices_trafo[:, [2, 1, 0]]
        out_shape = np.array(shape, dtype=int)[[2, 1, 0]]
    else:
        raise ValueError('"axis" should be x, y, or z')

    grid_points = np.array(np.meshgrid(*[np.arange(n) for n in out_shape[:2]], indexing="ij")).reshape((2, -1)).T
    grid_points_near = np.hstack([grid_points, np.zeros((len(grid_points), 1))])
    grid_points_far = np.hstack([grid_points, out_shape[2] * np.ones((len(grid_points), 1))])

    # Extend the rays when the mesh reaches past the volume, so they still cross
    # the whole mesh.
    if np.min(vertices_trafo[:, 2]) < 0:
        grid_points_near[:, 2] = 1.1 * np.min(vertices_trafo[:, 2])
    if np.max(vertices_trafo[:, 2]) > out_shape[2]:
        grid_points_far[:, 2] = 1.1 * np.max(vertices_trafo[:, 2])

    pairs, positions = segment_triangle_intersect(vertices_trafo, faces, grid_points_near, grid_points_far)

    inter_z = np.clip((positions[:, 2] + 1).astype(int), 0, out_shape[2])

    # Sorted crossings of each line, of which pairs enter and leave the surface. The count
    # should never be odd, and if it is the last crossing goes without a pair.
    order = np.lexsort((inter_z, pairs[:, 0]))
    lines = pairs[order, 0]
    inter_z = inter_z[order]
    _lines_intersecting, first, counts = np.unique(lines, return_index=True, return_counts=True)
    if np.any(counts % 2 == 1):
        logger.warning("Found an odd number of crossings! This could be an open surface or a self-intersection")
    position = np.arange(len(lines)) - np.repeat(first, counts)
    n_of_line = np.repeat(counts, counts)
    enters = (position % 2 == 0) & (position + 1 < n_of_line)
    x, y = grid_points[lines[enters]].T

    # Writing where segments start and end turns the mask into a running sum along z
    mask = np.zeros((*out_shape[:2], out_shape[2] + 1), dtype=np.int8)
    np.add.at(mask, (x, y, inter_z[enters]), 1)
    np.subtract.at(mask, (x, y, inter_z[np.flatnonzero(enters) + 1]), 1)
    mask = np.cumsum(mask, axis=2, dtype=np.int8)[:, :, :-1] > 0

    if axis == "z":
        pass
    elif axis == "y":
        mask = np.swapaxes(mask, 2, 1)
    elif axis == "x":
        mask = np.swapaxes(mask, 2, 0)

    return mask


def mask_from_surface(vertices, faces, affine, shape):
    """Create a binary mask of the voxels inside a closed surface.

    Parameters
    ----------
    vertices: ndarray
        Vertex positions in world coordinates.
    faces: ndarray
        0-indexed triangles.
    affine: 4x4 ndarray
        Voxel to world transformation.
    shape: 3x1 list
        Shape of the output mask.

    Returns
    ----------
    mask : ndarray of shape 'shape'
    """

    masks = []

    if len(vertices) == 0 or len(faces) == 0:
        logger.warning("Surface if empty! Return empty volume")
        return np.zeros(shape, dtype=bool)

    for axis in ["x", "y", "z"]:
        masks.append(_rasterize_surface(vertices, faces, affine, shape, axis=axis))

    # Rasterize along all three axes and keep voxels found by at least two, which
    # suppresses spurious voxels caused by bad topology.
    return np.sum(masks, axis=0, dtype=np.uint8) >= 2
