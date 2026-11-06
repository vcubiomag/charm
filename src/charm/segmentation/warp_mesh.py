import numpy as np
import scipy.ndimage

from charm.segmentation.gems import (
    KvlCostAndGradientCalculator,
    KvlOptimizer,
    KvlTransform,
)
from charm.segmentation.io import read_mesh_collection


def _rasterize_deformation(mesh, deformation, image_size):
    # An offset makes the deformation positive everywhere, which leaves zero to mean a voxel was not visited
    offset = 1.0 - np.min(deformation)
    dense = mesh.rasterize_values(image_size, np.ascontiguousarray(deformation + offset, dtype=np.float32))

    # Due to tetrahedral inside/outside checking performed in the rasterizer, voxels on the far boundary of the image
    # grid are never visited. They get the value of the neighboring voxel inside of the grid.
    for axis in range(3):
        last = dense[(slice(None),) * axis + (-1,)]
        second_to_last = dense[(slice(None),) * axis + (-2,)]
        unvisited = last == 0
        last[unvisited] = second_to_last[unvisited]

    # Anything else should be rare, and gets the value of the closest voxel that has been visited
    unvisited = dense[..., 0] == 0
    if unvisited.any():
        closest = scipy.ndimage.distance_transform_edt(unvisited, return_indices=True, return_distances=False)
        dense[unvisited] = dense[tuple(indices[unvisited] for indices in closest)]
    dense -= offset

    # Sliding boundary conditions make the normal deformation component zero on the image grid boundaries
    dense[0, :, :, 0] = 0
    dense[-1, :, :, 0] = 0
    dense[:, 0, :, 1] = 0
    dense[:, -1, :, 1] = 0
    dense[:, :, 0, 2] = 0
    dense[:, :, -1, 2] = 0
    return dense


def warp_mesh(source_file_name, source_deformation, target_file_name):
    """The deformation of the reference node positions of the target mesh collection that
    mimics the dense 3D deformation given by source_deformation on the source collection."""
    source_mesh = read_mesh_collection(source_file_name).reference_mesh
    source_positions = source_mesh.points
    target_collection = read_mesh_collection(target_file_name)
    # A low stiffness lets the target mesh follow the desired positions
    target_collection.k = 0.001
    target_mesh = target_collection.reference_mesh
    target_positions = target_mesh.points

    # Identical mesh connectivity needs no optimization
    if target_positions.shape[0] == source_positions.shape[0]:
        return source_positions + source_deformation - target_positions

    image_size = [int(1 + dim) for dim in np.max(source_positions, axis=0)]
    dense = _rasterize_deformation(source_mesh, source_deformation, image_size)

    desired_positions = target_positions + np.stack(
        [
            scipy.ndimage.map_coordinates(
                dense[:, :, :, axis], target_positions.T, mode="nearest", order=1, output=np.double
            )
            for axis in range(3)
        ],
        axis=1,
    )

    calculator = KvlCostAndGradientCalculator(
        "PointSet", [], "Sliding", KvlTransform(np.eye(4)), target_points=desired_positions
    )
    stop_criterion = 0.005
    optimizer = KvlOptimizer(
        "L-BFGS",
        target_mesh,
        calculator,
        {
            "Verbose": 0.0,
            "MaximalDeformationStopCriterion": stop_criterion,
            "LineSearchMaximalDeformationIntervalStopCriterion": stop_criterion,
            "BFGS-MaximumMemoryLength": 12,
        },
    )
    while optimizer.step_optimizer()[1] != 0:
        pass
    return target_mesh.points - target_positions
