import numpy as np
from scipy.cluster.vq import kmeans2

from charm.logging import logger
from charm.segmentation import gems
from charm.segmentation.io import read_mesh_collection
from charm.segmentation.warp_mesh import warp_mesh


def subject_to_template(positions, transform):
    matrix = transform.as_numpy_array
    return np.linalg.solve(matrix[:3, :3], (positions - matrix[:3, 3]).T).T


def template_to_subject(positions, transform):
    matrix = transform.as_numpy_array
    return positions @ matrix[:3, :3].T + matrix[:3, 3]


class ProbabilisticAtlas:
    def __init__(self):
        self._previous_deformation_mesh = None

    def get_mesh(
        self,
        mesh_collection_file_name,
        transform=None,
        k=None,
        initial_deformation=None,
        initial_deformation_file_name=None,
        return_initial_deformation=False,
    ):
        collection = read_mesh_collection(mesh_collection_file_name)
        if k is not None:
            collection.k = k

        if transform:
            collection.transform(transform)
        else:
            transform = gems.KvlTransform(np.eye(4))
        mesh = collection.reference_mesh

        deformation_in_template_space = None
        if initial_deformation is not None:
            # Warp in template space, undoing the affine registration
            deformation_in_template_space = warp_mesh(
                initial_deformation_file_name or mesh_collection_file_name,
                initial_deformation,
                mesh_collection_file_name,
            )
            mesh.points = template_to_subject(
                subject_to_template(mesh.points, transform) + deformation_in_template_space,
                transform,
            )

        if return_initial_deformation:
            if deformation_in_template_space is None:
                deformation_in_template_space = np.zeros_like(mesh.points)
            return mesh, deformation_in_template_space
        return mesh

    def deform_mesh(self, mesh, transform, class_likelihoods, voxel_selection, user_parameters=None):
        calculator = gems.KvlCostAndGradientCalculator(
            type_name="AtlasMeshToIntensityImage",
            images=[],
            boundary_condition="Sliding",
            transform=transform,
            likelihoods=class_likelihoods,
            voxel_selection=voxel_selection,
        )

        parameters = {
            "Verbose": False,
            "MaximalDeformationStopCriterion": 0.001,  # in voxels
            "LineSearchMaximalDeformationIntervalStopCriterion": 0.001,
            "MaximumNumberOfIterations": 20,
            "BFGS-MaximumMemoryLength": 12,
        }
        parameters.update(user_parameters or {})

        # Block coordinate descent: M problems of size N/M solve faster than one of size N. The blocks are only
        # loosely coupled, through the layer of tetrahedra connecting them.
        number_of_blocks = 6
        number_of_passes = 3  # Between 1 and parameters[ 'MaximumNumberOfIterations' ]

        if mesh != self._previous_deformation_mesh:
            # k-means keeps blocks compact, which limits the extra points added for boundary tetrahedra
            _, labels = kmeans2(mesh.points, number_of_blocks, minit="++", seed=0, missing="raise")
            self._masks, self._submeshes, self._optimizers = [], [], []
            for block in range(number_of_blocks):
                mask = labels == block
                submesh = mesh.get_submesh(mask)
                self._masks.append(mask)
                self._submeshes.append(submesh)
                self._optimizers.append(gems.KvlOptimizer("L-BFGS", submesh, calculator, parameters))
            self._previous_deformation_mesh = mesh
        else:
            # Keep the warm optimizers and only swap in the current calculator
            for optimizer in self._optimizers:
                optimizer.update_calculator(calculator)

        positions_before = mesh.points
        inner_loops = round(parameters["MaximumNumberOfIterations"] / number_of_passes)
        positions = positions_before.copy()
        while True:
            maximal_deformation = 0.0
            for mask, submesh, optimizer in zip(self._masks, self._submeshes, self._optimizers):
                submesh.points = positions[mask, :]
                optimizer.update_mesh(submesh)

                block_before = submesh.points
                for _ in range(inner_loops):
                    optimizer.step_optimizer()
                block_after = submesh.points
                positions[mask, :] = block_after

                maximal_deformation = max(
                    maximal_deformation, np.sqrt(np.max(np.sum((block_after - block_before) ** 2, 1)))
                )

            logger.debug(f"maximalDeformation={maximal_deformation:.4f}")
            if maximal_deformation == 0:
                break

        mesh.points = positions
        cost, _ = calculator.evaluate_mesh_position(mesh)

        maximal_deformation_applied = np.sqrt(np.max(np.sum((positions - positions_before) ** 2, 1)))
        return maximal_deformation_applied, cost
