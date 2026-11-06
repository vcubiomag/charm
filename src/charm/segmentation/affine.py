import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

import nibabel as nib
import numpy as np
import scipy.io
import scipy.ndimage
from scipy.optimize import minimize_scalar

from charm.logging import logger
from charm.segmentation import gems
from charm.segmentation.io import read_mesh_collection, write_mesh_collection
from charm.segmentation.probabilistic_atlas import (
    ProbabilisticAtlas,
    subject_to_template,
    template_to_subject,
)
from charm.segmentation.segmentation_utils import read_cropped_images
from charm.utils import nifti


@dataclass
class InitializationOptions:
    pitch_angles: Sequence[float] = (0.0,)  # in radians
    scales: Sequence[Sequence[float]] = ((1.0, 1.0, 1.0),)
    horizontal_table_shifts: Sequence[float] = (20.0, 10.0, 0.0, -10.0, -20.0)  # in mm, in template space
    vertical_table_shifts: Sequence[float] = (-10.0, 0.0, 10.0)  # in mm, in template space
    try_center_of_gravity: bool = True
    search_for_table_shifts_separately: bool = False
    pitch_center: Sequence[float] = (0.0, 0.0, 0.0)  # anterior commissure, in mm in template space
    scaling_center: Sequence[float] = (0.0, -120.0, 0.0)  # back of the head, in mm in template space
    initial_pitch_angle: float = 0.0  # in radians
    initial_scale: float = 1.0
    initial_table_shift: Sequence[float] = (0.0, 0.0, 0.0)  # in mm, in template space


class GridSearchResult(NamedTuple):
    pitch_angle: float
    scale: Sequence[float]
    horizontal_table_shift: float
    vertical_table_shift: float
    initial_table_shift: Sequence[float]
    image_to_image_transform: np.ndarray


class AffineRegistration:
    def __init__(self, image_file_name, mesh_collection_file_name, template_file_name):
        self.image_file_name = image_file_name
        self.mesh_collection_file_name = mesh_collection_file_name
        self.template_file_name = template_file_name
        self.options = InitializationOptions()

    def set_up(self, init_transform, target_downsampled_voxel_spacing):
        image = gems.KvlImage(self.image_file_name)
        image_to_world = image.transform_matrix.as_numpy_array
        template = gems.KvlImage(self.template_file_name)

        # Initializations have physical meaning in template space, so move the subject image instead of the template
        self.original_image_to_world = image_to_world
        if init_transform is not None:
            # Assume the initialization matrix is LPS2LPS
            logger.debug("initializing with predefined transform")
            image_to_world = np.linalg.solve(init_transform, image_to_world)

        voxel_spacing = np.linalg.norm(image_to_world[0:3, 0:3], axis=0)
        factors = np.maximum(np.round(target_downsampled_voxel_spacing / voxel_spacing), 1)
        self.upsampling_transform = np.diag(np.append(factors, 1))

        buffer = image.get_image_buffer()
        self.image = gems.KvlImage(buffer[:: int(factors[0]), :: int(factors[1]), :: int(factors[2])])
        self.image_to_world = image_to_world @ self.upsampling_transform
        self.template_to_world = template.transform_matrix.as_numpy_array
        self.expected_number_of_voxels_covered = np.prod(template.get_image_buffer().shape)

    def transform_matrix(
        self,
        pitch_angle=0.0,
        scale=(1.0, 1.0, 1.0),
        horizontal_table_shift=0.0,
        vertical_table_shift=0.0,
        initial_table_shift=None,
    ):
        # Simulates a table shift of the MRI scanner, then scaling around scaling_center, then pitch around
        # pitch_center, all in template world coordinates (mm)
        options = self.options
        if initial_table_shift is None:
            initial_table_shift = options.initial_table_shift

        translation = np.eye(4)
        translation[0:3, 3] = initial_table_shift
        translation[1, 3] += vertical_table_shift
        translation[2, 3] += horizontal_table_shift

        def around(center, matrix):
            centering = np.eye(4)
            centering[0:3, 3] = -center[0:3]
            return np.linalg.solve(centering, matrix @ centering)

        scaling = np.diag(np.append(np.asarray(scale) * options.initial_scale, 1.0))
        scaling = around(translation @ np.append(options.scaling_center, 1.0), scaling)

        angle = pitch_angle + options.initial_pitch_angle
        rotation = np.eye(4)
        rotation[1, 1] = np.cos(angle)
        rotation[2, 1] = np.sin(angle)
        rotation[1, 2] = -np.sin(angle)
        rotation[2, 2] = np.cos(angle)
        rotation = around(scaling @ translation @ np.append(options.pitch_center, 1.0), rotation)

        return rotation @ scaling @ translation

    def grid_search(
        self,
        mesh,
        positions_in_template_space,
        pitch_angles=(0.0,),
        scales=((1.0, 1.0, 1.0),),
        horizontal_table_shifts=(0.0,),
        vertical_table_shifts=(0.0,),
        initial_table_shifts=((0.0, 0.0, 0.0),),
    ):
        calculator = gems.KvlCostAndGradientCalculator("MutualInformation", [self.image], "Affine")

        best_cost = np.inf
        best = None
        for initial_table_shift in initial_table_shifts:
            for scale in scales:
                for horizontal_table_shift in horizontal_table_shifts:
                    for vertical_table_shift in vertical_table_shifts:
                        for pitch_angle in pitch_angles:
                            world_to_world = self.transform_matrix(
                                pitch_angle, scale, horizontal_table_shift, vertical_table_shift, initial_table_shift
                            )
                            image_to_image = np.linalg.solve(
                                self.image_to_world, world_to_world @ self.template_to_world
                            )

                            mesh.points = template_to_subject(
                                positions_in_template_space, gems.KvlTransform(image_to_image)
                            )
                            cost = calculator.evaluate_mesh_cost(mesh)
                            if cost < best_cost:
                                best_cost = cost
                                best = GridSearchResult(
                                    pitch_angle,
                                    scale,
                                    horizontal_table_shift,
                                    vertical_table_shift,
                                    initial_table_shift,
                                    image_to_image,
                                )

        if best is None:
            raise RuntimeError("no initialization of the affine registration has a finite cost")
        # Leave the mesh at the winner
        mesh.points = template_to_subject(positions_in_template_space, gems.KvlTransform(best.image_to_image_transform))
        return best

    def register(
        self,
        world_to_world=None,
        init_transform=None,
        ks=(20.0, 10.0, 5.0),
        options=None,
        target_downsampled_voxel_spacing=3.0,
        maximal_deformation_stop_criterion=0.005,
        noneck=False,
    ):
        """Returns the image-to-image and world-to-world transforms of the template to the image,
        and a summary of the optimization which is None if world_to_world was given."""
        self.set_up(init_transform, target_downsampled_voxel_spacing)
        self.options = options if options is not None else InitializationOptions()

        if world_to_world is not None:
            logger.debug("world-to-world transform supplied - skipping registration")
            image_to_image = np.linalg.inv(self.original_image_to_world) @ world_to_world @ self.template_to_world
            return image_to_image, world_to_world, None

        logger.debug(f"affine atlas registration of {self.template_file_name} to {self.image_file_name}")

        image_to_image = self._initialization(noneck)

        for k in ks:
            image_to_image, summary = self._optimize(image_to_image, k, maximal_deformation_stop_criterion, noneck)

        image_to_image = self.upsampling_transform @ image_to_image
        world_to_world = self.original_image_to_world @ image_to_image @ np.linalg.inv(self.template_to_world)
        return image_to_image, world_to_world, summary

    def _initialization(self, noneck):
        options = self.options

        # Even with K=0, put the reference mesh in subject space first. A tetrahedron with negative volume triggers
        # a sentinel infinite cost regardless of K.
        image_to_image = np.linalg.inv(self.image_to_world) @ self.template_to_world
        transform = gems.KvlTransform(image_to_image)
        mesh = ProbabilisticAtlas().get_mesh(self.mesh_collection_file_name, transform=transform, k=0.0)
        positions_in_template_space = subject_to_template(mesh.points, transform)

        best_initial_table_shift = options.initial_table_shift
        if options.try_center_of_gravity:
            mesh.points = positions_in_template_space
            template_size = np.round(np.max(mesh.points, axis=0) + 1).astype(int)
            priors = mesh.rasterize(template_size, -1)
            head = np.sum(priors[:, :, :, 2 if noneck else 1 :], axis=3)

            template_center = np.array(scipy.ndimage.center_of_mass(head))  # in image space
            image_center = np.array(scipy.ndimage.center_of_mass(self.image.get_image_buffer()))  # in image space
            to_world = self.transform_matrix() @ self.template_to_world
            template_center = to_world[0:3, 0:3] @ template_center + to_world[0:3, 3]  # in world space
            image_center = self.image_to_world[0:3, 0:3] @ image_center + self.image_to_world[0:3, 3]  # in world space

            best_initial_table_shift = self.grid_search(
                mesh,
                positions_in_template_space,
                initial_table_shifts=[options.initial_table_shift, image_center - template_center],
            ).initial_table_shift

        # Drop the background and neck alphas so they don't skew the scaling, rotation and shifts
        alphas = mesh.alphas
        mesh.alphas = alphas[:, 2:]
        horizontal_table_shifts = options.horizontal_table_shifts
        vertical_table_shifts = options.vertical_table_shifts
        if options.search_for_table_shifts_separately:
            # Search translation apart from scaling and rotation to limit the number of combinations
            best = self.grid_search(
                mesh,
                positions_in_template_space,
                horizontal_table_shifts=horizontal_table_shifts,
                vertical_table_shifts=vertical_table_shifts,
                initial_table_shifts=[best_initial_table_shift],
            )
            horizontal_table_shifts = [best.horizontal_table_shift]
            vertical_table_shifts = [best.vertical_table_shift]

        best = self.grid_search(
            mesh,
            positions_in_template_space,
            pitch_angles=options.pitch_angles,
            scales=options.scales,
            horizontal_table_shifts=horizontal_table_shifts,
            vertical_table_shifts=vertical_table_shifts,
            initial_table_shifts=[best_initial_table_shift],
        )
        mesh.alphas = alphas
        return best.image_to_image_transform

    def _optimize(self, initial_image_to_image, k, maximal_deformation_stop_criterion, noneck):
        # Mesh stiffness is relative to a log-likelihood that scales with the number of voxels. MI is normalized,
        # so divide the stiffness by the expected number of voxels covered.
        mesh = ProbabilisticAtlas().get_mesh(
            self.mesh_collection_file_name,
            transform=gems.KvlTransform(initial_image_to_image),
            k=k / self.expected_number_of_voxels_covered,
        )
        original_positions = mesh.points

        if noneck:
            alphas = mesh.alphas
            mesh.alphas = np.column_stack((alphas[:, 0] + alphas[:, 1], alphas[:, 2:]))

        calculator = gems.KvlCostAndGradientCalculator("MutualInformation", [self.image], "Affine")
        optimizer = gems.KvlOptimizer(
            "L-BFGS",
            mesh,
            calculator,
            {
                "Verbose": 0,
                "MaximalDeformationStopCriterion": maximal_deformation_stop_criterion,
                "LineSearchMaximalDeformationIntervalStopCriterion": maximal_deformation_stop_criterion,
                "BFGS-MaximumMemoryLength": 12.0,  # Affine registration only has 12 DOF
            },
        )

        number_of_iterations = 0
        while True:
            cost, maximal_deformation = optimizer.step_optimizer()
            logger.debug(f"maximalDeformation={maximal_deformation:.4f} minLogLikelihood={cost:.4f}")
            number_of_iterations += 1
            if maximal_deformation == 0:
                break

        # Four nodes are enough to recover an affine transform
        nodes = [0, 110, 201, 302]
        before = np.vstack((original_positions[nodes].T, np.ones(4)))
        after = np.vstack((mesh.points[nodes].T, np.ones(4)))
        extra_image_to_image = after @ np.linalg.inv(before)
        logger.debug(f"appliedScaling: {np.linalg.det(extra_image_to_image) ** (1 / 3):0.4f}")
        summary = {"numberOfIterations": number_of_iterations, "cost": cost}
        return extra_image_to_image @ initial_image_to_image, summary

    def save_results(
        self, image_name, template_name, save_path, template_coregistered_name, image_to_image, world_to_world
    ):
        # The saved world-to-world matrix is in RAS, gems works in LPS
        ras_to_lps = np.diag([-1, -1, 1, 1])
        scipy.io.savemat(
            os.path.join(save_path, "coregistrationMatrices.mat"),
            {
                "imageToImageTransformMatrix": image_to_image,
                "worldToWorldTransformMatrix": ras_to_lps @ world_to_world @ ras_to_lps,
            },
        )

        # The coregistered template has the estimated transform applied for historical reasons
        image_to_world = gems.KvlImage(image_name).transform_matrix.as_numpy_array
        gems.KvlImage(template_name).write(
            template_coregistered_name, gems.KvlTransform(image_to_world @ image_to_image)
        )

    def adjust_neck(
        self,
        t1,
        transformed_template_file_name,
        mesh_level1,
        mesh_level2,
        neck_bounds,
        neck_tissues,
        downsampling_target=3.0,
    ):
        """Bend the neck of the atlas to fit the image, writing the meshes next to the
        coregistered template. Returns whether the orientation of the image allowed for it."""
        orientation = nib.aff2axcodes(nifti.load(t1).affine)
        if "S" in orientation:
            z_dim = orientation.index("S")
            up = True
        elif "I" in orientation:
            z_dim = orientation.index("I")
            up = False
        else:
            logger.info("Can't figure out orientation. Skipping neck adjustment.")
            return False
        if "A" in orientation:
            x_dim = orientation.index("A")
        elif "P" in orientation:
            x_dim = orientation.index("P")
        else:
            logger.info("Can't figure out orientation. Skipping neck adjustment.")
            return False

        image_buffer, to_image, voxel_spacing, _ = read_cropped_images([t1], transformed_template_file_name)
        factors = np.maximum(np.round(downsampling_target / voxel_spacing), 1)
        image_buffer = image_buffer[:: int(factors[0]), :: int(factors[1]), :: int(factors[2])]

        # A tiny K lets the intensity cost dominate
        k = 1e-20
        collection = read_mesh_collection(mesh_level2)
        collection.k = k * np.prod(factors)
        collection.transform(to_image)
        mesh = collection.reference_mesh
        mesh.scale(1 / factors)

        alphas = mesh.alphas
        positions = mesh.points
        deformation = _neck_deformation(positions, alphas, neck_tissues, z_dim, up, x_dim)

        # Only internal air, spine, cortical bone and spongy bone enter the cost
        mesh.alphas = alphas[:, neck_tissues]

        calculator = gems.KvlCostAndGradientCalculator("MutualInformation", [gems.KvlImage(image_buffer)], "Affine")
        logger.info(f"Initial cost: {calculator.evaluate_mesh_position(mesh)[0]}")

        def cost(x):
            mesh.points = positions - x * deformation
            return calculator.evaluate_mesh_position(mesh)[0]

        best = minimize_scalar(cost, None, neck_bounds, method="Bounded").x
        mesh.points = positions - best * deformation

        mesh.scale(factors)
        mesh.alphas = alphas
        from_image = gems.KvlTransform(np.linalg.inv(to_image.as_numpy_array))
        collection.transform(from_image)
        directory = os.path.dirname(transformed_template_file_name)
        write_mesh_collection(collection, os.path.join(directory, "atlas_level2.h5"))

        collection = read_mesh_collection(mesh_level1)
        collection.k = k
        collection.transform(to_image)
        mesh = collection.reference_mesh
        positions = mesh.points
        mesh.points = positions - best * _neck_deformation(positions, mesh.alphas, neck_tissues, z_dim, up, x_dim)
        collection.transform(from_image)
        write_mesh_collection(collection, os.path.join(directory, "atlas_level1.h5"))
        return True


def _neck_deformation(positions, alphas, neck_tissues, z_dim, up, x_dim):
    # The distance from the top of the spine defines the amount of deformation in the A->P direction.
    # The mesh nodes are in voxels of the image, with values stored from the bottom of the head to the top.
    z = positions[:, z_dim]
    spine_z = z[alphas[:, neck_tissues[2]] > 0.1]
    if up:
        top = spine_z.max()
        is_neck = z < top
        distance = top - z[is_neck]
    else:
        top = spine_z.min()
        is_neck = z > top
        distance = z[is_neck] - top
    deformation = np.zeros_like(positions)
    deformation[is_neck, x_dim] = distance
    return deformation
