import os
import pickle
from dataclasses import dataclass, field, replace

import numpy as np

from charm.logging import logger
from charm.segmentation import gems
from charm.segmentation.bias_field import BiasField
from charm.segmentation.gmm import GMM
from charm.segmentation.io import read_compression_lookup_table, read_shared_gmm_parameters
from charm.segmentation.merge_alphas import merge_alphas, merging_fractions_table
from charm.segmentation.probabilistic_atlas import (
    ProbabilisticAtlas,
    subject_to_template,
)
from charm.segmentation.segmentation_utils import (
    log_transform,
    mask_out_background,
    read_cropped_images,
)

eps = np.finfo(float).eps


@dataclass
class ModelSpecification:
    atlas_file_name: str
    free_surfer_labels: list[int]
    names: list[str]
    shared_gmm_parameters: list
    use_diagonal_covariances: bool = True
    brain_masking_smoothing_sigma: float = 3.0  # sqrt of the variance of a Gaussian blurring kernel
    brain_masking_threshold: float = 0.01
    k: float = 0.1  # stiffness of the mesh
    bias_field_smoothing_kernel_size: float = 50  # distance in mm of sinc function center to first zero crossing

    @classmethod
    def from_atlas(cls, atlas_dir, **overrides):
        labels, names = read_compression_lookup_table(os.path.join(atlas_dir, "compressionLookupTable.txt"))
        defaults = {"atlas_file_name": os.path.join(atlas_dir, "atlas_level2.h5")}
        if "shared_gmm_parameters" not in overrides:
            defaults["shared_gmm_parameters"] = read_shared_gmm_parameters(
                os.path.join(atlas_dir, "sharedGMMParameters.txt")
            )
        return cls(free_surfer_labels=labels, names=names, **defaults | overrides)


@dataclass
class ResolutionLevel:
    atlas_file_name: str
    target_downsampled_voxel_spacing: float
    maximum_number_of_iterations: int = 100
    estimate_bias_field: bool = True


@dataclass
class OptimizationOptions:
    levels: list[ResolutionLevel]
    maximum_number_of_deformation_iterations: int = 20
    absolute_cost_per_voxel_decrease_stop_criterion: float = 1e-4
    maximal_deformation_stop_criterion: float = 0.001  # measured in pixels
    line_search_maximal_deformation_interval_stop_criterion: float = 0.001
    bfgs_maximum_memory_length: int = 12


@dataclass
class FittedModel:
    means: np.ndarray
    variances: np.ndarray
    mixture_weights: np.ndarray
    gaussians_per_class: list[int]
    fractions_table: np.ndarray
    deformation: np.ndarray
    deformation_atlas: str
    model_specification: ModelSpecification
    image_file_names: list[str]
    mask: np.ndarray
    cropping: tuple
    transform: np.ndarray
    image_buffers: np.ndarray | None = field(default=None, repr=False)
    bias_fields: np.ndarray | None = field(default=None, repr=False)


class Samseg:
    # Set by preprocess and fit_model
    bias_field: BiasField
    gmm: GMM
    image_buffers: np.ndarray
    mask: np.ndarray
    class_fractions: np.ndarray
    cropping: tuple
    transform: gems.KvlTransform
    voxel_spacing: np.ndarray

    def __init__(
        self,
        image_file_names,
        save_path,
        transformed_template_file_name,
        model_specification,
        optimization_options,
    ):
        self.image_file_names = image_file_names
        self.transformed_template_file_name = transformed_template_file_name
        self.model_specification = model_specification
        self.optimization_options = optimization_options
        self.atlas = ProbabilisticAtlas()

        logger.debug(f"output directory: {save_path}")
        logger.debug(f"input images: {image_file_names}")
        logger.debug(f"transformed template: {transformed_template_file_name}")
        logger.debug(f"model specification: {model_specification}")
        logger.debug(f"optimization options: {optimization_options}")
        os.makedirs(save_path, exist_ok=True)

        self.optimization_summary: list[dict] = []
        self.deformation: np.ndarray | None = None
        self.deformation_atlas_file_name: str | None = None

    def preprocess(self):
        self.image_buffers, self.transform, self.voxel_spacing, self.cropping = read_cropped_images(
            self.image_file_names, self.transformed_template_file_name
        )

        # Zeroing intensities outside a rough brain mask makes every later computation skip them
        specification = self.model_specification
        self.image_buffers, self.mask = mask_out_background(
            self.image_buffers,
            self.image_file_names,
            specification.atlas_file_name,
            self.transform,
            specification.brain_masking_smoothing_sigma,
            specification.brain_masking_threshold,
            self.atlas,
            upsampled=False,
        )

        # The bias field model is additive, but MR bias is multiplicative, so work with log intensities
        self.image_buffers = log_transform(self.image_buffers, self.mask)

    def fit_model(self):
        self.bias_field = BiasField(
            self.image_buffers.shape[0:3],
            self.model_specification.bias_field_smoothing_kernel_size / self.voxel_spacing,
        )
        self._initialize_gmm()
        self._estimate_parameters()

    def _initialize_gmm(self):
        # Each structure is a mixture of classes (super-structures), and each class is a mixture of Gaussians.
        # class_fractions is a classes x structures table holding the mixing weights of the classes in each structure.
        specification = self.model_specification
        self.class_fractions = merging_fractions_table(specification.names, specification.shared_gmm_parameters)
        self.gmm = GMM(
            [parameter.number_of_components for parameter in specification.shared_gmm_parameters],
            number_of_contrasts=self.image_buffers.shape[-1],
            use_diagonal_covariances=specification.use_diagonal_covariances,
        )

    def _downsampled_model(self, atlas_file_name, downsampling_factors):
        step = tuple(slice(None, None, int(factor)) for factor in downsampling_factors)
        mask = self.mask[step]
        image_buffers = np.asfortranarray(self.image_buffers[step], dtype=float)

        transform = gems.KvlTransform(
            np.diag(np.append(1.0 / downsampling_factors, 1.0)) @ self.transform.as_numpy_array
        )
        mesh, initial_deformation_applied = self.atlas.get_mesh(
            atlas_file_name,
            transform,
            self.model_specification.k,
            self.deformation,
            self.deformation_atlas_file_name,
            return_initial_deformation=True,
        )
        return image_buffers, mask, mesh, initial_deformation_applied, transform

    def _estimate_parameters(self):
        options = self.optimization_options
        self.optimization_summary = []
        self.deformation, self.deformation_atlas_file_name = None, None
        self.bias_field.coefficients = None
        stop_criterion = options.absolute_cost_per_voxel_decrease_stop_criterion

        for level_number, level in enumerate(options.levels):
            logger.debug(f"multi-resolution level {level_number}")

            downsampling_factors = np.round(level.target_downsampled_voxel_spacing / self.voxel_spacing).astype(
                np.uint32
            )
            downsampling_factors[downsampling_factors < 1] = 1

            image_buffers, mask, mesh, initial_deformation_applied, transform = self._downsampled_model(
                level.atlas_file_name, downsampling_factors
            )
            self.bias_field.downsample_basis_functions(downsampling_factors)

            self.gmm.downsample_hyperparameters(downsampling_factors)

            initial_positions = subject_to_template(mesh.points, transform)

            mesh.alphas = merge_alphas(mesh.alphas, self.class_fractions)

            # Voxels are kept in F order, the order of the rasterized and bias field volumes. Indexing those with a
            # boolean mask would walk them in C order, against the memory layout.
            voxels = np.flatnonzero(mask.ravel(order="F"))
            number_of_voxels = voxels.size
            voxel_selection = gems.KvlVoxelSelection(mask.shape, voxels)
            masked_images = np.ascontiguousarray(image_buffers.reshape(-1, image_buffers.shape[-1], order="F")[voxels])
            del image_buffers
            class_likelihoods = None
            data = None

            history_of_cost = [1 / eps]
            for iteration in range(level.maximum_number_of_iterations):
                logger.debug(f"iteration {iteration}")

                class_priors = mesh.rasterize_priors(voxel_selection)

                if self.gmm.means is None:
                    self.gmm.initialize_parameters(masked_images, class_priors)

                if self.bias_field.coefficients is None:
                    number_of_functions = [functions.shape[1] for functions in self.bias_field.basis_functions]
                    self.bias_field.coefficients = np.zeros((np.prod(number_of_functions), masked_images.shape[-1]))

                # The posteriors stay with the priors and only the statistics come back, unless the bias field
                # update needs them
                update_bias_field = level.estimate_bias_field and iteration > 0
                previous_em_cost = 1 / eps
                em_iteration = 0
                while True:
                    data = self.bias_field.corrected_data(masked_images, voxels, data)

                    statistics, min_log_likelihood = self.gmm.compute_posteriors(data, class_priors, update_bias_field)

                    min_log_gmm_prior = self.gmm.min_log_prior()
                    em_cost = min_log_likelihood + min_log_gmm_prior
                    em_iteration += 1
                    if em_iteration == 100 or (previous_em_cost - em_cost) / number_of_voxels < stop_criterion:
                        break
                    previous_em_cost = em_cost

                    self.gmm.fit_parameters(statistics, number_of_voxels)
                    if update_bias_field:
                        self.bias_field.fit(
                            masked_images, class_priors, self.gmm.means, self.gmm.variances, voxels, mask.shape
                        )

                class_likelihoods = self.gmm.class_likelihoods(data, class_likelihoods)
                maximal_deformation_applied, cost_with_deformation_prior = self.atlas.deform_mesh(
                    mesh,
                    transform,
                    class_likelihoods,
                    voxel_selection,
                    {
                        "Verbose": False,
                        "MaximalDeformationStopCriterion": options.maximal_deformation_stop_criterion,
                        "LineSearchMaximalDeformationIntervalStopCriterion": options.line_search_maximal_deformation_interval_stop_criterion,
                        "MaximumNumberOfIterations": options.maximum_number_of_deformation_iterations,
                        "BFGS-MaximumMemoryLength": options.bfgs_maximum_memory_length,
                    },
                )
                logger.debug(f"maximalDeformationApplied: {maximal_deformation_applied:.4f}")

                history_of_cost.append(cost_with_deformation_prior + min_log_gmm_prior)
                if (history_of_cost[-2] - history_of_cost[-1]) / number_of_voxels < stop_criterion:
                    break

            self.optimization_summary.append(
                {
                    "numberOfIterations": iteration + 1,
                    "perVoxelCost": history_of_cost[-1] / number_of_voxels,
                }
            )

            final_positions = subject_to_template(mesh.points, transform)

            # Keep only the cumulative deformation instead of a history of positions
            self.deformation = final_positions - initial_positions + initial_deformation_applied
            self.deformation_atlas_file_name = level.atlas_file_name

    def fitted_model(self, debug_file_name=None):
        # The last resolution level may have left the bias field basis functions downsampled
        if self.deformation is None or self.deformation_atlas_file_name is None:
            raise RuntimeError("fit_model must run before fitted_model")
        self.bias_field.downsample_basis_functions([1, 1, 1])
        means, variances, mixture_weights = self.gmm.parameters()
        model = FittedModel(
            means=means,
            variances=variances,
            mixture_weights=mixture_weights,
            gaussians_per_class=self.gmm.gaussians_per_class,
            fractions_table=self.class_fractions,
            deformation=self.deformation,
            deformation_atlas=self.deformation_atlas_file_name,
            model_specification=self.model_specification,
            image_file_names=self.image_file_names,
            mask=self.mask,
            cropping=self.cropping,
            transform=self.transform.as_numpy_array,
            image_buffers=self.image_buffers,
            bias_fields=self.bias_field.bias_fields(),
        )
        if debug_file_name is not None:
            with open(debug_file_name, "wb") as f:
                pickle.dump(replace(model, image_buffers=None, bias_fields=None), f, protocol=pickle.HIGHEST_PROTOCOL)
        return model
