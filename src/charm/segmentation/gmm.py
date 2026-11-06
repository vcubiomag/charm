import numpy as np

from charm.segmentation import gems

eps = np.finfo(float).eps


class GMM:
    def __init__(
        self,
        gaussians_per_class,
        number_of_contrasts,
        use_diagonal_covariances=True,
        means: np.ndarray | None = None,
        variances: np.ndarray | None = None,
        mixture_weights: np.ndarray | None = None,
    ):
        self.gaussians_per_class = gaussians_per_class
        self.number_of_classes = len(gaussians_per_class)
        self.number_of_gaussians = sum(gaussians_per_class)
        self.number_of_contrasts = number_of_contrasts
        self.use_diagonal_covariances = use_diagonal_covariances

        self.means = means
        self.variances = variances
        self.mixture_weights = mixture_weights

        self.gaussian_classes = np.repeat(np.arange(self.number_of_classes), gaussians_per_class)
        self.class_offsets = np.concatenate(([0], np.cumsum(gaussians_per_class, dtype=int)))[:-1]

        # The inverse-Wishart is normalizable when its number of measurements is at least (number_of_contrasts-1).
        # Use (number_of_contrasts-2)+1 instead to avoid near-zero variances when EM assigns almost no voxels to a
        # Gaussian, e.g. because of a tiny mixture weight.
        self.full_hyper_measurements = (number_of_contrasts - 2) + 1 + eps
        self.hyper_measurements = self.full_hyper_measurements

    def parameters(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The means, variances and mixture weights."""
        if self.means is None or self.variances is None or self.mixture_weights is None:
            raise RuntimeError("the GMM parameters have not been initialized")
        return self.means, self.variances, self.mixture_weights

    def _density_parameters(self):
        # What the mixture kernels need: the means, the inverses of the Cholesky factors of the variances, and the log
        # of everything multiplying the exponential of each Gaussian
        means, variances, mixture_weights = self.parameters()
        L = np.linalg.cholesky(variances)
        log_scales = (
            -0.5 * self.number_of_contrasts * np.log(2 * np.pi)
            - np.sum(np.log(np.diagonal(L, axis1=1, axis2=2)), axis=1)
            + np.log(mixture_weights)
        )
        return np.ascontiguousarray(means), np.ascontiguousarray(np.linalg.inv(L)), log_scales

    def _unpack_statistics(self, statistics):
        # statistics is ( groups x [ sum of weights, sum of weighted intensities, sums of weighted intensity products ] )
        first, second = np.triu_indices(self.number_of_contrasts)
        quadratic = np.zeros((statistics.shape[0], self.number_of_contrasts, self.number_of_contrasts))
        quadratic[:, first, second] = statistics[:, 1 + self.number_of_contrasts :]
        quadratic[:, second, first] = statistics[:, 1 + self.number_of_contrasts :]
        return statistics[:, 0], statistics[:, 1 : 1 + self.number_of_contrasts], quadratic

    def _force_diagonal(self, variances):
        return variances * np.eye(self.number_of_contrasts) if self.use_diagonal_covariances else variances

    def initialize_parameters(self, data, class_priors):
        components_per_class = np.asarray(self.gaussians_per_class)

        totals, linear, quadratic = self._unpack_statistics(
            class_priors.compute_class_statistics(np.ascontiguousarray(data), self.number_of_classes)
        )
        class_means = linear / totals[:, np.newaxis]
        class_variances = self._force_diagonal(
            quadratic / totals[:, np.newaxis, np.newaxis]
            - class_means[:, :, np.newaxis] * class_means[:, np.newaxis, :]
        )

        # Each component copies its class variance and takes as mean the middle of one of equal intervals splitting
        # [mean - std, mean + std]. With equal mixture weights this roughly mimics the single class Gaussian.
        classes = self.gaussian_classes
        standard_deviations = np.sqrt(np.diagonal(class_variances, axis1=1, axis2=2))[classes]
        components = (np.arange(self.number_of_gaussians) - self.class_offsets[classes])[:, np.newaxis]
        interval_sizes = 2 * standard_deviations / components_per_class[classes, np.newaxis]
        self.variances = class_variances[classes]
        self.means = class_means[classes] - standard_deviations + interval_sizes * (components + 0.5)
        self.mixture_weights = 1.0 / components_per_class[classes]

    def compute_posteriors(self, data, class_priors, keep_posteriors=False):
        # The E-step: returns the sufficient statistics to pass to fit_parameters and the minus log-likelihood
        min_log_likelihood, statistics = class_priors.compute_posteriors(
            data, *self._density_parameters(), self.gaussians_per_class, keep_posteriors
        )
        return statistics, min_log_likelihood

    def class_likelihoods(self, data, likelihoods=None):
        # P( data | class ) as a (voxels x classes) matrix, optionally in an existing one to save reallocating it
        if likelihoods is None:
            likelihoods = np.empty((data.shape[0], self.number_of_classes), dtype=np.float32)
        gems.compute_class_likelihoods(data, *self._density_parameters(), self.gaussians_per_class, likelihoods)
        return likelihoods

    def segment(self, mesh, voxel_selection, data, class_fractions, structure_factors, labels):
        # The structure of highest posterior in every voxel of the image, from its prior for voxels outside of voxel_selection
        return mesh.segment(
            voxel_selection,
            data,
            *self._density_parameters(),
            self.gaussians_per_class,
            class_fractions,
            structure_factors,
            labels,
        )

    def fit_parameters(self, statistics, number_of_voxels):
        totals, linear, quadratic = self._unpack_statistics(statistics)
        hyper_measurements = self.hyper_measurements

        self.means = linear / totals[:, np.newaxis]
        scatter = (
            quadratic - totals[:, np.newaxis, np.newaxis] * self.means[:, :, np.newaxis] * self.means[:, np.newaxis, :]
        )
        self.variances = self._force_diagonal(
            (scatter + np.eye(self.number_of_contrasts) * hyper_measurements)
            / (totals + hyper_measurements)[:, np.newaxis, np.newaxis]
        )

        # The weights within each class mixture sum to one
        mixture_weights = totals + number_of_voxels * eps
        self.mixture_weights = (
            mixture_weights / np.add.reduceat(mixture_weights, self.class_offsets)[self.gaussian_classes]
        )

    def min_log_prior(self):
        # -log IW( variance | hyperVariance * hyperVarianceNumberOfMeasurements,
        #                     hyperVarianceNumberOfMeasurements - numberOfContrasts - 2 )
        # with an identity hyperVariance, plus the log-determinant term of the (flat) prior on the means
        L = np.linalg.cholesky(self.parameters()[1])
        half_log_det_variances = np.sum(np.log(np.diagonal(L, axis1=1, axis2=2)), axis=1)
        traces = np.sum(np.linalg.inv(L) ** 2, axis=(1, 2))
        return np.sum(
            half_log_det_variances
            + traces * self.hyper_measurements / 2
            + self.hyper_measurements * half_log_det_variances
        )

    def downsample_hyperparameters(self, downsampling_factors):
        self.hyper_measurements = self.full_hyper_measurements / np.prod(downsampling_factors)
