import numpy as np
import scipy.linalg

from charm.segmentation import gems


class BiasField:
    def __init__(self, image_size, smoothing_kernel_size):
        self.full_basis_functions = _dct_basis_functions(image_size, smoothing_kernel_size)
        self.basis_functions = [functions.copy() for functions in self.full_basis_functions]
        self.coefficients: np.ndarray | None = None
        self._pairwise_basis_functions = None

    @staticmethod
    def _project(volume, matrices):
        # Computes W' * volume(:) with W = matrices[2] \kron matrices[1] \kron matrices[0]. The transpose of an
        # F-ordered volume is C-contiguous, so axis 0, whose reduction shrinks the data the most, needs no copy.
        # The result is the F-order flattening of the projected tensor.
        n0, n1, n2 = volume.shape
        projected = (volume.T.reshape(-1, n0) @ matrices[0]).reshape(n2, n1, -1)
        projected = matrices[1].T @ projected
        return (matrices[2].T @ projected.reshape(n2, -1)).ravel()

    @staticmethod
    def _backproject(coefficients, matrices):
        # Computes W * coefficients, the mirror image of _project. The result is F-ordered.
        M0, M1, M2 = (functions.shape[1] for functions in matrices)
        n0, n1, n2 = (functions.shape[0] for functions in matrices)
        expanded = matrices[2] @ coefficients.reshape(M2, -1)
        expanded = matrices[1] @ expanded.reshape(n2, M1, M0)
        return (expanded.reshape(-1, M0) @ matrices[0].T).reshape(n2, n1, n0).T

    def _precision(self, weights):
        # Computes W' * diag( weights ) * W. Pointwise products of each pair of basis functions turn this into a single
        # projection. These products are symmetric in the two basis functions, so only the unique ones go through it.
        if self._pairwise_basis_functions is None:
            self._pairwise_basis_functions = []
            unique_pairs = []
            for functions in self.basis_functions:
                M = functions.shape[1]
                first, second = np.triu_indices(M)
                self._pairwise_basis_functions.append(functions[:, first] * functions[:, second])
                pairs = np.empty((M, M), dtype=int)
                pairs[first, second] = pairs[second, first] = np.arange(first.size)
                unique_pairs.append(pairs)
            # For each pair of 3-D basis functions, the position of its product in the projected tensor
            M0, M1, M2 = (pairs.shape[0] for pairs in unique_pairs)
            i2, i1, i0 = np.unravel_index(np.arange(M0 * M1 * M2), (M2, M1, M0))
            positions = (
                unique_pairs[2][np.ix_(i2, i2)],
                unique_pairs[1][np.ix_(i1, i1)],
                unique_pairs[0][np.ix_(i0, i0)],
            )
            self._unique_pair_positions = np.ravel_multi_index(
                positions, (unique_pairs[2].max() + 1, unique_pairs[1].max() + 1, unique_pairs[0].max() + 1)
            )
        return self._project(weights, self._pairwise_basis_functions)[self._unique_pair_positions]

    def _fitted_coefficients(self) -> np.ndarray:
        if self.coefficients is None:
            raise RuntimeError("the bias field has not been fitted")
        return self.coefficients

    def bias_fields(self, mask=None):
        coefficients = self._fitted_coefficients()
        number_of_contrasts = coefficients.shape[-1]
        image_size = tuple(functions.shape[0] for functions in self.basis_functions)
        fields = np.zeros(image_size + (number_of_contrasts,), order="F")
        for contrast in range(number_of_contrasts):
            field = self._backproject(coefficients[:, contrast], self.basis_functions)
            fields[:, :, :, contrast] = field if mask is None else field * mask
        return fields

    def corrected_data(self, masked_images, voxels, data=None):
        # The (voxels x contrasts) images at the given F-order voxel indices minus the bias fields, optionally in an
        # existing array to save reallocating it
        if data is None:
            data = np.empty_like(masked_images)
        W0, W1, W2 = self.basis_functions
        for contrast, coefficients in enumerate(self._fitted_coefficients().T):
            expanded = W2 @ coefficients.reshape(W2.shape[1], -1)
            row_coefficients = W1 @ expanded.reshape(W2.shape[0], W1.shape[1], W0.shape[1])
            gems.subtract_bias_field(masked_images, contrast, row_coefficients, W0, voxels, data)
        return data

    def fit(self, masked_images, priors, means, variances, voxels, image_size):
        # Eq. 8 of Van Leemput, "Automated Model-based Bias Field Correction of MR Images of the Brain", IEEE TMI 1999
        number_of_contrasts = means.shape[1]
        number_of_3d_functions = np.prod([functions.shape[1] for functions in self.basis_functions])
        system_size = number_of_3d_functions * number_of_contrasts

        def contrast_range(contrast):
            return slice(contrast * number_of_3d_functions, (contrast + 1) * number_of_3d_functions)

        # precisions are symmetric, so only the upper triangle of contrast pairs needs its weights and lhs block
        precisions = np.linalg.inv(variances)
        first_contrasts, second_contrasts = np.triu_indices(number_of_contrasts)
        number_of_pairs = first_contrasts.size
        terms = priors.compute_bias_field_terms(
            np.ascontiguousarray(masked_images),
            np.ascontiguousarray(precisions[:, first_contrasts, second_contrasts]),
            np.einsum("gij,gj->gi", precisions, means),
            first_contrasts.tolist(),
            second_contrasts.tolist(),
        )

        lhs = np.empty((system_size, system_size))
        rhs = np.empty(system_size)
        volume = np.zeros(image_size, order="F")
        for pair, (contrast1, contrast2) in enumerate(zip(first_contrasts, second_contrasts)):
            gems.scatter_to_volume(terms, pair, voxels, volume)
            block = self._precision(volume)
            lhs[contrast_range(contrast1), contrast_range(contrast2)] = block
            lhs[contrast_range(contrast2), contrast_range(contrast1)] = block.T

        for contrast in range(number_of_contrasts):
            gems.scatter_to_volume(terms, number_of_pairs + contrast, voxels, volume)
            rhs[contrast_range(contrast)] = self._project(volume, self.basis_functions)

        # The system is the normal equations of a weighted least-squares problem, so symmetric positive definite
        solution = scipy.linalg.solve(lhs, rhs, assume_a="pos", check_finite=False)
        self.coefficients = solution.reshape((number_of_contrasts, number_of_3d_functions)).T

    def downsample_basis_functions(self, downsampling_factors):
        self.basis_functions = [
            functions[::factor].copy() for functions, factor in zip(self.full_basis_functions, downsampling_factors)
        ]
        self._pairwise_basis_functions = None


def _dct_basis_functions(image_size, smoothing_kernel_size):
    # The bias field is a combination of the lowest frequency DCT-II basis functions
    basis_functions = []
    for N, delta in zip(image_size, smoothing_kernel_size):
        M = int(np.ceil(N / delta)) + 1
        N_virtual = (M - 1) * delta
        scaling = np.full(M, np.sqrt(2 / N_virtual))
        scaling[0] /= np.sqrt(2)
        frequencies = (np.arange(N) + 0.5) * np.pi / N_virtual
        basis_functions.append(np.cos(np.outer(frequencies, np.arange(M))) * scaling)
    return basis_functions
