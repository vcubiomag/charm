"""Exercises the segmentation library surface that CHARM calls."""

import itertools
from pathlib import Path

import numpy as np
import pytest
import scipy.ndimage

from charm.segmentation import gems
from charm.segmentation.affine import AffineRegistration, InitializationOptions
from charm.segmentation.bias_field import BiasField
from charm.segmentation.gmm import GMM
from charm.segmentation.io import read_mesh_collection, write_mesh_collection
from charm.segmentation.probabilistic_atlas import ProbabilisticAtlas, subject_to_template
from charm.segmentation.samseg import ModelSpecification, OptimizationOptions, ResolutionLevel, Samseg
from charm.segmentation.segmentation_utils import log_transform, write_image
from charm.segmentation.warp_mesh import _rasterize_deformation

RAS2LPS = np.diag([-1.0, -1.0, 1.0, 1.0])


def write_shifted_copy(source: str, destination: Path, shift_in_ras: float) -> np.ndarray:
    image = gems.KvlImage(source)
    shift = np.eye(4)
    shift[:3, 3] = shift_in_ras
    image_to_world = RAS2LPS @ shift @ RAS2LPS @ image.transform_matrix.as_numpy_array
    image.write(str(destination), gems.KvlTransform(np.asfortranarray(image_to_world)))
    return shift


def test_affine_registration_recovers_known_shift(tmp_path: Path, mni_image: str) -> None:
    shifted = tmp_path / "shifted.nii.gz"
    shift = write_shifted_copy(mni_image, shifted, -10.0)

    registration = gems.KvlAffineRegistration(-100, 10, 0, [0], 0, [4.0], True, 1.0, "b")
    registration.read_images(str(shifted), mni_image)
    registration.initialize_transform()
    registration.register()

    estimated = np.linalg.inv(registration.get_transformation_matrix())
    np.testing.assert_allclose(shift, RAS2LPS @ estimated @ RAS2LPS, atol=1e-3)


def test_affine_registration_rejects_other_interpolators() -> None:
    with pytest.raises(ValueError):
        gems.KvlAffineRegistration(-100, 10, 0, [0], 0, [4.0], True, 1.0, "l")


def test_rigid_registration_writes_aligned_image(tmp_path: Path, mni_image: str) -> None:
    shifted = tmp_path / "shifted.nii.gz"
    write_shifted_copy(mni_image, shifted, -10.0)
    registered = tmp_path / "registered.nii.gz"

    registration = gems.KvlRigidRegistration()
    registration.read_images(str(shifted), mni_image)
    registration.initialize_transform()
    registration.register()
    registration.write_out_result(str(registered))

    fixed = gems.KvlImage(str(shifted)).get_image_buffer()
    result = gems.KvlImage(str(registered)).get_image_buffer()
    assert np.corrcoef(result.ravel(), fixed.ravel())[0, 1] > 0.99


@pytest.mark.parametrize("extension", ["mgz", "nii.gz"])
def test_image_round_trip(tmp_path: Path, cube_image: str, extension: str) -> None:
    image = gems.KvlImage(cube_image)
    path = str(tmp_path / f"image.{extension}")
    image.write(path, image.transform_matrix)

    reread = gems.KvlImage(path)
    np.testing.assert_array_equal(reread.get_image_buffer(), image.get_image_buffer())
    np.testing.assert_allclose(reread.transform_matrix.as_numpy_array, image.transform_matrix.as_numpy_array, atol=1e-5)


def test_cropping_to_a_smaller_template(tmp_path: Path, noisy_cube_image: str) -> None:
    image = gems.KvlImage(noisy_cube_image)
    template = tmp_path / "template.nii.gz"
    gems.KvlImage(np.asfortranarray(np.zeros((10, 12, 14), dtype=np.float32))).write(
        str(template), image.transform_matrix
    )

    cropped = gems.KvlImage(noisy_cube_image, str(template))

    # The mapped bounding box is one voxel larger than the template on each axis.
    assert cropped.crop_slices == (slice(0, 11, 1), slice(0, 13, 1), slice(0, 15, 1))
    np.testing.assert_array_equal(cropped.get_image_buffer(), image.get_image_buffer()[cropped.crop_slices])


def test_cropping_matches_for_every_file_format(tmp_path: Path, noisy_cube_image: str) -> None:
    image = gems.KvlImage(noisy_cube_image)
    template = tmp_path / "template.nii.gz"
    gems.KvlImage(np.asfortranarray(np.zeros((10, 12, 14), dtype=np.float32))).write(
        str(template), image.transform_matrix
    )

    for extension in ("nii", "nii.gz", "mgz"):
        path = str(tmp_path / f"image.{extension}")
        image.write(path, image.transform_matrix)
        cropped = gems.KvlImage(path, str(template))
        np.testing.assert_array_equal(cropped.get_image_buffer(), image.get_image_buffer()[cropped.crop_slices])


def test_assigning_again_replaces_the_mesh(cube_atlas: Path) -> None:
    collection = read_mesh_collection(cube_atlas / "atlas.h5")
    first = collection.reference_mesh.points
    contents = collection.contents()
    # Once without anything holding on to the mesh of the first assignment, and once with
    collection.assign(*contents)
    held = collection.reference_mesh
    collection.assign(*contents)
    np.testing.assert_array_equal(held.points, first)
    np.testing.assert_array_equal(collection.reference_mesh.points, first)


def test_image_buffer_conversions(tmp_path: Path) -> None:
    buffer = np.arange(24, dtype=np.float32).reshape(2, 3, 4, order="F")
    np.testing.assert_array_equal(gems.KvlImage(buffer).get_image_buffer(), buffer)
    # A C-ordered buffer has to come back with the same indexing, not transposed in memory.
    np.testing.assert_array_equal(
        gems.KvlImage(np.arange(24, dtype=np.float32).reshape(2, 3, 4)).get_image_buffer(),
        np.arange(24, dtype=np.float32).reshape(2, 3, 4),
    )
    # CHARM's neck adjustment passes (x, y, z, contrast) buffers.
    assert gems.KvlImage(buffer[..., np.newaxis]).get_image_buffer().shape == (2, 3, 4)

    smoothed = gems.KvlImage.smooth_image_buffer(np.ones((8, 8, 8), dtype=np.float32, order="F"), [1.0, 1.0, 1.0])
    np.testing.assert_allclose(smoothed, 1.0, atol=1e-5)

    assert gems.KvlImage(np.zeros((2, 2, 2), dtype=np.float32, order="F")).crop_slices == ()

    # The buffer of an image is a view of it, which stays valid as long as the buffer is around
    view = gems.KvlImage(buffer).get_image_buffer()
    assert not view.flags.writeable
    np.testing.assert_array_equal(view, buffer)


def test_itk_errors_surface_as_python_exceptions() -> None:
    with pytest.raises(RuntimeError):
        gems.KvlImage("/does/not/exist.nii.gz")
    with pytest.raises(FileNotFoundError):
        read_mesh_collection("/does/not/exist.h5")


def test_mesh_properties(cube_atlas: Path) -> None:
    collection = read_mesh_collection(cube_atlas / "atlas.h5")
    collection.k = 0.25
    assert collection.k == pytest.approx(0.25)
    mesh = collection.reference_mesh

    points = mesh.points
    points += 1.0  # CHARM updates the returned arrays in place
    mesh.points = points
    np.testing.assert_array_equal(collection.reference_mesh.points, points)

    # CHARM's background masking assigns nested lists.
    mesh.alphas = [[0.0, 1.0]] * points.shape[0]
    np.testing.assert_array_equal(mesh.alphas, np.tile([0.0, 1.0], (points.shape[0], 1)))

    size = np.array([40, 40, 40])
    assert mesh.rasterize(size, 1).shape == (40, 40, 40)
    assert mesh.rasterize((40, 40, 40)).shape == (40, 40, 40, 2)
    assert mesh.rasterize_values((40, 40, 40), mesh.points).shape == (40, 40, 40, 3)

    before = mesh.points
    mesh.scale(np.array([0.5, 0.5, 0.5]))
    np.testing.assert_allclose(mesh.points, before * 0.5)


def test_mesh_collection_round_trip(tmp_path: Path, cube_atlas: Path) -> None:
    source = read_mesh_collection(cube_atlas / "atlas.h5")
    source.k = 0.25
    path = tmp_path / "copy.h5"
    write_mesh_collection(source, path)

    copy = read_mesh_collection(path)
    assert copy.k == pytest.approx(0.25)
    np.testing.assert_allclose(copy.reference_mesh.points, source.reference_mesh.points, atol=1e-4)
    np.testing.assert_array_equal(copy.reference_mesh.alphas, source.reference_mesh.alphas)
    np.testing.assert_array_equal(copy.contents()[1], source.contents()[1])
    np.testing.assert_array_equal(copy.contents()[3], source.contents()[3])

    # A file being rewritten isn't served from the cache
    source.k = 0.5
    write_mesh_collection(source, path)
    assert read_mesh_collection(path).k == pytest.approx(0.5)


def test_get_submesh_adds_boundary_nodes_to_mask(cube_atlas: Path) -> None:
    collection = read_mesh_collection(cube_atlas / "atlas.h5")
    mesh = collection.reference_mesh

    mask = np.zeros(mesh.points.shape[0], dtype=bool)
    mask[0] = True
    submesh = mesh.get_submesh(mask)

    assert mask.sum() > 1
    np.testing.assert_array_equal(submesh.points, mesh.points[mask])
    with pytest.raises(TypeError):
        mesh.get_submesh(mask.astype(np.uint8))


def test_rasterizer_interpolates_linearly_and_covers_every_voxel_once(cube_atlas: Path) -> None:
    collection = read_mesh_collection(cube_atlas / "atlas.h5")
    mesh = collection.reference_mesh
    points = mesh.points
    size = [int(extent) + 1 for extent in points.max(axis=0)]

    # Voxels exactly on a face belong to the tetrahedron on one side of it, which for the far faces of the mesh is a
    # tetrahedron that doesn't exist. All other voxels must be drawn.
    interior = (slice(0, -1),) * 3

    # A linear function is reproduced exactly, whichever tetrahedron ends up drawing a voxel
    drawn = mesh.rasterize_values(size, points)
    voxels = np.stack(np.meshgrid(*[np.arange(n) for n in size], indexing="ij"), axis=-1)
    np.testing.assert_allclose(drawn[interior], voxels[interior], atol=1e-9)

    mesh.alphas = [[0.0, 1.0]] * points.shape[0]
    assert mesh.rasterize(size, 1)[interior].min() == 65535


def test_cost_visits_every_voxel_exactly_once(cube_atlas: Path) -> None:
    collection = read_mesh_collection(cube_atlas / "atlas.h5")
    collection.k = 0.0
    mesh = collection.reference_mesh
    points = mesh.points
    size = tuple(int(extent) + 1 for extent in points.max(axis=0))

    # Move the mesh to where voxels aren't exactly on its faces, and to where they are for the two extremes
    for offset in (0.0, 0.5, 0.123456):
        mesh.points = points * 0.97 + offset
        mesh.alphas = [[1.0]] * points.shape[0]
        voxels = np.arange(np.prod(size), dtype=np.int64)

        # A likelihood of e makes every visit to a voxel add exactly -1 to the cost
        likelihoods = np.full((voxels.size, 1), np.e, dtype=np.float32)
        calculator = gems.KvlCostAndGradientCalculator(
            "AtlasMeshToIntensityImage",
            [],
            "Sliding",
            likelihoods=likelihoods,
            voxel_selection=gems.KvlVoxelSelection(size, voxels),
        )
        visits = -calculator.evaluate_mesh_position(mesh)[0]

        mesh.alphas = [[0.0, 1.0]] * points.shape[0]
        covered = np.count_nonzero(mesh.rasterize(size, 1))
        assert visits == pytest.approx(covered, abs=1e-1)


def sparse_priors(mesh: gems.KvlMesh, shape: tuple) -> tuple:
    mask = np.zeros(shape, dtype=bool)
    mask[5:30, 7:21, 3:35] = True
    voxels = np.flatnonzero(mask.ravel(order="F"))
    return mesh.rasterize_priors(gems.KvlVoxelSelection(shape, voxels)), voxels


def test_rasterize_priors_matches_rasterize_at_the_requested_voxels(cube_atlas: Path) -> None:
    collection = read_mesh_collection(cube_atlas / "atlas.h5")
    mesh = collection.reference_mesh
    shape = (40, 40, 40)
    priors, voxels = sparse_priors(mesh, shape)

    # With a weight of one for each intensity, the statistics of a class hold the sum of its priors and intensities
    data = np.ones((voxels.size, 1))
    statistics = priors.compute_class_statistics(data, mesh.alphas.shape[1])

    expected = mesh.rasterize(shape, -1).reshape(-1, mesh.alphas.shape[1], order="F")[voxels] / 65535
    np.testing.assert_allclose(statistics[:, 0], expected.sum(axis=0), rtol=1e-3)


def test_results_do_not_depend_on_the_number_of_threads(cube_atlas: Path, noisy_cube_image: str) -> None:
    image = gems.KvlImage(noisy_cube_image)
    default = gems.get_number_of_threads()
    try:
        results = []
        for threads in (1, 3, 8):
            gems.set_number_of_threads(threads)
            collection = read_mesh_collection(cube_atlas / "atlas.h5")
            collection.k = 0.1
            mesh = collection.reference_mesh
            calculator = gems.KvlCostAndGradientCalculator("MutualInformation", [image], "Sliding")
            optimizer = gems.KvlOptimizer(
                "L-BFGS",
                mesh,
                calculator,
                {
                    "MaximalDeformationStopCriterion": 0.001,
                    "LineSearchMaximalDeformationIntervalStopCriterion": 0.001,
                    "BFGS-MaximumMemoryLength": 12,
                },
            )
            for _ in range(5):
                optimizer.step_optimizer()
            results.append((*calculator.evaluate_mesh_position(mesh), mesh.points))
    finally:
        gems.set_number_of_threads(default)

    for cost, gradient, points in results[1:]:
        assert cost == results[0][0]
        np.testing.assert_array_equal(gradient, results[0][1])
        np.testing.assert_array_equal(points, results[0][2])


def test_optimizer_converges_on_a_nonlinear_problem(cube_atlas: Path) -> None:
    collection = read_mesh_collection(cube_atlas / "atlas.h5")
    # The deformation prior makes this a nonlinear problem, unlike the quadratic fit to the target positions alone
    collection.k = 0.05
    mesh = collection.reference_mesh
    points = mesh.points
    inside = np.all((points > 0) & (points < 39), axis=1)[:, np.newaxis]
    target = points + 1.5 * np.sin(points * np.pi / 39.0)[:, ::-1] * inside
    calculator = gems.KvlCostAndGradientCalculator("PointSet", [], "Sliding", target_points=target + 1e-9)
    optimizer = gems.KvlOptimizer(
        "L-BFGS",
        mesh,
        calculator,
        {
            "MaximalDeformationStopCriterion": 1e-6,
            "LineSearchMaximalDeformationIntervalStopCriterion": 1e-6,
            "BFGS-MaximumMemoryLength": 3,
        },
    )

    start, startGradient = calculator.evaluate_mesh_position(mesh)
    costs = [start]
    while True:
        cost, maximalDeformation = optimizer.step_optimizer()
        if maximalDeformation == 0:
            break
        costs.append(cost)

    # More iterations than the length of the history, so that the oldest entries have been dropped along the way
    assert len(costs) > 8
    assert all(later <= earlier + 1e-9 for earlier, later in itertools.pairwise(costs))
    assert np.linalg.norm(calculator.evaluate_mesh_position(mesh)[1]) < 1e-2 * np.linalg.norm(startGradient)


def test_folded_mesh_gets_the_sentinel_cost(cube_atlas: Path, cube_image: str) -> None:
    collection = read_mesh_collection(cube_atlas / "atlas.h5")
    collection.k = 0.1
    mesh = collection.reference_mesh
    calculator = gems.KvlCostAndGradientCalculator("MutualInformation", [gems.KvlImage(cube_image)], "Affine")
    assert calculator.evaluate_mesh_cost(mesh) == calculator.evaluate_mesh_position(mesh)[0]

    # Mirroring the mesh turns all of its tetrahedra inside out
    mesh.points = mesh.points * [-1.0, 1.0, 1.0]
    cost, gradient = calculator.evaluate_mesh_position(mesh)
    assert cost == np.finfo(np.float64).max
    assert not gradient.any()


def test_gmm_matches_the_textbook_formulas(cube_atlas: Path) -> None:
    collection = read_mesh_collection(cube_atlas / "atlas.h5")
    mesh = collection.reference_mesh
    shape = (40, 40, 40)
    classPriors, voxels = sparse_priors(mesh, shape)
    priors = mesh.rasterize(shape, -1).reshape(-1, 2, order="F")[voxels] / 65535
    data = np.random.default_rng(0).normal(size=(voxels.size, 2))

    gmm = GMM([2, 1], 2, False)
    gmm.initialize_parameters(data, classPriors)
    gmm.downsample_hyperparameters(np.array([1, 1, 1]))
    statistics, cost = gmm.compute_posteriors(data, classPriors)

    densities = np.empty((voxels.size, 3))
    means, variances, mixture_weights = gmm.parameters()
    for g in range(3):
        deviations = data - means[g]
        distances = np.sum(deviations * np.linalg.solve(variances[g], deviations.T).T, axis=1)
        normalization = 2 * np.pi * np.sqrt(np.linalg.det(variances[g]))
        densities[:, g] = np.exp(-distances / 2) / normalization * mixture_weights[g]
    joint = densities * priors[:, gmm.gaussian_classes]
    posteriors = joint / (joint.sum(axis=1, keepdims=True) + np.finfo(float).eps)
    assert cost == pytest.approx(-np.log(joint.sum(axis=1) + np.finfo(float).eps).sum(), rel=1e-3)

    likelihoods = gmm.class_likelihoods(data)
    assert likelihoods.dtype == np.float32
    np.testing.assert_allclose(likelihoods[:, 0], densities[:, 0] + densities[:, 1], rtol=1e-6)

    gmm.fit_parameters(statistics, voxels.size)
    for g in range(3):
        weights = posteriors[:, g]
        total = weights.sum()
        mean = data.T @ weights / total
        deviations = data - mean
        hyper = gmm.hyper_measurements
        variance = (deviations.T @ (deviations * weights[:, np.newaxis]) + hyper * np.eye(2)) / (total + hyper)
        means, variances, mixture_weights = gmm.parameters()
        np.testing.assert_allclose(means[g], mean, rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(variances[g], variance, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(mixture_weights[:2].sum(), 1.0)


def test_kernels_specialized_on_the_number_of_contrasts_agree_with_the_general_one() -> None:
    rng = np.random.default_rng(0)
    for contrasts in (1, 2, 3, 4):
        data = rng.normal(size=(100, contrasts))
        A = rng.normal(size=(contrasts, contrasts))
        variance = A @ A.T + np.eye(contrasts)
        mean = rng.normal(size=(contrasts, 1))
        deviations = data - mean.T
        distances = np.sum(deviations * np.linalg.solve(variance, deviations.T).T, axis=1)
        expected = np.exp(-distances / 2) / np.sqrt((2 * np.pi) ** contrasts * np.linalg.det(variance))
        gmm = GMM([1], contrasts, False, mean.T, variance[np.newaxis], np.ones(1))
        np.testing.assert_allclose(gmm.class_likelihoods(data)[:, 0], expected, rtol=1e-5)


def test_bias_field_kronecker_operations_match_explicit_matrices() -> None:
    rng = np.random.default_rng(0)
    shape = (9, 7, 11)
    field = BiasField(shape, np.array([4.0, 4.0, 5.0]))
    W0, W1, W2 = field.basis_functions
    W = np.kron(W2, np.kron(W1, W0))
    number_of_functions = W.shape[1]

    coefficients = rng.normal(size=number_of_functions)
    field.coefficients = coefficients[:, np.newaxis]
    np.testing.assert_allclose(field.bias_fields()[..., 0].ravel(order="F"), W @ coefficients, atol=1e-12)

    volume = np.asfortranarray(rng.normal(size=shape))
    np.testing.assert_allclose(field._project(volume, field.basis_functions), W.T @ volume.ravel(order="F"), atol=1e-12)

    weights = np.asfortranarray(rng.uniform(size=shape))
    np.testing.assert_allclose(
        field._precision(weights), W.T @ (W * weights.ravel(order="F")[:, np.newaxis]), atol=1e-12
    )


def test_bias_field_fit_recovers_a_known_field(cube_atlas: Path) -> None:
    collection = read_mesh_collection(cube_atlas / "atlas.h5")
    mesh = collection.reference_mesh
    shape = (40, 40, 40)
    # With a single class that has all of the prior, the posteriors are all one and the fit a plain least squares
    mesh.alphas = [[1.0]] * mesh.points.shape[0]
    priors, voxels = sparse_priors(mesh, shape)

    field = BiasField(shape, np.array([40.0, 40.0, 40.0]))
    truth = np.random.default_rng(0).normal(size=(np.prod([f.shape[1] for f in field.basis_functions]), 2))
    field.coefficients = truth
    images = field.corrected_data(np.zeros((voxels.size, 2)), voxels) * -1

    # One Gaussian with mean zero: the images are nothing but the bias field
    gmm = GMM([1], 2, False, np.zeros((1, 2)), np.array([[[2.0, 0.5], [0.5, 1.0]]]), np.ones(1))
    gmm.compute_posteriors(images, priors, keep_posteriors=True)
    field.fit(images, priors, gmm.means, gmm.variances, voxels, shape)
    np.testing.assert_allclose(field.corrected_data(images, voxels), 0.0, atol=1e-4)


def test_affine_grid_search_prefers_the_aligned_position(cube_atlas: Path, cube_image: str) -> None:
    affine = AffineRegistration(cube_image, str(cube_atlas / "atlas.h5"), str(cube_atlas / "template.nii.gz"))
    affine.set_up(None, 1.0)
    affine.options = InitializationOptions(scaling_center=[0.0, 0.0, 0.0])

    to_image = np.linalg.solve(affine.image_to_world, affine.template_to_world)
    transform = gems.KvlTransform(to_image)
    mesh = ProbabilisticAtlas().get_mesh(affine.mesh_collection_file_name, transform=transform, k=0.0)
    positions = subject_to_template(mesh.points, transform)

    *_, best_shift, best_transform = affine.grid_search(
        mesh, positions, horizontal_table_shifts=[-8.0, 0.0, 8.0], initial_table_shifts=[[0.0, 0.0, 0.0]]
    )
    assert best_shift == [0.0, 0.0, 0.0]
    np.testing.assert_allclose(best_transform, to_image)


class CubeSamseg(Samseg):
    def __init__(self, image: str, specification: ModelSpecification, options: OptimizationOptions) -> None:
        self.atlas = ProbabilisticAtlas()
        self.model_specification = specification
        self.optimization_options = options

        loaded = gems.KvlImage(image)
        self.transform = gems.KvlTransform(np.eye(4))
        self.voxel_spacing = np.linalg.norm(loaded.transform_matrix.as_numpy_array[:3, :3], axis=0)
        buffers = loaded.get_image_buffer()[..., np.newaxis]
        self.mask = buffers[..., 0] > 0
        self.image_buffers = log_transform(buffers, self.mask)
        self.deformation = None
        self.deformation_atlas_file_name = None


def test_fit_model_segments_the_cube(tmp_path: Path, cube_atlas: Path, cube_image: str, noisy_cube_image: str) -> None:
    atlas_file = str(cube_atlas / "atlas.h5")
    specification = ModelSpecification.from_atlas(
        str(cube_atlas),
        atlas_file_name=atlas_file,
        use_diagonal_covariances=False,
        k=0.1,
        bias_field_smoothing_kernel_size=100,
    )
    options = OptimizationOptions(
        [ResolutionLevel(atlas_file, 1.0, maximum_number_of_iterations=10, estimate_bias_field=False)]
    )

    model = CubeSamseg(noisy_cube_image, specification, options)
    model.fit_model()
    assert len(model.optimization_summary) == 1

    mesh = model.atlas.get_mesh(atlas_file, model.transform, 0.1, model.deformation, model.deformation_atlas_file_name)
    assert isinstance(model.gmm, GMM)
    voxels = np.flatnonzero(model.mask.ravel(order="F"))
    bias_corrected = model.image_buffers - model.bias_field.bias_fields(model.mask)
    data = np.ascontiguousarray(bias_corrected.reshape(-1, 1, order="F")[voxels])
    segmentation = model.gmm.segment(
        mesh, gems.KvlVoxelSelection(model.mask.shape, voxels), data, np.eye(2), [1.0, 1.0], [0, 1]
    ).astype(np.float32)

    truth = gems.KvlImage(cube_image).get_image_buffer() == 1
    estimated = segmentation == 1
    assert 2.0 * np.sum(truth & estimated) / (np.sum(truth) + np.sum(estimated)) > 0.95

    output = tmp_path / "segmentation.nii.gz"
    everything = tuple(slice(0, n, 1) for n in model.mask.shape)
    write_image(str(output), segmentation, everything, gems.KvlImage(noisy_cube_image))
    np.testing.assert_array_equal(gems.KvlImage(str(output)).get_image_buffer(), segmentation)


def test_segment_matches_the_argmax_of_rasterized_priors_and_likelihoods(cube_atlas: Path) -> None:
    collection = read_mesh_collection(cube_atlas / "atlas.h5")
    mesh = collection.reference_mesh
    shape = (40, 40, 40)
    rng = np.random.default_rng(0)

    # Two classes, split over three structures through a fraction, with one structure's posterior scaled
    alphas = mesh.alphas
    mesh.alphas = np.column_stack((alphas[:, 0], alphas[:, 1] * 0.3, alphas[:, 1] * 0.7))
    fractions = np.array([[1.0, 0.25, 0.0], [0.0, 0.75, 1.0]])
    factors = np.array([1.0, 1.0, 1.5])
    labels = np.array([7, 11, 13], dtype=np.uint16)

    mask = rng.random(shape) < 0.7
    voxels = np.flatnonzero(mask.ravel(order="F"))
    data = rng.normal(size=(voxels.size, 2))
    means = np.array([[0.0, 0.0], [1.0, 1.0], [-1.0, 0.5]])
    variances = np.array([np.eye(2), 0.5 * np.eye(2), 2.0 * np.eye(2)])
    gmm = GMM([2, 1], 2, True, means, variances, np.array([0.3, 0.7, 1.0]))

    segmentation = gmm.segment(
        mesh, gems.KvlVoxelSelection(shape, voxels), data, fractions, factors.tolist(), labels.tolist()
    )

    priors = mesh.rasterize(shape, -1).reshape(-1, 3, order="F") / 65535.0
    selected = mask.ravel(order="F")
    posteriors = priors.copy()
    posteriors[selected] *= (gmm.class_likelihoods(data).astype(float) @ fractions) * factors
    expected = labels[np.argmax(posteriors, axis=1)]

    # Rounding the rasterized priors can only change the outcome for a near tie
    sorted_posteriors = np.sort(posteriors, axis=1)
    clear = sorted_posteriors[:, 2] > 1.001 * sorted_posteriors[:, 1]
    np.testing.assert_array_equal(segmentation.ravel(order="F")[clear], expected[clear])
    assert clear.mean() > 0.9


def test_deformation_fills_far_image_boundary_without_searching(
    cube_atlas: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    collection = read_mesh_collection(cube_atlas / "atlas.h5")
    mesh = collection.reference_mesh
    points = mesh.points
    size = [int(extent) + 1 for extent in points.max(axis=0)]

    def fail(*args, **kwargs):
        raise AssertionError("the far boundary should not need a search for the closest voxel")

    monkeypatch.setattr(scipy.ndimage, "distance_transform_edt", fail)

    # Linear in the position, which is what makes the value next to the far boundary a known one
    deformation = np.zeros(points.shape)
    deformation[:, 0] = points[:, 1] * 0.25
    dense = _rasterize_deformation(mesh, deformation, size)
    assert dense.shape == (*size, 3)
    y = np.arange(size[1], dtype=np.float32) * 0.25
    interiorY = min(size[1] - 2, 3)
    np.testing.assert_allclose(dense[2, interiorY, 2, 0], y[interiorY], atol=1e-5)
    np.testing.assert_allclose(dense[2, -1, 2, 0], y[-2], atol=1e-5)
