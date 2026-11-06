"""The fast morphology has to agree with scipy.ndimage for the structuring elements in use."""

import numpy as np
import pytest
from scipy import ndimage

from charm.utils import morphology


@pytest.fixture(params=[(2, "C"), (3, "C"), (3, "F")], ids=["2d", "3d", "3d-fortran"])
def mask(request):
    ndim, order = request.param
    rng = np.random.default_rng(ndim)
    shape = (23, 31, 17)[:ndim]
    smooth = ndimage.gaussian_filter(rng.normal(size=shape), 1.5)
    return np.array(smooth > 0.02, order=order)


@pytest.mark.parametrize("iterations", [1, 2, 3])
def test_dilation_and_erosion_match_scipy(mask, iterations):
    cross = ndimage.generate_binary_structure(mask.ndim, 1)
    cube = ndimage.generate_binary_structure(mask.ndim, mask.ndim)
    np.testing.assert_array_equal(
        morphology.dilate_cross(mask, iterations), ndimage.binary_dilation(mask, cross, iterations)
    )
    np.testing.assert_array_equal(
        morphology.dilate_cube(mask, iterations), ndimage.binary_dilation(mask, cube, iterations)
    )
    np.testing.assert_array_equal(
        morphology.erode_cross(mask, iterations), ndimage.binary_erosion(mask, cross, iterations)
    )
    np.testing.assert_array_equal(
        morphology.erode_cube(mask, iterations), ndimage.binary_erosion(mask, cube, iterations)
    )
    np.testing.assert_array_equal(
        morphology.open_cross(mask, iterations), ndimage.binary_opening(mask, cross, iterations)
    )
    np.testing.assert_array_equal(
        morphology.open_cube(mask, iterations), ndimage.binary_opening(mask, cube, iterations)
    )


def test_fill_holes_matches_scipy(mask):
    np.testing.assert_array_equal(morphology.fill_holes(mask), ndimage.binary_fill_holes(mask))


def test_input_is_left_alone(mask):
    original = mask.copy()
    morphology.dilate_cube(mask, 2)
    morphology.erode_cross(mask, 2)
    np.testing.assert_array_equal(mask, original)
