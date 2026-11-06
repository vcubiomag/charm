import numpy as np

from charm.utils import _thickness


def create_rings(radii, img_size):
    img_size = 100

    coords = np.meshgrid(
        np.arange(img_size) - img_size / 2,
        np.arange(img_size) - img_size / 2,
        np.arange(img_size) - img_size / 2,
        indexing="xy",
    )

    R = np.sqrt(coords[0] ** 2 + coords[1] ** 2 + coords[2] ** 2)

    rings = np.zeros((img_size, img_size, img_size), dtype=np.uint8)
    for i, r_inner, r_outer in zip(range(len(radii)), radii[:-1], radii[1:]):
        rings[(R > r_inner) * (R < r_outer)] = i + 1

    return rings


def test_thickness_ring():
    outer_radius = 50
    inner_radius = 30
    img_size = 150

    coords = np.meshgrid(
        np.arange(img_size) - img_size / 2,
        np.arange(img_size) - img_size / 2,
        indexing="xy",
    )

    R = np.sqrt(coords[0] ** 2 + coords[1] ** 2)

    ring = (R < outer_radius) * (R > inner_radius)

    thick = _thickness.thickness_slice(ring.astype(np.uint8))
    assert np.allclose(thick[ring], 10, rtol=1e-1)


def test_calc_thickness():
    rings = create_rings([10, 20, 26], 60)
    thick = _thickness.calc_thickness(rings)
    thick = thick[:, :20:40]
    rings = rings[:, :20:40]
    assert np.allclose(thick[rings == 1], 5, atol=1)
    assert np.allclose(thick[rings == 2], 3, atol=1)


def test_results_do_not_depend_on_the_number_of_threads():
    labels = create_rings([10, 20, 26], 60)
    _thickness.set_number_of_threads(1)
    single = _thickness.calc_thickness(labels)
    _thickness.set_number_of_threads(8)
    many = _thickness.calc_thickness(labels)
    np.testing.assert_array_equal(single, many)


def test_label_filling_a_whole_slice():
    labels = np.ones((30, 40, 50), dtype=np.uint16)
    labels[0] = 0
    thickness = _thickness.calc_thickness(labels)
    assert np.all(thickness[0] == 0)
    assert np.all(thickness[1:] > 0)
