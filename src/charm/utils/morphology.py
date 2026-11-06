"""Binary morphology on boolean arrays, for the structuring elements charm uses.

scipy.ndimage visits the structuring element for every voxel. These are logical
operations between shifted slices of the array instead, which the structuring
elements in use allow: the full cube is a product of lines along each axis, and
the cross a union of them. Nothing outside of the image is ever foreground.
"""

import numpy as np
from scipy import ndimage


def _slices(ndim, axis):
    head = [slice(None)] * ndim
    tail = [slice(None)] * ndim
    head[axis] = slice(1, None)
    tail[axis] = slice(None, -1)
    return tuple(head), tuple(tail)


def _face(ndim, axis, index):
    face = [slice(None)] * ndim
    face[axis] = index
    return tuple(face)


def dilate_cross(mask, iterations=1):
    """Dilation with the structuring element of connectivity 1."""
    for _ in range(iterations):
        result = mask.copy(order="K")
        for axis in range(mask.ndim):
            head, tail = _slices(mask.ndim, axis)
            result[head] |= mask[tail]
            result[tail] |= mask[head]
        mask = result
    return mask


def dilate_cube(mask, iterations=1):
    """Dilation with the full structuring element."""
    for _ in range(iterations):
        for axis in range(mask.ndim):
            head, tail = _slices(mask.ndim, axis)
            result = mask.copy(order="K")
            result[head] |= mask[tail]
            result[tail] |= mask[head]
            mask = result
    return mask


def erode_cross(mask, iterations=1):
    """Erosion with the structuring element of connectivity 1."""
    for _ in range(iterations):
        result = mask.copy(order="K")
        for axis in range(mask.ndim):
            head, tail = _slices(mask.ndim, axis)
            result[head] &= mask[tail]
            result[tail] &= mask[head]
            result[_face(mask.ndim, axis, 0)] = False
            result[_face(mask.ndim, axis, -1)] = False
        mask = result
    return mask


def erode_cube(mask, iterations=1):
    """Erosion with the full structuring element."""
    for _ in range(iterations):
        for axis in range(mask.ndim):
            head, tail = _slices(mask.ndim, axis)
            result = mask.copy(order="K")
            result[head] &= mask[tail]
            result[tail] &= mask[head]
            result[_face(mask.ndim, axis, 0)] = False
            result[_face(mask.ndim, axis, -1)] = False
            mask = result
    return mask


def open_cross(mask, iterations=1):
    return dilate_cross(erode_cross(mask, iterations), iterations)


def open_cube(mask, iterations=1):
    return dilate_cube(erode_cube(mask, iterations), iterations)


def fill_holes(mask):
    """Fill the background components that have no connection of connectivity 1 to the image border."""
    labels, count = ndimage.label(~mask)
    reaches_border = np.zeros(count + 1, dtype=bool)
    for axis in range(mask.ndim):
        reaches_border[labels[_face(mask.ndim, axis, 0)]] = True
        reaches_border[labels[_face(mask.ndim, axis, -1)]] = True
    reaches_border[0] = False
    return ~reaches_border[labels]


def dilate(image, n):
    image = np.nan_to_num(image) > 0.5
    return dilate_cube(image, n)


def erosion(image, n):
    image = np.nan_to_num(image) > 0.5
    return ~dilate(~image, n)
