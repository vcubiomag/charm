#ifndef GEMS_PYKVLNUMPY_H
#define GEMS_PYKVLNUMPY_H

#include <cstdlib>
#include <new>
#include <vector>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
namespace nb = nanobind;

template<class T>
using NumpyArray = nb::ndarray<nb::numpy, T>;

// Image buffers are copied voxel by voxel in ITK's memory order, so they have to be F-contiguous.
// They are not restricted to three dimensions because callers also pass (x, y, z, 1) buffers.
using ImageBuffer = nb::ndarray<const float, nb::f_contig>;
using Matrix = nb::ndarray<const double, nb::ndim<2>>;

// Allocates memory for a numpy array, zeroed lazily by the operating system so that untouched pages stay free
template<class T>
T *allocateZeroedArray(size_t count){
    auto *data = static_cast<T *>(std::calloc(count, sizeof(T)));
    if (!data) {
        throw std::bad_alloc();
    }
    return data;
}

// Takes ownership of memory from allocateZeroedArray
template<class T>
NumpyArray<T> createNumpyArray(const std::vector<size_t> &shape, T *data, char order){
    nb::capsule free_when_done(data, [](void *d) noexcept { std::free(d); });
    return NumpyArray<T>(data, shape.size(), shape.data(), free_when_done, nullptr,
                         nb::dtype<T>(), nb::device::cpu::value, 0, order);
}

template<class T>
NumpyArray<T> createNumpyArrayCStyle(const std::vector<size_t> &shape, T *data){
    return createNumpyArray(shape, data, 'C');
}

template<class T>
NumpyArray<T> createNumpyArrayFStyle(const std::vector<size_t> &shape, T *data){
    return createNumpyArray(shape, data, 'F');
}

template<class MatrixType, class OffsetType>
NumpyArray<double> transformToNumpy(const MatrixType &matrix, const OffsetType &offset){
    auto *data = allocateZeroedArray<double>(16);
    for ( unsigned int row = 0; row < 3; row++ )
    {
        for ( unsigned int col = 0; col < 3; col++ )
        {
            data[ col * 4 + row ] = matrix[ row ][ col ];
        }
        data[ 12 + row ] = offset[ row ];
    }
    data[ 15 ] = 1.0;
    return createNumpyArrayFStyle<double>({4, 4}, data);
}

#endif //GEMS_PYKVLNUMPY_H
