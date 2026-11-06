#include <algorithm>
#include <cstdlib>
#include <new>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "sanlm.h"

namespace nb = nanobind;

namespace {

nb::ndarray<nb::numpy, float> Sanlm(const nb::ndarray<const float, nb::ndim<3>, nb::f_contig, nb::device::cpu> &image,
                                    int v, int f, bool useRician) {
    const size_t shape[3] = {image.shape(0), image.shape(1), image.shape(2)};
    auto *denoised = static_cast<float *>(std::malloc(image.size() * sizeof(float)));
    if (!denoised) {
        throw std::bad_alloc();
    }
    std::copy(image.data(), image.data() + image.size(), denoised);
    {
        nb::gil_scoped_release release;
        const int dims[3] = {int(shape[0]), int(shape[1]), int(shape[2])};
        anlm(denoised, v, f, useRician, dims);
    }
    nb::capsule owner(denoised, [](void *d) noexcept { std::free(d); });
    return nb::ndarray<nb::numpy, float>(denoised, 3, shape, owner, nullptr, nb::dtype<float>(), nb::device::cpu::value, 0, 'F');
}

} // namespace

NB_MODULE(_sanlm, m) {
    m.def("sanlm", &Sanlm, nb::arg("image"), nb::arg("v"), nb::arg("f"), nb::arg("use_rician") = false,
          "Adaptive non-local means denoising, as in CAT12, of a copy of the image");
}
