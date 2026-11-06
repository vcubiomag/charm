#include <algorithm>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>

#include "cgal.h"

namespace nb = nanobind;

namespace {

using SizingField = nb::ndarray<const float, nb::ndim<3>, nb::f_contig, nb::device::cpu>;
template <typename T>
using Points = nb::ndarray<const T, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu>;

int MeshImage(const std::string &imageFileName, const std::string &outputFileName, float facetAngle, float facetSize,
              float facetDistance, float cellRadiusEdgeRatio, float cellSize, bool doPerturb, bool doExude, bool doLloyd) {
    nb::gil_scoped_release release;
    return mesh_image(imageFileName.c_str(), outputFileName.c_str(), facetAngle, facetSize, facetDistance,
                      cellRadiusEdgeRatio, cellSize, doPerturb, doExude, doLloyd);
}

int MeshImageSizingField(const std::string &imageFileName, const std::string &outputFileName, float facetAngle,
                         const SizingField &facetSize, const SizingField &facetDistance, float cellRadiusEdgeRatio,
                         const SizingField &cellSize, bool doPerturb, bool doExude, bool doLloyd) {
    nb::gil_scoped_release release;
    return mesh_image_sizing_field(imageFileName.c_str(), outputFileName.c_str(), facetAngle, facetSize.data(),
                                   facetDistance.data(), cellRadiusEdgeRatio, cellSize.data(), doPerturb, doExude, doLloyd);
}

template <typename Output, typename Input>
nb::ndarray<nb::numpy, Output> ToRows(const std::vector<Input> &values, size_t columns) {
    auto *data = new Output[values.size()];
    std::copy(values.begin(), values.end(), data);
    nb::capsule owner(data, [](void *d) noexcept { delete[] static_cast<Output *>(d); });
    const size_t shape[2] = {values.size() / columns, columns};
    return nb::ndarray<nb::numpy, Output>(data, 2, shape, owner);
}

std::tuple<nb::ndarray<nb::numpy, std::int64_t>, nb::ndarray<nb::numpy, double>> SegmentTriangleIntersection(const Points<float> &vertices, const Points<std::int32_t> &faces,
                                      const Points<float> &segmentStart, const Points<float> &segmentEnd) {
    if (segmentStart.shape(0) != segmentEnd.shape(0)) {
        throw nb::value_error("segment_start and segment_end must have the same number of points");
    }
    std::pair<std::vector<int>, std::vector<float>> intersections;
    {
        nb::gil_scoped_release release;
        intersections = segment_triangle_intersection(vertices.data(), int(vertices.shape(0)), faces.data(), int(faces.shape(0)),
                                                      segmentStart.data(), segmentEnd.data(), int(segmentStart.shape(0)));
    }
    return {ToRows<std::int64_t>(intersections.first, 2), ToRows<double>(intersections.second, 3)};
}

} // namespace

NB_MODULE(_cgal, m) {
    m.def("mesh_image", &MeshImage, nb::arg("fn_image"), nb::arg("fn_out"), nb::arg("facet_angle"), nb::arg("facet_size"),
          nb::arg("facet_distance"), nb::arg("cell_radius_edge_ratio"), nb::arg("cell_size"), nb::arg("do_perturb") = false,
          nb::arg("do_exude") = false, nb::arg("do_lloyd") = false,
          "Tetrahedral mesh of a labeled .inr image, written to a .mesh file. Returns nonzero on failure.");
    m.def("mesh_image_sizing_field", &MeshImageSizingField, nb::arg("fn_image"), nb::arg("fn_out"), nb::arg("facet_angle"),
          nb::arg("facet_size"), nb::arg("facet_distance"), nb::arg("cell_radius_edge_ratio"), nb::arg("cell_size"),
          nb::arg("do_perturb") = false, nb::arg("do_exude") = false, nb::arg("do_lloyd") = false,
          "As mesh_image, with sizes and distances given per voxel");
    m.def("segment_triangle_intersection", &SegmentTriangleIntersection, nb::arg("vertices"), nb::arg("faces"),
          nb::arg("segment_start"), nb::arg("segment_end"),
          "The segment and triangle of each intersection between line segments and a triangle mesh, and its position");
}
