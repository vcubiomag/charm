#include <cstdint>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>

#include <mmg/mmg3d/libmmg3d.h>

namespace nb = nanobind;
using namespace nb::literals;

template <typename T, size_t N>
using Rows = nb::ndarray<const T, nb::shape<-1, N>, nb::c_contig, nb::device::cpu>;
template <typename T>
using Column = nb::ndarray<const T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

template <typename T, typename... Shape>
static nb::ndarray<nb::numpy, T> to_numpy(std::vector<T> &&data, Shape... shape) {
  auto *owner = new std::vector<T>(std::move(data));
  nb::capsule deleter(owner, [](void *p) noexcept {
    delete static_cast<std::vector<T> *>(p);
  });
  size_t dims[] = {static_cast<size_t>(shape)...};
  return nb::ndarray<nb::numpy, T>(owner->data(), sizeof...(shape), dims, deleter);
}

template <typename T>
static std::vector<MMG5_int> to_mmg_int(const T *data, size_t size) {
  return std::vector<MMG5_int>(data, data + size);
}

template <typename T>
static std::vector<int64_t> to_int64(const std::vector<T> &data) {
  return std::vector<int64_t>(data.begin(), data.end());
}

class MmgMesh {
public:
  MmgMesh() {
    MMG3D_Init_mesh(MMG5_ARG_start, MMG5_ARG_ppMesh, &mesh, MMG5_ARG_ppMet, &met,
                    MMG5_ARG_end);
  }
  ~MmgMesh() {
    MMG3D_Free_all(MMG5_ARG_start, MMG5_ARG_ppMesh, &mesh, MMG5_ARG_ppMet, &met,
                   MMG5_ARG_end);
  }
  MmgMesh(const MmgMesh &) = delete;
  MmgMesh &operator=(const MmgMesh &) = delete;

  MMG5_pMesh mesh = nullptr;
  MMG5_pSol met = nullptr;
};

static void check(int ok, const char *what) {
  if (!ok)
    throw std::runtime_error(std::string("mmg3d: ") + what + " failed");
}

using Doubles = nb::ndarray<nb::numpy, double>;
using Integers = nb::ndarray<nb::numpy, int64_t>;

// Improve a tetrahedral mesh in place of running `mmg3d_O3 -nosurf -nofem -rmc`.
// Node numbers are 1-based, as in Msh.
static std::tuple<Doubles, Integers, Integers, Integers, Integers> remesh(Rows<double, 3> nodes, Rows<int64_t, 4> tetrahedra,
                        Column<int64_t> tetrahedra_tags, Rows<int64_t, 3> triangles,
                        Column<int64_t> triangles_tags,
                        std::optional<Column<double>> sizing, double hgrad,
                        std::optional<double> hmin, std::optional<double> hsiz,
                        bool noinsert, int verbosity) {
  const MMG5_int n_nodes = nodes.shape(0);
  const MMG5_int n_tetrahedra = tetrahedra.shape(0);
  const MMG5_int n_triangles = triangles.shape(0);
  if (tetrahedra_tags.shape(0) != tetrahedra.shape(0) ||
      triangles_tags.shape(0) != triangles.shape(0))
    throw std::invalid_argument("every element needs exactly one tag");
  if (sizing && sizing->shape(0) != nodes.shape(0))
    throw std::invalid_argument("sizing needs exactly one value per node");

  std::vector<double> coords(nodes.data(), nodes.data() + 3 * n_nodes);
  auto tets = to_mmg_int(tetrahedra.data(), 4 * n_tetrahedra);
  auto tet_refs = to_mmg_int(tetrahedra_tags.data(), n_tetrahedra);
  auto tris = to_mmg_int(triangles.data(), 3 * n_triangles);
  auto tri_refs = to_mmg_int(triangles_tags.data(), n_triangles);

  MmgMesh m;
  std::vector<int64_t> out_tets, out_tet_tags, out_tris, out_tri_tags;
  std::vector<double> out_nodes;
  MMG5_int np, ne, nt;
  {
    nb::gil_scoped_release release;

    check(MMG3D_Set_meshSize(m.mesh, n_nodes, n_tetrahedra, 0, n_triangles, 0, 0),
          "Set_meshSize");
    check(MMG3D_Set_vertices(m.mesh, coords.data(), nullptr), "Set_vertices");
    check(MMG3D_Set_tetrahedra(m.mesh, tets.data(), tet_refs.data()), "Set_tetrahedra");
    if (n_triangles > 0)
      check(MMG3D_Set_triangles(m.mesh, tris.data(), tri_refs.data()), "Set_triangles");

    check(MMG3D_Set_solSize(m.mesh, m.met, MMG5_Vertex, sizing ? n_nodes : 0, MMG5_Scalar),
          "Set_solSize");
    if (sizing) {
      std::vector<double> values(sizing->data(), sizing->data() + n_nodes);
      check(MMG3D_Set_scalarSols(m.met, values.data()), "Set_scalarSols");
    }

    check(MMG3D_Set_iparameter(m.mesh, m.met, MMG3D_IPARAM_verbose, verbosity), "verbose");
    check(MMG3D_Set_iparameter(m.mesh, m.met, MMG3D_IPARAM_nosurf, 1), "nosurf");
    check(MMG3D_Set_iparameter(m.mesh, m.met, MMG3D_IPARAM_nofem, 1), "nofem");
    check(MMG3D_Set_iparameter(m.mesh, m.met, MMG3D_IPARAM_noinsert, noinsert), "noinsert");
    // Zero selects mmg's default volume fraction, as a bare -rmc does.
    check(MMG3D_Set_dparameter(m.mesh, m.met, MMG3D_DPARAM_rmc, 0.), "rmc");
    check(MMG3D_Set_dparameter(m.mesh, m.met, MMG3D_DPARAM_hgrad, hgrad), "hgrad");
    if (hmin)
      check(MMG3D_Set_dparameter(m.mesh, m.met, MMG3D_DPARAM_hmin, *hmin), "hmin");
    if (hsiz)
      check(MMG3D_Set_dparameter(m.mesh, m.met, MMG3D_DPARAM_hsiz, *hsiz), "hsiz");

    if (MMG3D_mmg3dlib(m.mesh, m.met) == MMG5_STRONGFAILURE)
      throw std::runtime_error("mmg3d: unable to improve the mesh");

    check(MMG3D_Get_meshSize(m.mesh, &np, &ne, nullptr, &nt, nullptr, nullptr), "Get_meshSize");

    out_nodes.resize(3 * np);
    check(MMG3D_Get_vertices(m.mesh, out_nodes.data(), nullptr, nullptr, nullptr), "Get_vertices");

    std::vector<MMG5_int> raw_tets(4 * ne), raw_tet_refs(ne), raw_tris(3 * nt), raw_tri_refs(nt);
    check(MMG3D_Get_tetrahedra(m.mesh, raw_tets.data(), raw_tet_refs.data(), nullptr),
          "Get_tetrahedra");
    if (nt > 0)
      check(MMG3D_Get_triangles(m.mesh, raw_tris.data(), raw_tri_refs.data(), nullptr),
            "Get_triangles");

    out_tets = to_int64(raw_tets);
    out_tet_tags = to_int64(raw_tet_refs);
    out_tris = to_int64(raw_tris);
    out_tri_tags = to_int64(raw_tri_refs);
  }

  return {to_numpy(std::move(out_nodes), np, 3), to_numpy(std::move(out_tets), ne, 4),
          to_numpy(std::move(out_tet_tags), ne), to_numpy(std::move(out_tris), nt, 3),
          to_numpy(std::move(out_tri_tags), nt)};
}

NB_MODULE(_mmg3d, module) {
  module.def("remesh", &remesh, "nodes"_a, "tetrahedra"_a, "tetrahedra_tags"_a,
             "triangles"_a, "triangles_tags"_a, nb::kw_only(), "sizing"_a = nb::none(),
             "hgrad"_a = -1.0, "hmin"_a = nb::none(), "hsiz"_a = nb::none(),
             "noinsert"_a = false, "verbosity"_a = -1);
}
