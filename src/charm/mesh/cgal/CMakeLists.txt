# CGAL 6 is header-only, so it is fetched as a pinned release tarball. SOURCE_SUBDIR names a path that does not
# exist so that FetchContent_MakeAvailable skips CGAL's own CMake setup and only provides the include tree.
#
# Only Mesh_3, AABB_tree and Surface_mesh are used. None of them reach CGAL_Core or zlib, so both are left out
# (zlib only matters for gzipped .inr, and meshing._write_inr writes uncompressed).
#
# Eigen is required even though Mesh_3 does not include it directly. C3T3_helpers.h includes
# linear_least_squares_fitting_3.h, which picks Default_diagonalize_traits based on CGAL_EIGEN3_ENABLED. Without
# Eigen, CGAL falls back to the deprecated CGAL::Diagonalize_traits, which upstream warns "can lead to precision
# issues". This runs in refine_mesh_3's surface vertex projection, so it changes mesh geometry.
#
# CGAL picks up GMP/MPFR through __has_include, see CGAL/Installation/internal/enable_third_party_libraries.h.
FetchContent_Declare(
  cgal
  URL https://github.com/CGAL/cgal/releases/download/v6.2/CGAL-6.2-library.tar.xz
  URL_HASH SHA256=f75d2b2486842fb47222c780c7241c262d392802d0811143c4e9b539972ee55b
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  SOURCE_SUBDIR do-not-add-subdirectory
)
# Pinned to 3.4.0, the series CGAL 6.2 is tested against (it asks for >= 3.3.7). Distro packages have moved to
# Eigen 5, which CGAL does not support.
FetchContent_Declare(
  eigen
  URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
  URL_HASH SHA256=8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  SOURCE_SUBDIR do-not-add-subdirectory
)
FetchContent_MakeAvailable(cgal eigen)

if(POLICY CMP0167)
  cmake_policy(SET CMP0167 NEW)
endif()
find_package(Boost 1.74 REQUIRED CONFIG)

function(find_math_library name header lib)
  find_path(${name}_INCLUDE_DIR NAMES ${header} REQUIRED)
  find_library(${name}_LIBRARY NAMES ${lib} REQUIRED)
  add_library(charm::${name} INTERFACE IMPORTED)
  target_include_directories(charm::${name} SYSTEM INTERFACE ${${name}_INCLUDE_DIR})
  target_link_libraries(charm::${name} INTERFACE ${${name}_LIBRARY})
endfunction()

find_math_library(GMP gmp.h gmp)
find_math_library(MPFR mpfr.h mpfr)

charm_add_module(_cgal module.cpp mesh_volumes.cpp segment_intersection.cpp)
target_include_directories(_cgal SYSTEM PRIVATE "${cgal_SOURCE_DIR}/include" "${eigen_SOURCE_DIR}")
target_compile_definitions(_cgal PRIVATE CGAL_CONCURRENT_MESH_3 CGAL_LINKED_WITH_TBB CGAL_EIGEN3_ENABLED)
# Mesh_3 allocates through tbb::scalable_allocator
target_link_libraries(_cgal PRIVATE Boost::headers TBB::tbb TBB::tbbmalloc charm::GMP charm::MPFR)
