charm_add_module(_kernels _kernels.cpp)

# A fork of mmg 5.8 that is about 1.5 times as fast on head meshes. EXCLUDE_FROM_ALL keeps mmg's own install
# rules out of the wheel.
FetchContent_Declare(
  mmg
  URL https://github.com/srimanachanta/mmg/archive/cd44074ae3f6d90c5e5ef24009c6ec95363103c7.tar.gz
  URL_HASH SHA256=f8f0ac70ba68a959437616b41d002ee8f4a9bf02bd0d9401e49cba86223d5340
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  EXCLUDE_FROM_ALL
)
set(BUILD "MMG3D" CACHE STRING "" FORCE)
set(LIBMMG3D_STATIC ON CACHE BOOL "" FORCE)
set(LIBMMG3D_SHARED OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(USE_ELAS OFF CACHE BOOL "" FORCE)
set(USE_VTK OFF CACHE BOOL "" FORCE)
# meshing._spatially_ordered does better than Scotch renumbering on head meshes.
set(USE_SCOTCH OFF CACHE BOOL "" FORCE)
# mmg still declares cmake_minimum_required(3.5).
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
FetchContent_MakeAvailable(mmg)

# mmg takes most of the time of a run, and spends it in small functions spread over many files.
# mmg's old minimum CMake version makes it ignore INTERPROCEDURAL_OPTIMIZATION.
set(MMG_OPTIONS -flto -fno-math-errno -fno-trapping-math -ffp-contract=fast)
if(CMAKE_C_COMPILER_ID STREQUAL "GNU")
  # The static library may be archived without GCC's LTO plugin
  list(APPEND MMG_OPTIONS -ffat-lto-objects)
endif()
target_compile_options(libmmg3d_a PRIVATE ${MMG_OPTIONS})

charm_add_module(_mmg3d _mmg3d.cpp)
target_link_libraries(_mmg3d PRIVATE libmmg3d_a)
target_link_options(_mmg3d PRIVATE -flto)

add_subdirectory(cgal)
