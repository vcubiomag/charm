include(${PROJECT_SOURCE_DIR}/cmake/ITK.cmake)
find_package(ITK 5.4 REQUIRED CONFIG COMPONENTS ${CHARM_ITK_MODULES})
# Registers the IO factories of the components above, which KvlImage relies on to read NIfTI and MGH files.
include(${ITK_USE_FILE})

include(CheckIPOSupported)
check_ipo_supported(RESULT GEMS_IPO_SUPPORTED OUTPUT GEMS_IPO_MESSAGE LANGUAGES CXX)
if(GEMS_IPO_SUPPORTED)
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
endif()

add_library(kvlGEMSCommon STATIC
  core/kvlAtlasMeshCollection.cxx
  core/kvlAtlasMeshDeformationLBFGSOptimizer.cxx
  core/kvlAtlasMeshDeformationOptimizer.cxx
  core/kvlAtlasMeshPositionCostAndGradientCalculator.cxx
  core/kvlAtlasMeshToIntensityImageCostAndGradientCalculator.cxx
  core/kvlAtlasMeshToPointSetCostAndGradientCalculator.cxx
  core/kvlMutualInformationCostAndGradientCalculator.cxx
  core/kvlTetrahedronTable.cxx
)
target_include_directories(kvlGEMSCommon PUBLIC core)
target_link_libraries(kvlGEMSCommon PUBLIC ${ITK_LIBRARIES})
target_compile_options(kvlGEMSCommon PRIVATE -Wall -Wextra -Wno-sign-compare -Wno-unused-parameter)
# The numerics never read errno or floating point exception flags, and these allow math calls to be inlined and
# vectorized. Results are still reproducible for one build; it's only across compilers that they may round differently.
target_compile_options(kvlGEMSCommon PUBLIC -fno-math-errno -fno-trapping-math -ffp-contract=fast -fopenmp-simd)

charm_add_module(gemsbindings
  bindings/module.cxx
  bindings/pyKvlImage.cxx
  bindings/pyKvlMesh.cxx
  bindings/pyKvlTransform.cxx
)
target_include_directories(gemsbindings PRIVATE bindings)
target_compile_options(gemsbindings PRIVATE -Wall -Wextra -Wno-sign-compare -Wno-unused-parameter)
target_link_libraries(gemsbindings PRIVATE kvlGEMSCommon)
# nanobind only garbage collects sections on Linux when building for size, and most of each static ITK library is
# unreachable from here
if(NOT APPLE)
  target_link_options(gemsbindings PRIVATE -Wl,--gc-sections)
endif()

