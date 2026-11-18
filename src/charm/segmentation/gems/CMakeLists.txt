include(CharmITK)
find_package(ITK 5.4 REQUIRED CONFIG COMPONENTS ${CHARM_ITK_MODULES})
# Writes the headers that register the IO factories of the components above, which KvlImage relies on to read NIfTI
# and MGH files. This is the part of ITK_USE_FILE that is needed, without its directory-wide include paths and flags.
itk_generate_factory_registration()

set(GEMS_WARNINGS -Wall -Wextra -Wno-sign-compare -Wno-unused-parameter)

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
target_include_directories(kvlGEMSCommon PUBLIC core "${CMAKE_CURRENT_BINARY_DIR}/ITKFactoryRegistration")
target_link_libraries(kvlGEMSCommon PUBLIC ${ITK_LIBRARIES})
foreach(factory IN LISTS ITK_FACTORY_LIST)
  string(TOUPPER ${factory} factory)
  target_compile_definitions(kvlGEMSCommon PUBLIC ITK_${factory}_FACTORY_REGISTER_MANAGER)
endforeach()
target_compile_options(kvlGEMSCommon PRIVATE ${GEMS_WARNINGS})
# The numerics never read errno or floating point exception flags, and these allow math calls to be inlined and
# vectorized. Results are still reproducible for one build; it's only across compilers that they may round differently.
target_compile_options(kvlGEMSCommon PUBLIC -fno-math-errno -fno-trapping-math -ffp-contract=fast -fopenmp-simd)
set_target_properties(kvlGEMSCommon PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)

charm_add_module(gemsbindings
  bindings/module.cxx
  bindings/pyKvlImage.cxx
  bindings/pyKvlMesh.cxx
  bindings/pyKvlTransform.cxx
)
target_include_directories(gemsbindings PRIVATE bindings)
target_compile_options(gemsbindings PRIVATE ${GEMS_WARNINGS})
target_link_libraries(gemsbindings PRIVATE kvlGEMSCommon)
# nanobind only garbage collects sections on Linux when building for size, and most of each static ITK library is
# unreachable from here
if(NOT APPLE)
  target_link_options(gemsbindings PRIVATE -Wl,--gc-sections)
endif()
