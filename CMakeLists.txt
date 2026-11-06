cmake_minimum_required(VERSION 3.28)
project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION} LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Default build type" FORCE)
endif()

find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module NumPy)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE nanobind_ROOT
)
find_package(nanobind CONFIG REQUIRED)
# Shared, so that the thread cap set through charm.utils._thickness also holds for the CGAL modules
find_package(TBB 2021 REQUIRED)

include(FetchContent)
include(cmake/CharmModules.cmake)

add_subdirectory(src/charm/mesh)
add_subdirectory(src/charm/segmentation)
add_subdirectory(src/charm/utils)
