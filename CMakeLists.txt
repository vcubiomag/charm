cmake_minimum_required(VERSION 3.28...4.4)
project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION} LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# Every library links statically into an extension module, which only needs to export its init function
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)
# Shared, so that the thread cap set through charm.utils._thickness also holds for the CGAL modules
find_package(TBB 2021 REQUIRED CONFIG)

include(FetchContent)
include(CharmModules)

add_subdirectory(src/charm/mesh)
add_subdirectory(src/charm/segmentation)
add_subdirectory(src/charm/utils)
