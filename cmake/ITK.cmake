# ITK is built from source as static libraries with only the modules gems uses, so it links into gemsbindings and
# the wheel carries no ITK libraries. The install goes into CHARM_DEPS_DIR under a key of everything that affects
# the binaries, which lets wheel builds for different Python versions share one ITK build.

set(CHARM_ITK_VERSION 5.4.8)
set(CHARM_ITK_URL https://github.com/InsightSoftwareConsortium/ITK/archive/refs/tags/v${CHARM_ITK_VERSION}.tar.gz)
set(CHARM_ITK_SHA256 ff66d6648d896ced24107b93470e03e3f89e94026ecd785c08dd50061a2e1832)
# The commit ITK 5.4.8 pins in Modules/Remote/MGHIO.remote.cmake. Fetching it as an archive keeps git out of the
# build and pins the content by hash.
set(CHARM_MGHIO_URL
  https://github.com/InsightSoftwareConsortium/itkMGHImageIO/archive/af74507c1ddc82722637c37d1f5d169e3000553a.tar.gz)
set(CHARM_MGHIO_SHA256 a3078333cb1c318fe9dc6c74e14bbc30a3acce765d295dbd98403f6b2564a186)

set(CHARM_ITK_MODULES
  ITKCommon
  ITKIOImageBase
  ITKIONIFTI
  ITKImageGrid
  ITKImageIntensity
  ITKMesh
  ITKMetricsv4
  ITKOptimizersv4
  ITKRegistrationMethodsv4
  ITKSmoothing
  MGHIO
)

if(DEFINED ENV{CHARM_DEPS_DIR})
  set(charm_deps_default "$ENV{CHARM_DEPS_DIR}")
else()
  set(charm_deps_default "${CMAKE_SOURCE_DIR}/build/deps")
endif()
set(CHARM_DEPS_DIR "${charm_deps_default}" CACHE PATH "Where dependencies built from source are kept")
option(CHARM_BUILD_ITK "Build ITK from source rather than use the one ITK_DIR points to" ON)

if(CHARM_BUILD_ITK)
  set(itk_args
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    -DCMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD}
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    -DCMAKE_C_VISIBILITY_PRESET=hidden
    -DCMAKE_CXX_VISIBILITY_PRESET=hidden
    -DCMAKE_VISIBILITY_INLINES_HIDDEN=ON
    # Lets the linker drop the parts of each ITK library that gemsbindings doesn't reach
    "-DCMAKE_C_FLAGS=-ffunction-sections -fdata-sections"
    "-DCMAKE_CXX_FLAGS=-ffunction-sections -fdata-sections"
    -DBUILD_SHARED_LIBS=OFF
    -DBUILD_TESTING=OFF
    -DBUILD_EXAMPLES=OFF
    -DITK_BUILD_DEFAULT_MODULES=OFF
    # On by default, and it requests every core module including the test kernel, which needs every IO format
    -DITKGroup_Core=OFF
    -DITK_FORBID_DOWNLOADS=ON
    -DITK_LEGACY_REMOVE=ON
    -DITK_WRAP_PYTHON=OFF
  )
  if(APPLE)
    list(APPEND itk_args
      -DCMAKE_OSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}
      -DCMAKE_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES}
      -DCMAKE_OSX_SYSROOT=${CMAKE_OSX_SYSROOT}
    )
  endif()
  foreach(module IN LISTS CHARM_ITK_MODULES)
    list(APPEND itk_args -DModule_${module}=ON)
  endforeach()

  string(SHA256 itk_key
    "${CHARM_ITK_SHA256};${CHARM_MGHIO_SHA256};${CMAKE_CXX_COMPILER_ID};${CMAKE_CXX_COMPILER_VERSION};${itk_args}")
  string(SUBSTRING ${itk_key} 0 12 itk_key)
  set(itk_root "${CHARM_DEPS_DIR}/itk-${CHARM_ITK_VERSION}-${itk_key}")
  set(itk_prefix "${itk_root}/install")

  if(NOT EXISTS "${itk_prefix}/.complete")
    file(MAKE_DIRECTORY "${CHARM_DEPS_DIR}")
    file(LOCK "${itk_root}.lock" GUARD PROCESS TIMEOUT 7200)
    if(NOT EXISTS "${itk_prefix}/.complete")
      message(STATUS "Building ITK ${CHARM_ITK_VERSION} into ${itk_prefix}; this takes several minutes")
      file(REMOVE_RECURSE "${itk_root}")

      file(DOWNLOAD ${CHARM_ITK_URL} "${itk_root}/itk.tar.gz" EXPECTED_HASH SHA256=${CHARM_ITK_SHA256})
      file(DOWNLOAD ${CHARM_MGHIO_URL} "${itk_root}/mghio.tar.gz" EXPECTED_HASH SHA256=${CHARM_MGHIO_SHA256})
      file(ARCHIVE_EXTRACT INPUT "${itk_root}/itk.tar.gz" DESTINATION "${itk_root}")
      set(itk_source "${itk_root}/ITK-${CHARM_ITK_VERSION}")
      file(ARCHIVE_EXTRACT INPUT "${itk_root}/mghio.tar.gz" DESTINATION "${itk_root}/mghio")
      file(GLOB mghio_source "${itk_root}/mghio/*")
      file(RENAME "${mghio_source}" "${itk_source}/Modules/Remote/MGHIO")

      if(CMAKE_CXX_COMPILER_LAUNCHER)
        list(APPEND itk_args -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER})
      endif()
      if(DEFINED ENV{CMAKE_BUILD_PARALLEL_LEVEL})
        set(jobs $ENV{CMAKE_BUILD_PARALLEL_LEVEL})
      else()
        cmake_host_system_information(RESULT jobs QUERY NUMBER_OF_LOGICAL_CORES)
      endif()

      execute_process(
        COMMAND ${CMAKE_COMMAND} -S "${itk_source}" -B "${itk_root}/build" -G ${CMAKE_GENERATOR}
                -DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM} -DCMAKE_INSTALL_PREFIX=${itk_prefix} ${itk_args}
        COMMAND_ERROR_IS_FATAL ANY
      )
      execute_process(
        COMMAND ${CMAKE_COMMAND} --build "${itk_root}/build" --target install --parallel ${jobs}
        COMMAND_ERROR_IS_FATAL ANY
      )
      # Only the install is used; the sources and build tree take over a gigabyte.
      file(REMOVE_RECURSE "${itk_root}/build" "${itk_source}" "${itk_root}/mghio")
      file(REMOVE "${itk_root}/itk.tar.gz" "${itk_root}/mghio.tar.gz")
      file(TOUCH "${itk_prefix}/.complete")
    endif()
    file(LOCK "${itk_root}.lock" RELEASE)
  endif()

  string(REGEX MATCH "^[0-9]+\\.[0-9]+" itk_series ${CHARM_ITK_VERSION})
  set(ITK_DIR "${itk_prefix}/lib/cmake/ITK-${itk_series}" CACHE PATH "" FORCE)
endif()
