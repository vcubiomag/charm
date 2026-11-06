# Builds a nanobind module and its type stub, and installs both into the package directory of the CMakeLists.txt that
# defines them. The stub is written next to the sources, where type checkers find it in an editable install.
function(charm_add_module name)
  # The modules are the numerical hot spots of a run, which nanobind's default of building for size would slow down
  nanobind_add_module(${name} NB_STATIC NOMINSIZE LTO ${ARGN})
  file(RELATIVE_PATH destination "${PROJECT_SOURCE_DIR}/src" "${CMAKE_CURRENT_SOURCE_DIR}")
  install(TARGETS ${name} LIBRARY DESTINATION "${destination}")
  nanobind_add_stub(${name}_stub
    MODULE ${name}
    OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/${name}.pyi"
    PYTHON_PATH $<TARGET_FILE_DIR:${name}>
    DEPENDS ${name}
  )
  install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/${name}.pyi" DESTINATION "${destination}")
endfunction()
