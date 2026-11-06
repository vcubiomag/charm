charm_add_module(_thickness _thickness.cpp)
target_compile_options(_thickness PRIVATE -fno-math-errno)
target_link_libraries(_thickness PRIVATE TBB::tbb)
