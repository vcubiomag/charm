find_package(Threads REQUIRED)
charm_add_module(_sanlm _sanlm.cpp sanlm.c)
target_link_libraries(_sanlm PRIVATE Threads::Threads)

add_subdirectory(gems)
