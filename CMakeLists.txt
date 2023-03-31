cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
set(jusha_VERSION 0.1.0)
set(jusha_VERSION_MAJOR 0)
set(jusha_VERSION_MINOR 1)
set(jusha_VERSION_PATCH 0)

project(jusha VERSION ${jusha_VERSION} LANGUAGES CXX CUDA)

set(CMAKE_MODULE_PATH
    ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

message(STATUS "System name: ${CMAKE_SYSTEM_NAME}")

# CUDA is always required.
set(CMAKE_FIND_LIBRARY_SUFFIXES .a .so)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CUDA_ARCH "60;70;75;80;86;87;89" CACHE STRING "CUDA Architectures")

# set_target_properties(jusha PROPERTIES CUDA_ARCHITECTURES "${CUDA_ARCH}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall")

# set(CUDA_PROPAGATE_HOST_FLAGS ON)
set(CUDA_BLOCKSIZE 256 CACHE STRING "CUDA block size.")

# Make the gampack headers visible without extra paths.
include_directories(src)
include_directories("${CMAKE_BINARY_DIR}/src")

set(CPP_SRC_LIST
    src/utility.cpp
    src/timer.cpp
    src/cuda/cuda_config.cpp
    src/cuda/heap_manager.cpp
    src/cuda/heap_allocator.cpp
    src/cuda/kernel.cpp

    src/sparse/solver/diag_precond.cpp
    src/sparse/solver/jacobi_precond.cpp
    src/sparse/distri_obj.cpp
    src/sparse/matrix_reader.cpp
    src/sparse/mmio.cpp
    src/sparse/csr_matrix.cpp

    src/sparse/parser/su2_mesh_parser.cpp
)

set(CU_SRC_LIST
    src/cuda/utility.cu
    src/cuda/bitmap_scan.cu
    src/cuda/array.cu

    src/sparse/csr_matrix.cu
    src/sparse/mesh.cu
)

# Headers are here just so CMake will put them in project files.
set(HEADERS_LIST
    src/timer.h
    src/utility.h

    src/cuda/bitmap_scan.h
    src/cuda/heap_manager.h
    src/cuda/heap_allocator.h
    src/cuda/array.h
    src/cuda/array_util.h
    src/cuda/allocator.h
    src/cuda/for_each.hpp
    src/cuda/kernel.hpp
    src/cuda/intrinsic.hpp
    src/cuda/policy.hpp

    src/sparse/matrix.h
    src/sparse/distri_obj.h
    src/sparse/csr_matrix.h
    src/sparse/matrix_reader.h
    src/sparse/mmio.h
    src/sparse/mesh.h

    src/sparse/solver/precond.h

    src/sparse/parser/su2_mesh_parser.h
)
ADD_LIBRARY(jusha STATIC ${CU_SRC_LIST} ${CPP_SRC_LIST} ${HEADERS_LIST})
target_link_libraries(jusha ${CUDA_curand_LIBRARY})

add_subdirectory(tests)

# add_subdirectory (utest)
# for distribution
set(CPACK_PACKAGE_VERSION_MAJOR ${jusha_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${jusha_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${jusha_VERSION_PATCH})
set(CPACK_SOURCE_TBZ2 "OFF")
set(CPACK_SOURCE_TGZ "ON")
set(CPACK_SOURCE_TZ "OFF")

# set (CPACK_SOURCE_IGNORE_FILES "/dist/;/drop/;/example/;/examples/;cudalt.py;.*~;Makefile.am;.*sln;.*vcxproj*;.*png;.*eps;.*bib")
# set (CPACK_PROJECT_CONFIG_FILE "${PROJECT_BINARY_DIR}/CPackOptions.cmake")
INSTALL(TARGETS jusha jusha RUNTIME DESTINATION bin LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
INSTALL(DIRECTORY src/ DESTINATION include/jusha FILES_MATCHING PATTERN "*.h")
INSTALL(DIRECTORY src/ DESTINATION include/jusha FILES_MATCHING PATTERN "*.hpp")
include(CPack)
