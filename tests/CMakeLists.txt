Include(FetchContent)

FetchContent_Declare(
	Catch2
	GIT_REPOSITORY https://github.com/catchorg/Catch2.git
	GIT_TAG v2.13.4)

FetchContent_MakeAvailable(Catch2)
include(CTest)

add_executable(test_d2d src/test_d2d.cu)
target_link_libraries(test_d2d jusha Catch2::Catch2)
target_include_directories(test_d2d PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

# add_executable (test_scan src/test_scan.cu)
# target_link_libraries(test_scan jusha)

# cuda_add_executable(test_primitive_scan src/test_primitive_scan.cu)
# target_link_libraries(test_primitive_scan jusha)
add_executable(jutest src/unit_test.cpp
	src/test_array.cu
	src/foreach_test.cu
	src/foreach_block_test.cu
	src/foreach_shm_test.cu
	src/intrinsic_test.cu
	src/allocator_test.cpp
	src/distri_obj_test.cpp
)

target_link_libraries(jutest jusha Catch2::Catch2)
target_include_directories(jutest PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
