#include <thrust/transform.h>

// #define CATCH_CONFIG_RUNNER
#include <catch2/catch.hpp>

#include "timer.h"
#include "cuda/array.h"
#include "cuda/primitive.h"
#include "cuda/array_util.h"

TEST_CASE( "Simple", "[Array]" ) {
  int test_size = 1 << 23;
  jusha::cuda::MirroredArray<int> vector(test_size);
  REQUIRE( vector.size() == test_size );
  vector.resize(test_size+1);
  REQUIRE( vector.size() == (test_size +1));
  REQUIRE( vector.getGpuPtr() );
}


TEST_CASE( "Algorithm", "[bitmap scan]" ) {
  int test_size = 20;
  jusha::cuda::MirroredArray<int> vector(test_size);
  thrust::fill(vector.gbegin(), vector.gend(), 0xffffffff);
  
}

TEST_CASE( "VecMultiply", "[simple]" ) {
  int test_size = 20;
  JVector<int> v0(test_size), v1(test_size);
  v0.sequence(0);
  v1.sequence(0);
  JVector<int> v0v1(test_size);
  v0v1.zero();
  jusha::multiply(v0, v1, v0v1);

  for (int i = 0; i != test_size; i++)
    REQUIRE(v0v1[i] == (i*i));
 
}

TEST_CASE( "VecScale", "[array]" ) {
  int test_size = 256 * 20000 +124;
//int test_size = 256*1;
  JVector<int> v0(test_size);
  v0.sequence(0);
  v0.scale(0.0);
  JVector<int> v0v1(test_size);
  v0v1.zero();
  // both zeros
  REQUIRE(v0.isEqualTo(v0v1));  
}

    
TEST_CASE( "VecAdd", "[array]" ) {
int test_size = 256 * 20000 +124;
//int test_size = 256*1;
  JVector<int> v0(test_size), v1(test_size);
  v0.sequence(0);
  v1.sequence(0);
  JVector<int> v0v1(test_size);
  JVector<int> v0v1_ref(test_size);
  for (int i = 0; i != test_size; i++)
    v0v1_ref[i] = i*2;

  v0v1.zero();
  jusha::plus(v0, v1, v0v1);
  cudaDeviceSynchronize();
  double start = jusha::jusha_get_wtime();
  jusha::plus(v0, v1, v0v1);
  jusha::plus(v0, v1, v0v1);
  jusha::plus(v0, v1, v0v1);
  cudaDeviceSynchronize();
  double end = jusha::jusha_get_wtime();
  printf("jusha  plus took %f seconds\n", end-start);
  REQUIRE(v0v1.isEqualTo(v0v1_ref));
  // for (int i = 0; i != test_size; i++)
  //   REQUIRE(v0v1[i] == (i*2));
  jusha::plus_thrust(v0, v1, v0v1);
  cudaDeviceSynchronize();
  v0v1.zero();
  start = jusha::jusha_get_wtime();
  jusha::plus_thrust(v0, v1, v0v1);
  jusha::plus_thrust(v0, v1, v0v1);
  jusha::plus_thrust(v0, v1, v0v1);
  cudaDeviceSynchronize();
  end = jusha::jusha_get_wtime();
  printf("thrust plus took %f seconds\n", end-start);

  v0v1.zero();
  jusha::plus_block(v0, v1, v0v1);
  cudaDeviceSynchronize();
  start = jusha::jusha_get_wtime();
  jusha::plus_block(v0, v1, v0v1);
  jusha::plus_block(v0, v1, v0v1);
  jusha::plus_block(v0, v1, v0v1);
  cudaDeviceSynchronize();
  end = jusha::jusha_get_wtime();
  printf("block plus took %f seconds\n", end-start);
  REQUIRE(v0v1.isEqualTo(v0v1_ref));
  // for (int i = 0; i != test_size; i++)
  //   REQUIRE(v0v1[i] == (i*2));
  jusha::cuda_event_print() ;
}


TEST_CASE( "VecMinus", "[array]" ) {
  int test_size = 257;
  JVector<int> v0(test_size), v1(test_size);
  v0.sequence(0);
  v1.sequence(0);
  JVector<int> v0v1(test_size);
//v0v1.fill(1000);
  //  v0v1.zero();
  jusha::minus(v0, v1, v0v1);

  for (int i = 0; i != test_size; i++)
    REQUIRE(v0v1[i] == 0);
}


TEST_CASE( "VecResize", "[array]" ) {
  int size1 = 257;
  JVector<int> v0(size1);
  v0.fill(10);
  v0.setGpuArray();
  v0.resize(500);
  REQUIRE(v0[0] == 10);
  v0.resize(1000);
  REQUIRE(v0[size1-1] == 10);
}

TEST_CASE( "VecResize2", "[array]" ) {
  int size1 = 257;
  JVector<int> v0;
  v0.resize(size1);
}


