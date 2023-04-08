#pragma once
#include <cassert>

#include "./cuda_config.h"

namespace jusha {
  namespace cuda {

    /*!*****************************************************************
     *                                 Reduction
     ******************************************************************/
    /* Warp level reduction, Only the first lane id gets the reduction */
    template <class T>
    __inline__ __device__
    T warpReduceSum(T val) {
      for (int offset = JC_cuda_warpsize/2; offset > 0; offset /= 2) 
        val += __shfl_down_sync(JC_cuda_full_warp_mask, val, offset);
      return val;
    }
    
    /* Warp level reduction, all lane id get the reduction */
    template <class T>
    __inline__ __device__
    T warpAllReduceSum(T val) {
      for (int mask = warpSize/2; mask > 0; mask /= 2) 
        val += __shfl_xor_sync(JC_cuda_full_warp_mask, val, mask);
      return val;
    }


    /* block reduce with provided shared memory
       shared memory should be greater than 32
     */
    template <class T>
    __inline__ __device__
    T blockReduceSum(T val, T *shared) {
      //      static __shared__ T shared[32]; // Shared mem for 32 partial sums
      int lane = threadIdx.x % warpSize;
      int wid = threadIdx.x / warpSize;
      
      val = warpReduceSum(val);     // Each warp performs partial reduction
      
      if (lane==0) shared[wid]=val; // Write reduced value to shared memory
      
      __syncthreads();              // Wait for all partial reductions

      //read from shared memory only if that warp existed
      // do it sequentially
      if (threadIdx.x == 0) {
        for (int i = 1; i < (blockDim.x + warpSize-1)/warpSize; i++)
          val += shared[i];
        shared[0] = val;
      }
      __syncthreads();
      if (threadIdx.x != 0)
        val = shared[0];
      // if (threadIdx.x == 0)
      //   printf("*** final val is %f.\n", shared[0]);
      return val;
    }

    /* block level reduction */
    template <class T>
    __inline__ __device__
    T blockReduceSum(T val) {
      assert(blockDim.x <= (32*32));
      static __shared__ T shared[32]; // Shared mem for 32 partial sums
      return blockReduceSum(val, shared);
    }


    /* block min with provided shared memory
       only the first thread returns the valid value
     */
    template <class T>
    __inline__ __device__
    T blockMinSum(T val, int id, T *shared_val, int *shared_id) {
      shared_val[threadIdx.x] = val;
      shared_id [threadIdx.x] = id;
      __syncthreads();
      for (int stride = blockDim.x/2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride)
          if (val > shared_val[threadIdx.x + stride]) {
            val = shared_val[threadIdx.x + stride];
            id = shared_id[threadIdx.x + stride];
            shared_val[threadIdx.x] = val;
            shared_id [threadIdx.x] = id;
          }
        __syncthreads();
      }
      return val;
    }

    /*!*****************************************************************
     *                                 Scan
     ******************************************************************/
    /*! Block level in-place scan over a range */
    template <class T, class Op, int BS, bool exclusive>
    __inline__ __device__
    void blockScan(T *start, T *end, T *scan_val) {
      //      static __shared__ T scan_val[BS];
      assert(blockDim.x == BS);
      Op op;
      int N = end - start;
      T carry_out = T();
      T outval;
      for (int id = threadIdx.x;  id < (N + BS-1)/BS * BS; id+=BS) {
        T val = T();
        if (id < N)
          val = start[id];
        if (threadIdx.x == 0)  {
          val = op(val, carry_out);
        }
        scan_val[threadIdx.x] = val;
        __syncthreads();
        if (threadIdx.x >=  1)  val = op(scan_val[threadIdx.x], scan_val[threadIdx.x -  1]); __syncthreads();
        if (threadIdx.x >=  1)  scan_val[threadIdx.x] = val; __syncthreads();

        if (threadIdx.x >=  2)  val = op(scan_val[threadIdx.x], scan_val[threadIdx.x -  2]); __syncthreads();
        if (threadIdx.x >=  2)  scan_val[threadIdx.x] = val; __syncthreads();

        if (threadIdx.x >=  4)  val = op(scan_val[threadIdx.x], scan_val[threadIdx.x -  4]); __syncthreads();
        if (threadIdx.x >=  4)  scan_val[threadIdx.x] = val; __syncthreads();

        if (threadIdx.x >=  8)  val = op(scan_val[threadIdx.x], scan_val[threadIdx.x -  8]); __syncthreads();
        if (threadIdx.x >=  8)  scan_val[threadIdx.x] = val; __syncthreads();

        if (threadIdx.x >=  16)  val = op(scan_val[threadIdx.x], scan_val[threadIdx.x -  16]); __syncthreads();
        if (threadIdx.x >=  16)  scan_val[threadIdx.x] = val; __syncthreads();

        if (threadIdx.x >=  32)  val = op(scan_val[threadIdx.x], scan_val[threadIdx.x -  32]); __syncthreads();
        if (threadIdx.x >=  32)  scan_val[threadIdx.x] = val; __syncthreads();

        if (threadIdx.x >=  64)  val = op(scan_val[threadIdx.x], scan_val[threadIdx.x -  64]); __syncthreads();
        if (threadIdx.x >=  64)  scan_val[threadIdx.x] = val; __syncthreads();

        if (threadIdx.x >=  128)  val = op(scan_val[threadIdx.x], scan_val[threadIdx.x -  128]); __syncthreads();
        if (threadIdx.x >=  128)  scan_val[threadIdx.x] = val; __syncthreads();

        if (threadIdx.x >=  256)  val = op(scan_val[threadIdx.x], scan_val[threadIdx.x -  256]); __syncthreads();
        if (threadIdx.x >=  256)  scan_val[threadIdx.x] = val; __syncthreads();

        if (threadIdx.x >=  512)  val = op(scan_val[threadIdx.x], scan_val[threadIdx.x -  512]); __syncthreads();
        if (threadIdx.x >=  512)  scan_val[threadIdx.x] = val; __syncthreads();
        if (!exclusive)
          outval = scan_val[threadIdx.x];
        else {
          if (threadIdx.x == 0) {
            outval = carry_out;
          }
          else
            outval = scan_val[threadIdx.x-1];
        }
        carry_out = scan_val[BS-1];

        if (id < N) {
          start[id] = outval;
        }
        
      }
    }


    /*! Block level in-place scan over a range */
    template <class T, class Op, int BS, bool exclusive>
    __inline__ __device__
    void blockScan(T *start, T *end) {
      static __shared__ T scan_val[BS];
      blockScan<T, Op, BS, exclusive>(start, end, scan_val);
    }

    /*********** sort *
     */

  } // cuda
} // jusha
