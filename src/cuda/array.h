#pragma once

#include <cuda_runtime_api.h>
#include <iostream>
#include <fstream>
#include <string>

#include <thrust/device_ptr.h>
#include <curand.h>
#include "../utility.h"
#include "./external_lib_wrapper.h"
#include "./heap_manager.h"
// #define USE_SHARED_PTR 1
#include <cstddef>
// #include <memory>

// extern cudaDeviceProp gDevProp;

namespace jusha
{
  extern HeapManager gHeapManager;
  namespace cuda
  {
#define MAX_PRINT_SIZE 32

    template <typename T>
    void fill(T *begin, T *end, const T &val);

    template <typename T>
    void fill(thrust::device_ptr<T> begin, thrust::device_ptr<T> end, const T &val);

    enum class ArrayType
    {
      CPU_ARRAY = 0,
      GPU_ARRAY = 1
    };

    template <class T>
    class MirroredArray
    {
    public:
      explicit MirroredArray(int size = 0) : mSize(size),
                                             mCapacity(size),
                                             mHostBase(),
                                             mDvceBase(),
                                             mIsCpuValid(false),
                                             mIsGpuValid(false),
                                             mGpuAllocated(false),
                                             mCpuAllocated(false),
                                             mIsGpuArray(false),
                                             gpuNeedToFree(true),
                                             cpuNeedToFree(true)
      {
      }

      ~MirroredArray()
      {
        destroy();
      }

      void SetGpuArray()
      {
        mIsGpuArray = true;
      }

      void SetCpuArray()
      {
        mIsGpuArray = false;
      }

      bool IsGpuArray() const
      {
        return mIsGpuArray;
      }

      void destroy()
      {
        if (mDvceBase && gpuNeedToFree)
        {
#ifdef USE_SHARED_PTR
          gHeapManager.NeFree(GPU_HEAP, mDvceBase.get());
#else
          gHeapManager.NeFree(GPU_HEAP, mDvceBase, size() * sizeof(T));
#endif
          mDvceBase = NULL;
        }
        if (mHostBase && mCapacity > 0 && cpuNeedToFree)
        {
          gHeapManager.NeFree(CPU_HEAP, mHostBase, size() * sizeof(T));
          mHostBase = NULL;
        }
        init_state();
      }
      // Copy Constructor
      MirroredArray(const MirroredArray<T> &rhs)
      {
        init_state();
        deep_copy(rhs);
      }

      explicit MirroredArray(const std::vector<T> &rhs)
      {
        init_state();
        clean_resize(rhs.size());
        cudaMemcpy(getOverwriteGpuPtr(), rhs.data(), size() * sizeof(T), cudaMemcpyDefault);
      }

      void operator=(const std::vector<T> &rhs)
      {
        //        init_state();
        clean_resize(rhs.size());
        cudaMemcpy(getOverwriteGpuPtr(), rhs.data(), size() * sizeof(T), cudaMemcpyDefault);
      }

      /* Init from raw pointers
       */
      void init(const T *ptr, size_t _size)
      {
        clean_resize(_size);
        T *g_ptr = getGpuPtr();
        cudaMemcpy(g_ptr, ptr, sizeof(T) * _size, cudaMemcpyDefault);
      }
      // dangerous, used at your own risk!
      // does not check size consistency
      void setGpuPtr(T *ptr, int _size, bool needToFree = false)
      {
#if USE_SHARED_PTR
        if (!needToFree)
        {
          std::shared_ptr<T> newmDvceBase((T *)ptr, EmptyDeviceDeleter);
          mDvceBase = newmDvceBase;
        }
        else
        {
          std::shared_ptr<T> newmDvceBase((T *)ptr, GpuDeviceDeleter);
          mDvceBase = newmDvceBase;
        }
#else
        if (mDvceBase && gpuNeedToFree)
          gHeapManager.NeFree(GPU_HEAP, mDvceBase, sizeof(T) * mSize);
        mDvceBase = ptr;
#endif
        //        mIsCpuValid = false;
        mSize = _size;
        mIsGpuValid = true;
        gpuNeedToFree = needToFree;
        mGpuAllocated = true;
      }

      void setPtr(T *ptr, int _size)
      {
#if USE_SHARED_PTR
        mHostBase.reset(ptr);
#else
        if (mHostBase && mCapacity >= 0 && cpuNeedToFree)
          gHeapManager.NeFree(CPU_HEAP, mHostBase, sizeof(T) * mSize);
        mHostBase = ptr;
#endif
        mSize = _size;
        mIsCpuValid = true;
        mCpuAllocated = true;
        cpuNeedToFree = false;
        //        mCapacity = -1; // to disable calling free
      }

      MirroredArray<T> &operator=(const MirroredArray<T> &rhs)
      {
        deep_copy(rhs);
        return *this;
      }

      // deep copy from
      void deep_copy(const MirroredArray<T> &src)
      {
        clean_resize(src.size());
        //        printf("deep copy src gpuvalid %d my gpuvalid %d gpu alloc %d.\n", src.mIsGpuValid, mIsGpuValid, mGpuAllocated);
        if (src.mIsGpuValid)
        {
          if (src.size())
            cudaMemcpy(getGpuPtr(), src.getReadOnlyGpuPtr(), sizeof(T) * mSize, cudaMemcpyDeviceToDevice);
          //            printf("deep copy src gpuvalid %d my gpuvalid %d %p size %zd.\n", src.mIsGpuValid, mIsGpuValid,mDvceBase, src.size());
        }
        else if (src.mIsCpuValid)
        {
          if (src.size())
            memcpy(getPtr(), src.getReadOnlyPtr(), sizeof(T) * mSize);
        }
        mIsGpuArray = src.mIsGpuArray;
      }

      bool GpuHasLatest() const
      {
        return mIsGpuValid;
      }

      bool CpuHasLatest() const
      {
        return mIsCpuValid;
      }

      // deep copy to
      void clone(MirroredArray<T> &dst) const
      {
        dst.clean_resize(size());
        if (mIsGpuValid)
        {
          cudaMemcpy(dst.getGpuPtr(), getReadOnlyGpuPtr(), sizeof(T) * mSize, cudaMemcpyDeviceToDevice);
        }
        if (mIsCpuValid)
        {
          memcpy(dst.getPtr(), getReadOnlyPtr(), sizeof(T) * mSize);
        }

        dst.mIsCpuValid = mIsCpuValid;
        dst.mIsGpuValid = mIsGpuValid;
        dst.mGpuAllocated = mGpuAllocated;
        dst.mCpuAllocated = mCpuAllocated;
      }

      void alias(const MirroredArray<T> &dst)
      {
        shallow_copy(dst);
        mCapacity = -1; // to disable calling free
      }

      void clear()
      {
        resize(0);
      }

      /* swap info between two arrays */
      void swap(MirroredArray<T> &rhs)
      {
        MirroredArray<T> temp(rhs);
        //    temp.clone(rhs);
        rhs = *this;
        *this = temp;
        //    temp.mHostBase = 0;
        //    temp.mDvceBase = 0;
        temp.mCpuAllocated = false;
        temp.mGpuAllocated = false;
      }

      int size() const
      {
        return mSize;
      }

      /*! A clean version of resize.
        It does not copy the old data,
        nor does it initialize the data
      */
      void clean_resize(int64_t _size)
      {
        if (mCapacity >= _size || _size == 0)
        {
          mSize = _size;
          mIsGpuValid = false;
          mIsCpuValid = false;
          return;
        }
        if (_size > 0)
        {
          // depending on previous state
          if (mGpuAllocated)
          {
            if (mDvceBase)
              gHeapManager.NeFree(GPU_HEAP, mDvceBase, mCapacity * sizeof(T));
            gHeapManager.NeMalloc(GPU_HEAP, (void **)&mDvceBase, _size * sizeof(T));
            mIsGpuValid = false;
          }
          if (mCpuAllocated)
          {
            if (mHostBase)
              gHeapManager.NeFree(CPU_HEAP, mHostBase, mCapacity * sizeof(T));
            gHeapManager.NeMalloc(CPU_HEAP, (void **)&mHostBase, _size * sizeof(T));
            mIsCpuValid = false;
          }
          mSize = _size;
          mCapacity = _size;
        }
      }

      void resize(int64_t _size)
      {
#ifdef _DEBUG_
        std::cout << "new size " << _size << " old size " << mSize << std::endl;
#endif
        if (_size <= mCapacity)
        {
          // free memory if resize to zero
          if (_size == 0 && mSize > 0)
          {
            destroy();
          }
          mSize = _size;
        }
        else // need to reallocate
        {
#if USE_SHARED_PTR
          if (mGpuAllocated)
          {
            std::shared_ptr<T> newmDvceBase((T *)GpuDeviceAllocator(_size * sizeof(T)), GpuDeviceDeleter);
            if (mIsGpuValid)
            {
              cudaError_t error = cudaMemcpy(newmDvceBase, mDvceBase, mSize * sizeof(T), cudaMemcpyDeviceToDevice);
              //            std::cout << "memcpy d2d size:" << mSize*sizeof(T)  << std::endl;
              assert(error == cudaSuccess);
            }
            mDvceBase = newmDvceBase;
          }
          if (mCpuAllocated)
          {
            std::shared_ptr<T> newmHostBase((T *)GpuHostAllocator(_size * sizeof(T)), GpuHostDeleter);
            if (mIsCpuValid)
              memcpy(newmHostBase, mHostBase, mSize * sizeof(T));
            mHostBase = newmHostBase;
          }
          mSize = _size;
          mCapacity = _size;
#else
          T *newmDvceBase(0);
          T *newmHostBase(0);
          if (!mGpuAllocated && !mCpuAllocated)
          {
            if (mIsGpuArray)
            {
              gHeapManager.NeMalloc(GPU_HEAP, (void **)&newmDvceBase, _size * sizeof(T));
              assert(newmDvceBase);
            }
            else
            {
              gHeapManager.NeMalloc(CPU_HEAP, (void **)&newmHostBase, _size * sizeof(T));
              //            newmHostBase = (T*)malloc(size * sizeof(T));
              assert(newmHostBase);
            }
          }
          if (mGpuAllocated)
          {
            // cutilSafeCall(cudaMalloc((void**) &newmDvceBase, size * _sizeof(T)));
            gHeapManager.NeMalloc(GPU_HEAP, (void **)&newmDvceBase, _size * sizeof(T));
            assert(newmDvceBase);
            // TODO memcpy
          }
          if (mCpuAllocated)
          {
            gHeapManager.NeMalloc(CPU_HEAP, (void **)&newmHostBase, _size * sizeof(T));
            //            newmHostBase = (T*)malloc(size * sizeof(T));
            assert(newmHostBase);
          }
          if (mIsCpuValid && mCpuAllocated)
          {
            memcpy(newmHostBase, mHostBase, mSize * sizeof(T));
          }
          if (mIsGpuValid && mGpuAllocated)
          {
            cudaError_t error = cudaMemcpy(newmDvceBase, mDvceBase, mSize * sizeof(T), cudaMemcpyDeviceToDevice);
            jassert(error == cudaSuccess);
          }
          if (mHostBase && mCapacity > 0 && cpuNeedToFree)
            gHeapManager.NeFree(CPU_HEAP, mHostBase, sizeof(T) * mSize);
          //          free(mHostBase);
          if (mDvceBase && gpuNeedToFree)
          {
            gHeapManager.NeFree(GPU_HEAP, mDvceBase, sizeof(T) * mSize);
            //            cutilSafeCall(cudaFree(mDvceBase));
          }
#ifdef _DEBUG_
          std::cout << "free at resize:" << std::hex << mDvceBase << std::endl;
#endif
          mHostBase = newmHostBase;
          mDvceBase = newmDvceBase;
          // if (mHostBase)
          //   std::fill(mHostBase+mSize, mHostBase + _size, T());
          // if (mDvceBase)
          //   jusha::cuda::fill(mDvceBase + mSize, mDvceBase + _size, T());
          mSize = _size;
          mGpuAllocated = mDvceBase == 0 ? false : true;
          mCpuAllocated = mHostBase == 0 ? false : true;
          mCapacity = _size;
#endif
        }
      }

      void zero()
      {
        if (mIsGpuArray)
        {
          cudaMemset((void *)getOverwriteGpuPtr(), 0, sizeof(T) * mSize);
          check_cuda_error("after cudaMemset", __FILE__, __LINE__);
        }
        else
        {
          memset((void *)getOverwritePtr(), 0, sizeof(T) * mSize);
        }
      }

      /*! return the pointer without changing the internal state */
      T *getRawPtr()
      {
        allocateCpuIfNecessary();
        return mHostBase;
      }

      /*! return the gpu pointer without changing the internal state */
      T *getRawGpuPtr()
      {
        allocateGpuIfNecessary();
        return mDvceBase;
      }

      const T *getReadOnlyPtr() const
      {
        enableCpuRead();
        return mHostBase;
      }

      T *getPtr()
      {
        enableCpuWrite();
        return mHostBase;
      }

      const T *getReadOnlyGpuPtr() const
      {
        enableGpuRead();
        return mDvceBase;
      }

      T *getGpuPtr()
      {
        enableGpuWrite();
        return mDvceBase;
      }

      T *getOverwritePtr()
      {
        allocateCpuIfNecessary();
        mIsCpuValid = true;
        mIsGpuValid = false;
        return mHostBase;
      }

      T *getOverwriteGpuPtr()
      {
        allocateGpuIfNecessary();
        mIsCpuValid = false;
        mIsGpuValid = true;
        return mDvceBase;
      }

      T &operator[](int index)
      {
        assert(index < mSize);
        T *host = getPtr();
        return host[index];
      }

      const T &operator[](int index) const
      {
        assert(index < mSize);
        const T *host = getReadOnlyPtr();
        return host[index];
      }

      // friend
      // MirroredArray<T> &operator-(const MirroredArray<T> &lhs, const MirroredArray<T> &rhs);

      /* only dma what's needed, instead of the whole array */
      const T getElementAt(const int index) const
      {
        assert(index < mSize);
        //        printf("cpu valid %d gpu valid %d.\n", mIsCpuValid, mIsGpuValid);
        assert(mIsCpuValid || mIsGpuValid);
        if (mIsCpuValid)
          //          return mHostBase[index];
          return mHostBase[index];
        T ele;
        allocateCpuIfNecessary();
        //        cudaError_t error = cudaMemcpy(&ele, mDvceBase+index, sizeof(T),cudaMemcpyDeviceToHost);
        //        printf("calling cudamemcpy \n");
        cudaError_t error = cudaMemcpy(&ele, mDvceBase + index, sizeof(T), cudaMemcpyDeviceToHost);
        //    std::cout << "memcpy d2h size:" << sizeof(T)  << std::endl;
        jassert(error == cudaSuccess);
        return ele;
      }

      void setElementAt(T &value, const int index)
      {
        jassert(index < mSize);
        jassert(mIsCpuValid || mIsGpuValid);
        if (mIsCpuValid)
          //          mHostBase[index] = value;
          mHostBase[index] = value;
        if (mIsGpuValid)
        {
          //            cudaError_t error = cudaMemcpy(mDvceBase+index, &value, sizeof(T), cudaMemcpyHostToDevice);
          cudaError_t error = cudaMemcpy(mDvceBase + index, &value, sizeof(T), cudaMemcpyHostToDevice);
          jassert(error == cudaSuccess);
        }
      }

      void randomize()
      {
        RandomWrapper<CURAND_RNG_PSEUDO_MTGP32, T> rng;
        rng.apply(getOverwriteGpuPtr(), mSize);
      }

      // scale the array
      void scale(const T &ratio);

      // set the array to the same value
      void fill(const T &val)
      {
        if (mIsGpuArray)
        {
          //      if (true) {
          jusha::cuda::fill(owbegin(), owend(), val);
          check_cuda_error("array fill", __FILE__, __LINE__);
        }
        else
        {
          std::fill(getOverwritePtr(), getOverwritePtr() + size(), val);
        }
      }

      // use sequence in thrust
      void sequence(int dir)
      {
        T *ptr = getPtr();
        if (dir == 0) // ascending
        {
          for (int i = 0; i != mSize; i++)
            *ptr++ = (T)i;
        }
        else // descending
        {
          for (int i = 0; i != mSize; i++)
            *ptr++ = (T)(mSize - i);
        }
      }

      // for DEBUG purpose
      void print(const char *header = 0, int print_size = MAX_PRINT_SIZE) const
      {
        const T *ptr = getReadOnlyPtr();
        int size = mSize > print_size ? print_size : mSize;
        if (header)
          std::cout << header << std::endl;
        for (int i = 0; i != size; i++)
          std::cout << " " << ptr[i];

        std::cout << std::endl;
      }

      /* math functions on mirrored array */
      /*  void reduce(CudppPlanFactory *factory, MirroredArray<T> &total, uint op)
          {
          // todo: test a threshold to determine whether do on CPU or GPU
          // currently do it on GPUs always

          }*/

      void saveToFile(const char *filename) const
      {
        //      assert(0);
        std::ofstream file;
        file.open(filename);
        assert(file);
        file << "size is " << size() << "\n";
        //      int size = printSize < size()? printSize:size();
        const T *ptr = getReadOnlyPtr();
        for (int i = 0; i != size(); i++)
          file << ptr[i] << "(" << i << ")"
               << " ";
        file.close();
      }

      bool isSubsetOf(const MirroredArray<T> &super)
      {
        const T *myBase = getReadOnlyPtr();
        const T *superBase = super.getReadOnlyPtr();
        size_t mySize = size();
        size_t superSize = super.size();
        if (mySize > superSize)
          return false;

        for (int i = 0; i != mySize; i++)
        {
          bool found = false;
          for (int j = 0; j != superSize; j++)
          {
            if (superBase[j] == myBase[i])
            {
              //                std::cout << "found " << myBase[i];
              found = true;
            }
          }
          if (!found)
          {
            std::cout << "not finding " << myBase[i] << " in super array.\n";
            return false;
          }
        }
        return true;
      }

      bool isAllZero() const
      {
        const T *buffer = getReadOnlyPtr();
        bool allzero = true;
        for (int i = 0; i < mSize; i++)
        {
          if (buffer[i] != 0)
          {
            std::cout << "the " << i << "th value " << buffer[i] << " is not zero " << std::endl;
            allzero = false;
            break;
          }
        }
        return allzero;
      }

      bool isEqualTo(const MirroredArray<T> &rhs) const
      {
        if (rhs.size() != size())
          return false;
        const T *buffer = getReadOnlyPtr();
        const T *buffer2 = rhs.getReadOnlyPtr();
        bool equal = true;
        for (int i = 0; i < mSize; i++)
        {
          if (buffer[i] != buffer2[i])
          {
            equal = false;
            break;
          }
        }
        return equal;
      }

      bool isFSorted(int begin, int end) const
      {
        const T *buffer = getReadOnlyPtr();
        bool sorted = true;
        if (begin == -1)
          begin = 0;
        if (end == -1)
          end = mSize;
        for (int i = begin; i < end - 1; i++)
        {
          if (buffer[i] > buffer[i + 1])
          {
            std::cout << "the " << i << "th value " << buffer[i] << " is bigger than " << buffer[i + 1] << std::endl;
            sorted = false;
            break;
          }
        }
        return sorted;
      }

      void invalidateGpu()
      {
        mIsGpuValid = false;
      }

      void invalidateCpu()
      {
        mIsCpuValid = false;
      }

      inline typename thrust::device_ptr<T> gbegin()
      //{ return thrust::retag<srt_thrust_tag>(thrust::device_ptr<T>(data_pointer)); }
      {
        enableGpuWrite();
        return thrust::device_ptr<T>(getGpuPtr());
      }

      /*! \brief Return the last iterator (the first invalid iterator) in the srt::vector */
      inline typename thrust::device_ptr<T> gend()
      //{ return thrust::retag<srt_thrust_tag>(thrust::device_ptr<T>(data_pointer+m_size)); }
      {
        enableGpuWrite();
        return thrust::device_ptr<T>(getGpuPtr() + mSize);
      }

      inline typename thrust::device_ptr<T> owbegin()
      //{ return thrust::retag<srt_thrust_tag>(thrust::device_ptr<T>(data_pointer)); }
      {
        return thrust::device_ptr<T>(getOverwriteGpuPtr());
      }

      /*! \brief Return the last iterator (the first invalid iterator) in the srt::vector */
      inline typename thrust::device_ptr<T> owend()
      //{ return thrust::retag<srt_thrust_tag>(thrust::device_ptr<T>(data_pointer+m_size)); }
      {
        return thrust::device_ptr<T>(getOverwriteGpuPtr() + mSize);
      }

      /*! \brief Return the iterator to the first element in the srt::vector */
      inline typename thrust::device_ptr<T> gbegin() const
      //{ return thrust::retag<srt_thrust_tag>(thrust::device_ptr<T>(data_pointer)); }
      {
        enableGpuRead();
        return thrust::device_ptr<T>(const_cast<T *>(getReadOnlyGpuPtr()));
      }

      /*! \brief Return the last iterator (the first invalid iterator) in the srt::vector */
      inline typename thrust::device_ptr<T> gend() const
      //{ return thrust::retag<srt_thrust_tag>(thrust::device_ptr<T>(data_pointer+m_size)); }
      {
        enableGpuRead();
        return thrust::device_ptr<T>(const_cast<T *>(getReadOnlyGpuPtr() + mSize));
      }

      /*! explicitly sync to GPU buffer */
      void syncToGpu() const
      {
        //	assert(!(mIsGpuValid && !mIsCpuValid));
        allocateGpuIfNecessary();
        fromHostToDvce();
        mIsGpuValid = true;
      }

      /*! explicitly sync to CPU buffer */
      void syncToCpu() const
      {
        //	assert(!(mIsCpuValid && !mIsGpuValid));
        allocateCpuIfNecessary();
        fromDvceToHost();
        mIsCpuValid = true;
      }

      inline void enableGpuRead() const
      {
        allocateGpuIfNecessary();
        if (!mIsGpuValid)
        {
          fromHostToDvceIfNecessary();
          setGpuAvailable();
        }
      }

      inline void enableGpuWrite() const
      {
        allocateGpuIfNecessary();
        if (!mIsGpuValid)
          fromHostToDvceIfNecessary();

        mIsCpuValid = false;
        mIsGpuValid = true;
      }

      inline void enableCpuRead() const
      {
        allocateCpuIfNecessary();
        if (!mIsCpuValid)
        {
          fromDvceToHostIfNecessary();
          mIsCpuValid = true;
        }
      }

      inline void enableCpuWrite() const
      {
        allocateCpuIfNecessary();
        if (!mIsCpuValid)
          fromDvceToHostIfNecessary();

        mIsCpuValid = true;
        mIsGpuValid = false;
      }

      void setGpuAvailable() const
      {
        mIsGpuValid = true;
      }

    private:
      void init_state()
      {
        mSize = 0;
        mCapacity = 0;
        mHostBase = nullptr;
        mDvceBase = nullptr;
        mIsCpuValid = false;
        mIsGpuValid = false;
        mGpuAllocated = false;
        mCpuAllocated = false;
        mIsGpuArray = false;
      }

      inline void allocateCpuIfNecessary() const
      {
        if (!mCpuAllocated && mSize)
        {
#if USE_SHARED_PTR
          std::shared_ptr<T> newmHostBase((T *)GpuHostAllocator(mSize * sizeof(T)), GpuHostDeleter);
          mHostBase = newmHostBase;
#else
          gHeapManager.NeMalloc(CPU_HEAP, (void **)&mHostBase, mSize * sizeof(T));
          assert(mHostBase);
#endif
          //        mHostBase = (T *)malloc(mSize * sizeof(T));

          mCpuAllocated = true;
        }
      }

      inline void allocateGpuIfNecessary() const
      {
        if (!mGpuAllocated && mSize)
        {
          //        cutilSafeCall(cudaMalloc((void**) &mDvceBase, mSize * sizeof(T)));
#if USE_SHARED_PTR
          std::shared_ptr<T> newmDvceBase((T *)GpuDeviceAllocator(mSize * sizeof(T)), GpuDeviceDeleter);
          assert(newmDvceBase != 0);
          mDvceBase = newmDvceBase;
#else
          gHeapManager.NeMalloc(GPU_HEAP, (void **)&mDvceBase, mSize * sizeof(T));
          assert(mDvceBase);
#endif
          mGpuAllocated = true;
        }
      }

      inline void fromHostToDvce() const
      {
        if (mSize)
        {
          jassert(mHostBase);
          jassert(mDvceBase);
          cudaError_t error = cudaMemcpy(mDvceBase, mHostBase, mSize * sizeof(T), cudaMemcpyHostToDevice);
          //        std::cout << "memcpy h2d size:" << mSize*sizeof(T)  << std::endl;
          jassert(error == cudaSuccess);
        }
      }

      inline void fromHostToDvceIfNecessary() const
      {
        if (mIsCpuValid && !mIsGpuValid)
        {
#ifdef _DEBUG_
          std::cout << "sync mirror array from host 0x" << std::hex << mHostBase << " to device 0x" << mDvceBase << " size(" << mSize << "); \n";
#endif
          fromHostToDvce();
          //            cudaError_t error = cudaMemcpy(mDvceBase, mHostBase, mSize* sizeof(T), cudaMemcpyHostToDevice);
        }
      }

      inline void fromDvceToHost() const
      {
        if (mSize)
        {
          jassert(mHostBase);
          jassert(mDvceBase);
          cudaError_t error = cudaMemcpy(mHostBase, mDvceBase, mSize * sizeof(T), cudaMemcpyDeviceToHost);
          jassert(error == cudaSuccess);
        }
      }

      inline void fromDvceToHostIfNecessary() const
      {
        if (mIsGpuValid && !mIsCpuValid)
        {
          //            check_cuda_error("before memcpy", __FILE__, __LINE__);
          if (size())
          {
            jassert(mDvceBase);
            jassert(mHostBase);
          }
#ifdef _DEBUG_
          std::cout << "sync mirror array from device 0x" << std::hex << mDvceBase << " to host 0x" << mHostBase << " size(" << mSize << "); \n";
          /* assert(gHeapManager.find(CPU_HEAP, mHostBase) >= (mSize * (int)sizeof(T))); */
          /* assert(gHeapManager.find(GPU_HEAP, mDvceBase) >= (mSize * (int)sizeof(T))); */
          assert(gHeapManager.find(CPU_HEAP, mHostBase) >= (mSize * (int)sizeof(T)));
          assert(gHeapManager.find(GPU_HEAP, mDvceBase) >= (mSize * (int)sizeof(T)));
#endif
          //            cudaError_t error = cudaMemcpy(mHostBase, mDvceBase, mSize * sizeof(T),cudaMemcpyDeviceToHost);
          fromDvceToHost();
        }
      }

      int64_t mSize;
      int mCapacity;
#if USE_SHARED_PTR
      std::shared_ptr<T> mHostBase;
      std::shared_ptr<T> mDvceBase;
#else
      mutable T *mHostBase = 0;
      mutable T *mDvceBase = 0;
#endif
      mutable bool mIsCpuValid;
      mutable bool mIsGpuValid;
      mutable bool mGpuAllocated;
      mutable bool mCpuAllocated;
      mutable bool mIsGpuArray;
      mutable bool gpuNeedToFree = true;
      mutable bool cpuNeedToFree = true;

      void shallow_copy(const MirroredArray<T> &rhs)
      {
        mSize = rhs.mSize;
        mCapacity = rhs.mCapacity;
        mHostBase = rhs.mHostBase;
        mDvceBase = rhs.mDvceBase;
        mIsCpuValid = rhs.mIsCpuValid;
        mIsGpuValid = rhs.mIsGpuValid;
        mGpuAllocated = rhs.mGpuAllocated;
        mCpuAllocated = rhs.mCpuAllocated;
        if (mIsGpuValid)
          assert(mGpuAllocated);
        if (mIsCpuValid)
          assert(mCpuAllocated);
      }

      static curandGenerator_t curandGen;
    };

    template <typename T, int BATCH>
    struct BatchInit
    {
      T *ptrs[BATCH];
      size_t sizes[BATCH];
      T vals[BATCH];

      T *big_ptrs[BATCH];
      size_t big_sizes[BATCH];
      T vals2[BATCH];
    };

    template <typename T, int BATCH>
    void batch_fill_wrapper(int num_small_arrays, int num_big_arrays, const BatchInit<T, BATCH> &init, cudaStream_t stream);

    /*! Help class to initialize multiple vectors at the same time
     *
     */
    template <class T, int BATCH>
    class BatchInitializer
    {
    public:
      void push_back(MirroredArray<T> *array, T val)
      {
        m_arrays.push_back(array);
        m_vals.push_back(val);
        assert(m_arrays.size() < BATCH);
      }
      void init(cudaStream_t stream = 0)
      {
        BatchInit<T, BATCH> init;
        memset(&init, 0, sizeof(init));
        if (m_arrays.size() > BATCH)
          std::cerr << "Number of arrays " << m_arrays.size() << " exceeding template BATCH " << BATCH << ", please increase BATCH." << std::endl;
        int small_idx = 0, big_idx = 0;
        for (int i = 0; i != m_arrays.size(); i++)
        {
          size_t _size = m_arrays[i]->size();
          if (_size < 100000)
          {
            init.ptrs[small_idx] = m_arrays[i]->getOverwriteGpuPtr();
            init.sizes[small_idx] = _size;
            init.vals[small_idx] = m_vals[i];
            ++small_idx;
          }
          else
          {
            init.big_ptrs[big_idx] = m_arrays[i]->getOverwriteGpuPtr();
            init.big_sizes[big_idx] = _size;
            init.vals2[big_idx] = m_vals[i];
            ++big_idx;
          }
        }
        batch_fill_wrapper<T, BATCH>(small_idx, big_idx, init, stream);
      }

    private:
      std::vector<MirroredArray<T> *> m_arrays;
      std::vector<T> m_vals;
    };
  } // cuda

// aliasing C++11 feature
/*  template <typename T>
    using JVector = cuda::MirroredArray<T>;*/
#define JVector jusha::cuda::MirroredArray

  /* array operations */

  // y = x0 * x1
  template <class T>
  void multiply(const JVector<T> &x0, const JVector<T> &x1, JVector<T> &y);

  // norm
  template <class T>
  T norm(const JVector<T> &vec);

  template <class T>
  void addConst(JVector<T> &vec, T val);

} // jusha

/*
template <class T>
__global__
void  arrayMinusKernel(T *dst, const T * lhs, const T *rhs, int size)
{
  GET_GID
  OUTER_FOR
  {
    dst[curId] = lhs[curId] - rhs[curId];
  }

}
*/
/*template <class T>
MirroredArray<T> &operator-(const MirroredArray<T> &lhs, const MirroredArray<T> &rhs)
{
  MirroredArray<T> result(lhs.size());
  assert(lhs.size() == rhs.size());
  int size = lhs.size();
  cudaDeviceProp *devProp = &gDevProp;
  arrayMinusKernel<<<KERNEL_SETUP(size)>>>(result.getGpuPtr(), lhs.getReadOnlyGpuPtr(), rhs.getReadOnlyGpuPtr(), size);
  return result;
  }*/
