#include "CudaRenderBuffer.h"

#include <stdexcept>
#include <cstring>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "Debug.h"

namespace aco
{
	namespace gfx
	{
		CudaRenderBuffer::CudaRenderBuffer()
			: mWidth(0)
			, mHeight(0)
			, mMipArray(nullptr)
			, mArray(nullptr)
			, mTexture(0)
			, mSurface(0)
			, mDeviceSurface(nullptr)
			, mExternalMemory(nullptr)
		{

		}

		CudaRenderBuffer::~CudaRenderBuffer()
		{
			Release();
		}

		void CudaRenderBuffer::Initialize(int width, int height, const cudaChannelFormatDesc& formatDesc)
		{
			mWidth = width;
			mHeight = height;
			cudaExtent extent = make_cudaExtent(width, height, 0);

			CUDA_CHECK(cudaMallocMipmappedArray(&mMipArray, &formatDesc, extent, 1));
			CUDA_CHECK(cudaGetMipmappedArrayLevel(&mArray, mMipArray, 0));

			createSurfaceAndTexture(mArray);
		}

		void CudaRenderBuffer::InitializeWithExternalMemory(int width, int height,
			cudaExternalMemoryHandleDesc cudaExtMemHandleDesc,
			const cudaChannelFormatDesc& formatDesc)
		{
			mWidth = width;
			mHeight = height;

			CUDA_CHECK(cudaImportExternalMemory(&mExternalMemory, &cudaExtMemHandleDesc));

			cudaExternalMemoryMipmappedArrayDesc mipDesc = {};
			mipDesc.offset = 0;
			mipDesc.extent = make_cudaExtent(width, height, 0);
			mipDesc.formatDesc = formatDesc;
			mipDesc.numLevels = 1;

			CUDA_CHECK(cudaExternalMemoryGetMappedMipmappedArray(&mMipArray, mExternalMemory, &mipDesc));
			CUDA_CHECK(cudaGetMipmappedArrayLevel(&mArray, mMipArray, 0));

			createSurfaceAndTexture(mArray);
		}

		void CudaRenderBuffer::Release()
		{
			if (mTexture)
				cudaDestroyTextureObject(mTexture);
			if (mSurface)
				cudaDestroySurfaceObject(mSurface);
			if (mDeviceSurface)
				cudaFree(mDeviceSurface);
			if (mMipArray && !mExternalMemory)
				cudaFreeMipmappedArray(mMipArray);
			if (mExternalMemory)
				cudaDestroyExternalMemory(mExternalMemory);

			mTexture = 0;
			mSurface = 0;
			mDeviceSurface = nullptr;
			mMipArray = nullptr;
			mArray = nullptr;
			mExternalMemory = nullptr;
		}

		void CudaRenderBuffer::createSurfaceAndTexture(cudaArray_t array)
		{
			cudaResourceDesc resDesc = {};
			resDesc.resType = cudaResourceTypeArray;
			resDesc.res.array.array = array;

			CUDA_CHECK(cudaCreateSurfaceObject(&mSurface, &resDesc));

			cudaTextureDesc texDesc = {};
			texDesc.normalizedCoords = true;
			texDesc.filterMode = cudaFilterModeLinear;
			texDesc.mipmapFilterMode = cudaFilterModeLinear;
			texDesc.addressMode[0] = cudaAddressModeWrap;
			texDesc.addressMode[1] = cudaAddressModeWrap;
			texDesc.readMode = cudaReadModeNormalizedFloat;

			cudaResourceDesc texRes = {};
			texRes.resType = cudaResourceTypeArray;
			texRes.res.array.array = array;

			CUDA_CHECK(cudaCreateTextureObject(&mTexture, &texRes, &texDesc, nullptr));

			CUDA_CHECK(cudaMalloc((void**)&mDeviceSurface, sizeof(cudaSurfaceObject_t)));
			CUDA_CHECK(cudaMemcpy(mDeviceSurface, &mSurface, sizeof(cudaSurfaceObject_t), cudaMemcpyHostToDevice));
		}
	}
}