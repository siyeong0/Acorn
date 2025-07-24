#pragma once
#include <string>

#include <cuda_runtime.h>

namespace aco
{
	namespace gfx
	{
		class CudaRenderBuffer
		{
		public:
			CudaRenderBuffer();
			~CudaRenderBuffer();

			void Initialize(int width, int height, const cudaChannelFormatDesc& formatDesc);
			void InitializeWithExternalMemory(
				int width, int height, 
				cudaExternalMemoryHandleDesc cudaExtMemHandleDesc,
				const cudaChannelFormatDesc& formatDesc);
			void Release();

			cudaTextureObject_t GetTextureObject() const { return mTexture; }
			cudaSurfaceObject_t* GetDeviceSurfacePointer() const { return mDeviceSurface; }
			cudaArray_t GetArray() const { return mArray; }

		private:
			void createSurfaceAndTexture(cudaArray_t array);

		private:
			int mWidth;
			int mHeight;

			cudaMipmappedArray_t mMipArray = nullptr;
			cudaArray_t mArray = nullptr;
			cudaTextureObject_t mTexture = 0;
			cudaSurfaceObject_t mSurface = 0;
			cudaSurfaceObject_t* mDeviceSurface = nullptr;
			cudaExternalMemory_t mExternalMemory = nullptr;
		};
	}
}