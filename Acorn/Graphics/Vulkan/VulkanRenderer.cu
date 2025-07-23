#include "VulkanRenderer.h"

#include "../Helper/helper_math.h" // copied from cuda_sample

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace aco
{
	namespace gfx
	{
		// convert floating point rgba color to 32-bit integer
		__device__ unsigned int rgbaFloatToInt(float4 rgba)
		{
			rgba.x = __saturatef(rgba.x); // clamp to [0.0, 1.0]
			rgba.y = __saturatef(rgba.y);
			rgba.z = __saturatef(rgba.z);
			rgba.w = __saturatef(rgba.w);
			return ((unsigned int)(rgba.w * 255.0f) << 24) | ((unsigned int)(rgba.z * 255.0f) << 16) |
				((unsigned int)(rgba.y * 255.0f) << 8) | ((unsigned int)(rgba.x * 255.0f));
		}

		__device__ float4 rgbaIntToFloat(unsigned int c)
		{
			float4 rgba;
			rgba.x = (c & 0xff) * 0.003921568627f;         //  /255.0f;
			rgba.y = ((c >> 8) & 0xff) * 0.003921568627f;  //  /255.0f;
			rgba.z = ((c >> 16) & 0xff) * 0.003921568627f; //  /255.0f;
			rgba.w = ((c >> 24) & 0xff) * 0.003921568627f; //  /255.0f;
			return rgba;
		}

		// column pass using coalesced global memory reads
		__global__ void copyKernel(
			cudaSurfaceObject_t* dstSurfMipMapArray,
			cudaTextureObject_t textureMipMapInput,
			size_t baseWidth, size_t baseHeight)
		{
			unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
			unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

			if (x < baseWidth && y < baseHeight)
			{

				float4 t = tex2DLod<float4>(textureMipMapInput, float(x) / baseWidth, float(y) / baseHeight, 0);

				unsigned int dataB = rgbaFloatToInt(t);
				surf2Dwrite(dataB, dstSurfMipMapArray[0], 4 * x, y);

				// x 앞에 4 곱하는 이유
				// 3.2.14.2.1. Surface Object API
				// https://docs.nvidia.com/cuda/cuda-c-programming-guide/#surface-object-api
			}
		}

		__global__ void paintRedKernel(
			cudaSurfaceObject_t* dstSurface,
			size_t baseWidth, size_t baseHeight)
		{
			unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
			unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

			if (x >= baseWidth || y >= baseHeight) return;

			float4 redColor = make_float4(1.0f, 0.0f, 0.0f, 1.0f);

			unsigned int uiRedColor = rgbaFloatToInt(redColor);
			surf2Dwrite(uiRedColor, dstSurface[0], 4 * x, y);
		}

		void VulkanRenderer::cudaUpdateVkImage()
		{
			cudaVkSemaphoreWait(mCudaExtVkUpdateCudaSemaphore);

			int nthreads = 16;
			dim3 dimBlock(nthreads, nthreads, 1); // dimBlock.x * dimBlock.y * dimBlock.z <= 1024
			dim3 dimGrid(
				(uint32_t)ceil(float(WIDTH) / dimBlock.x),
				(uint32_t)ceil(float(HEIGHT) / dimBlock.y),
				1);

			copyKernel << <dimGrid, dimBlock >> > (
				dev_mSurfaceObjectList,
				mTextureObjMipMapInput,
				WIDTH, HEIGHT);

			paintRedKernel << <dimGrid, dimBlock, 0, mStreamToRun>> > (
				dev_mSurfaceObjectList,
				WIDTH, HEIGHT);

			cudaVkSemaphoreSignal(mCudaExtCudaUpdateVkSemaphore);
		}
	}
}