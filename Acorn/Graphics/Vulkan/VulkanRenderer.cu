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

				float4 t =
					tex2DLod<float4>(textureMipMapInput, float(x) / baseWidth, float(y) / baseHeight, 0);

				unsigned int dataB = rgbaFloatToInt(t);
				surf2Dwrite(dataB, dstSurfMipMapArray[0], 4 * x, y);

				// x 앞에 4 곱하는 이유
				// 3.2.14.2.1. Surface Object API
				// https://docs.nvidia.com/cuda/cuda-c-programming-guide/#surface-object-api
			}
		}

		// Single-pass gaussian blur
		// https://www.shadertoy.com/view/4tSyzy

		__constant__ const float pi = 3.141592f;
		//__constant__ const int samples = 35;
		//__constant__ const float sigma = float(samples) * 0.25;

		__device__ float gaussian(int dx, int dy, float sigma)
		{
			return 1.0 / (2.0 * pi * sigma * sigma) * exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma));
		}

		__global__ void blurKernel(
			cudaSurfaceObject_t* dstSurfMipMapArray,
			cudaSurfaceObject_t* srcSurfMipMapArray,
			size_t baseWidth, size_t baseHeight,
			int samples, int boundary)
		{
			unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
			unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

			float sigma = float(samples) * 0.25;
			if (x < baseWidth && y < baseHeight)
			{

				if (x > boundary)
				{
					unsigned int pix = surf2Dread<unsigned int>(srcSurfMipMapArray[0], 4 * x, y);
					surf2Dwrite(pix, dstSurfMipMapArray[0], 4 * x, y);
				}
				else
				{
					float4 col = float4{ 0.0f, 0.0f, 0.0f, 0.0f };
					float accum = 0.0;
					float weight;

					int radius = ceil(samples / 2.0f);

					for (int dx = -radius; dx < radius; ++dx)
					{
						for (int dy = -radius; dy < radius; ++dy)
						{
							weight = gaussian(dx, dy, sigma);

							int sx = clamp(int(x) + dx, 0, baseWidth - 1);
							int sy = clamp(int(y) + dy, 0, baseHeight - 1);

							unsigned int pix = surf2Dread<unsigned int>(srcSurfMipMapArray[0], 4 * sx, sy);

							col += rgbaIntToFloat(pix) * weight;
							accum += weight;
						}
					}

					col = col / accum;

					unsigned int dataB = rgbaFloatToInt(col);
					surf2Dwrite(dataB, dstSurfMipMapArray[0], 4 * x, y);
				}
			}
		}

		__device__ float gaussian1D(int x, float sigma)
		{
			return expf(-0.5f * (x * x) / (sigma * sigma)) / (sqrtf(2.0f * pi) * sigma);
		}

		__global__ void gaussianBlurHorizontal(
			cudaSurfaceObject_t* intermediateSurf,
			cudaSurfaceObject_t* srcSurf,
			size_t width, size_t height,
			float sigma, int boundary)
		{
			unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
			unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

			if (x < width && y < height && x < boundary)
			{
				float4 col = float4{ 0.0f, 0.0f, 0.0f, 0.0f };
				float accum = 0.0f;
				int radius = ceilf(3.0f * sigma);

				for (int dx = -radius; dx <= radius; ++dx)
				{
					int sx = clamp(int(x) + dx, 0, int(width) - 1);
					unsigned int pix = surf2Dread<unsigned int>(srcSurf[0], 4 * sx, y);
					float4 pixelCol = rgbaIntToFloat(pix);
					float weight = gaussian1D(dx, sigma);
					col += pixelCol * weight;
					accum += weight;
				}

				col /= accum;

				unsigned int dataB = rgbaFloatToInt(col);
				surf2Dwrite(dataB, intermediateSurf[0], 4 * x, y);
			}
		}

		__global__ void gaussianBlurVertical(
			cudaSurfaceObject_t* dstSurf,
			cudaSurfaceObject_t* intermediateSurf,
			size_t width, size_t height,
			float sigma, int boundary)
		{
			unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
			unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

			if (x < width && y < height && x < boundary)
			{
				float4 col = float4{ 0.0f, 0.0f, 0.0f, 0.0f };
				float accum = 0.0f;
				int radius = ceilf(3.0f * sigma);

				for (int dy = -radius; dy <= radius; ++dy)
				{
					int sy = clamp(int(y) + dy, 0, int(height) - 1);
					unsigned int pix = surf2Dread<unsigned int>(intermediateSurf[0], 4 * x, sy);
					float4 pixelCol = rgbaIntToFloat(pix);
					float weight = gaussian1D(dy, sigma);
					col += pixelCol * weight;
					accum += weight;
				}

				col /= accum;
				unsigned int dataB = rgbaFloatToInt(col);
				surf2Dwrite(dataB, dstSurf[0], 4 * x, y);
			}
		}

		__global__ void sobelKernel(
			cudaSurfaceObject_t* dstSurfMipMapArray,
			cudaSurfaceObject_t* srcSurfMipMapArray,
			size_t baseWidth, size_t baseHeight, int boundary)
		{
			unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
			unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

			if (x < baseWidth && y < baseHeight)
			{
				if (x > boundary)
				{
					unsigned int pix = surf2Dread<unsigned int>(srcSurfMipMapArray[0], 4 * x, y);
					surf2Dwrite(pix, dstSurfMipMapArray[0], 4 * x, y);
				}
				else
				{
					float Gx = 0.0f;
					float Gy = 0.0f;

					int sobelX[3][3] = { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} };

					int sobelY[3][3] = { {-1, -2, -1}, {0, 0, 0}, {1, 2, 1} };

					for (int dx = -1; dx <= 1; ++dx)
					{
						for (int dy = -1; dy <= 1; ++dy)
						{
							int sx = clamp(int(x) + dx, 0, baseWidth - 1);
							int sy = clamp(int(y) + dy, 0, baseHeight - 1);

							unsigned int pix = surf2Dread<unsigned int>(srcSurfMipMapArray[0], 4 * sx, sy);
							float4 col = rgbaIntToFloat(pix);

							float gray = 0.2989f * col.x + 0.5870f * col.y + 0.1140f * col.z;

							Gx += gray * sobelX[dy + 1][dx + 1];
							Gy += gray * sobelY[dy + 1][dx + 1];
						}
					}

					float magnitude = sqrtf(Gx * Gx + Gy * Gy);
					magnitude = magnitude > 1.0f ? 1.0f : magnitude;

					float4 edgeCol = make_float4(magnitude, magnitude, magnitude, 1.0f);
					unsigned int dataB = rgbaFloatToInt(edgeCol);
					surf2Dwrite(dataB, dstSurfMipMapArray[0], 4 * x, y);
				}
			}
		}

		__global__ void sepiaKernel(
			cudaSurfaceObject_t* dstSurfMipMapArray,
			cudaSurfaceObject_t* srcSurfMipMapArray,
			size_t baseWidth, size_t baseHeight, int boundary)
		{
			unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
			unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

			if (x < baseWidth && y < baseHeight)
			{
				if (x > boundary)
				{
					unsigned int pix = surf2Dread<unsigned int>(srcSurfMipMapArray[0], 4 * x, y);
					surf2Dwrite(pix, dstSurfMipMapArray[0], 4 * x, y);
				}
				else
				{
					unsigned int pix = surf2Dread<unsigned int>(srcSurfMipMapArray[0], 4 * x, y);
					float4 col = rgbaIntToFloat(pix);

					float sepiaR = clamp(0.393f * col.x + 0.769f * col.y + 0.189f * col.z, 0.0f, 1.0f);
					float sepiaG = clamp(0.349f * col.x + 0.686f * col.y + 0.168f * col.z, 0.0f, 1.0f);
					float sepiaB = clamp(0.272f * col.x + 0.534f * col.y + 0.131f * col.z, 0.0f, 1.0f);

					float4 sepiaCol = make_float4(sepiaR, sepiaG, sepiaB, col.w);

					unsigned int dataB = rgbaFloatToInt(sepiaCol);
					surf2Dwrite(dataB, dstSurfMipMapArray[0], 4 * x, y);
				}
			}
		}

		__device__ float3 rgbToHsv(float3 rgb)
		{
			float r = rgb.x, g = rgb.y, b = rgb.z;
			float max = fmaxf(r, fmaxf(g, b));
			float min = fminf(r, fminf(g, b));
			float h, s, v = max;

			float d = max - min;
			s = max == 0 ? 0 : d / max;

			if (max == min)
			{
				h = 0; // achromatic
			}
			else
			{
				if (max == r)
				{
					h = (g - b) / d + (g < b ? 6 : 0);
				}
				else if (max == g)
				{
					h = (b - r) / d + 2;
				}
				else if (max == b)
				{
					h = (r - g) / d + 4;
				}
				h /= 6;
			}

			return make_float3(h, s, v);
		}

		__device__ float3 hsvToRgb(float3 hsv)
		{
			float h = hsv.x, s = hsv.y, v = hsv.z;
			int i = floor(h * 6);
			float f = h * 6 - i;
			float p = v * (1 - s);
			float q = v * (1 - f * s);
			float t = v * (1 - (1 - f) * s);

			float3 rgb;

			switch (i % 6)
			{
			case 0:
				rgb = make_float3(v, t, p);
				break;
			case 1:
				rgb = make_float3(q, v, p);
				break;
			case 2:
				rgb = make_float3(p, v, t);
				break;
			case 3:
				rgb = make_float3(p, q, v);
				break;
			case 4:
				rgb = make_float3(t, p, v);
				break;
			case 5:
				rgb = make_float3(v, p, q);
				break;
			}

			return rgb;
		}

		__global__ void hueRotationKernel(
			cudaSurfaceObject_t* dstSurfMipMapArray,
			cudaSurfaceObject_t* srcSurfMipMapArray,
			size_t baseWidth, size_t baseHeight, float hueShift)
		{
			unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
			unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

			if (x < baseWidth && y < baseHeight)
			{
				unsigned int pix = surf2Dread<unsigned int>(srcSurfMipMapArray[0], 4 * x, y);
				float4 col = rgbaIntToFloat(pix);

				// Convert RGB to HSV
				float3 hsv = rgbToHsv(make_float3(col.x, col.y, col.z));

				// Rotate hue
				hsv.x += hueShift;
				if (hsv.x > 1.0f)
				{
					hsv.x -= 1.0f;
				}
				if (hsv.x < 0.0f)
				{
					hsv.x += 1.0f;
				}

				// Convert back to RGB
				float3 rgb = hsvToRgb(hsv);
				float4 rotatedCol = make_float4(rgb.x, rgb.y, rgb.z, col.w);

				unsigned int dataB = rgbaFloatToInt(rotatedCol);
				surf2Dwrite(dataB, dstSurfMipMapArray[0], 4 * x, y);
			}
		}

		__global__ void sharpenKernel(
			cudaSurfaceObject_t* dstSurfMipMapArray,
			cudaSurfaceObject_t* srcSurfMipMapArray,
			size_t baseWidth, size_t baseHeight, int boundary)
		{
			unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
			unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

			if (x < baseWidth && y < baseHeight)
			{
				if (x > boundary)
				{
					unsigned int pix = surf2Dread<unsigned int>(srcSurfMipMapArray[0], 4 * x, y);
					surf2Dwrite(pix, dstSurfMipMapArray[0], 4 * x, y);
				}
				else
				{
					float4 col = float4{ 0.0f, 0.0f, 0.0f, 0.0f };

					// Sharpen filter kernel
					int sharpenKernel[3][3] = { {0, -1, 0}, {-1, 5, -1}, {0, -1, 0} };

					for (int dx = -1; dx <= 1; ++dx)
					{
						for (int dy = -1; dy <= 1; ++dy)
						{
							int sx = clamp(int(x) + dx, 0, baseWidth - 1);
							int sy = clamp(int(y) + dy, 0, baseHeight - 1);

							unsigned int pix = surf2Dread<unsigned int>(srcSurfMipMapArray[0], 4 * sx, sy);
							float4 pixelCol = rgbaIntToFloat(pix);

							col += pixelCol * sharpenKernel[dx + 1][dy + 1];
						}
					}

					// Clamp the color values to the [0, 1] range
					col = make_float4(clamp(col.x, 0.0f, 1.0f), clamp(col.y, 0.0f, 1.0f), clamp(col.z, 0.0f, 1.0f), col.w);

					unsigned int dataB = rgbaFloatToInt(col);
					surf2Dwrite(dataB, dstSurfMipMapArray[0], 4 * x, y);
				}
			}
		}

		__global__ void embossKernel(
			cudaSurfaceObject_t* dstSurfMipMapArray,
			cudaSurfaceObject_t* srcSurfMipMapArray,
			size_t baseWidth, size_t baseHeight, int boundary)
		{
			unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
			unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

			if (x < baseWidth && y < baseHeight)
			{
				if (x > boundary)
				{
					unsigned int pix = surf2Dread<unsigned int>(srcSurfMipMapArray[0], 4 * x, y);
					surf2Dwrite(pix, dstSurfMipMapArray[0], 4 * x, y);
				}
				else
				{
					float4 col = float4{ 0.0f, 0.0f, 0.0f, 0.0f };

					// Emboss filter kernel
					int embossKernel[3][3] = { {-2, -1, 0}, {-1, 1, 1}, {0, 1, 2} };

					for (int dx = -1; dx <= 1; ++dx)
					{
						for (int dy = -1; dy <= 1; ++dy)
						{
							int sx = clamp(int(x) + dx, 0, baseWidth - 1);
							int sy = clamp(int(y) + dy, 0, baseHeight - 1);

							unsigned int pix = surf2Dread<unsigned int>(srcSurfMipMapArray[0], 4 * sx, sy);
							float4 pixelCol = rgbaIntToFloat(pix);

							col += pixelCol * embossKernel[dx + 1][dy + 1];
						}
					}

					// Normalize the color values
					col = (col + 1.0f) * 0.5f;
					col = make_float4(clamp(col.x, 0.0f, 1.0f), clamp(col.y, 0.0f, 1.0f),
						clamp(col.z, 0.0f, 1.0f), col.w);

					unsigned int dataB = rgbaFloatToInt(col);
					surf2Dwrite(dataB, dstSurfMipMapArray[0], 4 * x, y);
				}
			}
		}

		extern int filter_radius;
		extern int boundary;
		extern int boundary_dir;

		void VulkanRenderer::cudaUpdateVkImage()
		{
			cudaVkSemaphoreWait(mCudaExtVkUpdateCudaSemaphore);

			boundary += boundary_dir; // 세로 경계선이 움직이는 효과
			if (boundary > WIDTH - 1)
			{
				boundary = WIDTH - 1;
				boundary_dir = -boundary_dir;
			}
			if (boundary < 0)
			{
				boundary = 0;
				boundary_dir = -boundary_dir;
			}

			int nthreads = 16;
			dim3 dimBlock(nthreads, nthreads, 1); // dimBlock.x * dimBlock.y * dimBlock.z <= 1024
			dim3 dimGrid(
				(uint32_t)ceil(float(mImageWidth) / dimBlock.x),
				(uint32_t)ceil(float(mImageHeight) / dimBlock.y),
				1);

			//copyKernel << <dimGrid, dimBlock >> > (
			//	dev_mSurfaceObjectList,
			//	mTextureObjMipMapInput,
			//	mImageWidth, mImageHeight);
			//gaussianBlurHorizontal << <dimGrid, dimBlock, 0, mStreamToRun >> > (
			//	dev_mSurfaceObjectListTemp,
			//	dev_mSurfaceObjectList,
			//	mImageWidth, mImageHeight,
			//	10, boundary);
			//gaussianBlurVertical << <dimGrid, dimBlock, 0, mStreamToRun >> > (
			//	dev_mSurfaceObjectList,
			//	dev_mSurfaceObjectListTemp,
			//	mImageWidth, mImageHeight,
			//	10, boundary);

			copyKernel << <dimGrid, dimBlock >> > (
				dev_mSurfaceObjectListTemp,
				mTextureObjMipMapInput,
				mImageWidth, mImageHeight);

			//blurKernel << <dimGrid, dimBlock, 0, mStreamToRun >> > (
			//	dev_mSurfaceObjectList,
			//	dev_mSurfaceObjectListTemp,
			//	mImageWidth, mImageHeight,
			//	filter_radius, boundary);

			//sobelKernel << <dimGrid, dimBlock, 0, mStreamToRun >> > (
			//	dev_mSurfaceObjectList,
			//	dev_mSurfaceObjectListTemp,
			//	mImageWidth, mImageHeight,
			//	boundary);

			//sepiaKernel << <dimGrid, dimBlock, 0, mStreamToRun >> > (
			//	dev_mSurfaceObjectList,
			//	dev_mSurfaceObjectListTemp,
			//	mImageWidth, mImageHeight,
			//	boundary);

			//hueRotationKernel << <dimGrid, dimBlock, 0, mStreamToRun >> > (
			//	dev_mSurfaceObjectList,
			//	dev_mSurfaceObjectListTemp,
			//	mImageWidth, mImageHeight,
			//	float(boundary) / WIDTH * 3.141592f);

			embossKernel << <dimGrid, dimBlock, 0, mStreamToRun >> > (
				dev_mSurfaceObjectList,
				dev_mSurfaceObjectListTemp,
				mImageWidth, mImageHeight,
				boundary);

			cudaVkSemaphoreSignal(mCudaExtCudaUpdateVkSemaphore);
		}
	}
}