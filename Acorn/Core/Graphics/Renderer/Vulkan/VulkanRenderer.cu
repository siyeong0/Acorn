#include "VulkanRenderer.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Math/CuMath.cuh"
#include "Graphics/Voxel/Avox.h"

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

		struct Ray
		{
			float3 Origin;
			float3 Dir;
		};

		struct RayTracingMaterial
		{
			float4 Color;
			float4 EmissionColor;
			float4 SpecularColor;
			float EmissionStrength;
			float Smoothness;
			float SpecularProbability;
			int Flag;
		};

		struct HitInfo
		{
			bool bHit;
			float Distance;
			float3 HitPoint;
			float3 Normal;
			RayTracingMaterial Material;
		};

		__host__ __device__ HitInfo RaySphere(Ray ray, float3 sphereCenter, float sphereRadius)
		{
			HitInfo hitInfo = {};
			float3 offsetToRayOrigin = ray.Origin - sphereCenter;

			float a = cumath::Dot(ray.Dir, ray.Dir);
			float b = 2.0 * cumath::Dot(offsetToRayOrigin, ray.Dir);
			float c = cumath::Dot(offsetToRayOrigin, offsetToRayOrigin) - sphereRadius * sphereRadius;
			float discriminant = b * b - 4 * a * c;

			if (discriminant >= 0.0)
			{
				float dist = (-b - sqrt(discriminant)) / (2.0 * a);

				if (dist > 0.0)
				{
					hitInfo.bHit = true;
					hitInfo.Distance = dist;
					hitInfo.HitPoint = ray.Origin + ray.Dir * dist;
					hitInfo.Normal = cumath::Normalize(hitInfo.HitPoint - sphereCenter);
				}
			}

			return hitInfo;
		}

		__global__ void TraceRayKernel(
			cudaSurfaceObject_t* dstSurface, size_t baseWidth, size_t baseHeight,
			float4x4 cameraMatrix, float3 viewParams, float3 sphereCenter, float radius)
		{
			unsigned int ux = blockIdx.x * blockDim.x + threadIdx.x;
			unsigned int uy = blockIdx.y * blockDim.y + threadIdx.y;

			if (ux >= baseWidth || uy >= baseHeight) return;

			float ndcX = ((float)ux / (float)baseWidth) * 2.0f - 1.0f;
			float ndcY = ((float)uy / (float)baseHeight) * 2.0f - 1.0f;

			// Row-major
			float3 cameraRight = make_float3(cameraMatrix._m00, cameraMatrix._m01, cameraMatrix._m02);
			float3 cameraUp = make_float3(cameraMatrix._m10, cameraMatrix._m11, cameraMatrix._m12);
			float3 cameraForward = make_float3(cameraMatrix._m20, cameraMatrix._m21, cameraMatrix._m22);
			float3 cameraPos = make_float3(cameraMatrix._m30, cameraMatrix._m31, cameraMatrix._m32);

			Ray ray;
			ray.Origin = cameraPos;
			float3 viewPointLocal = make_float3(ndcX * viewParams.x * 0.5f, ndcY * viewParams.y * 0.5f, viewParams.z);
			float4 viewPointTemp = cumath::Mul(make_float4(viewPointLocal.x, viewPointLocal.y, viewPointLocal.z, 1.0f), cameraMatrix);
			float3 viewPoint = make_float3(viewPointTemp.x, viewPointTemp.y, viewPointTemp.z);
			ray.Dir = cumath::Normalize(viewPoint - ray.Origin);

			if (RaySphere(ray, sphereCenter, radius).bHit)
			{
				float4 redColor = make_float4(1.0f, 0.0f, 0.0f, 1.0f);
				unsigned int uiRedColor = rgbaFloatToInt(redColor);
				surf2Dwrite(uiRedColor, dstSurface[0], 4 * ux, uy);
			}
			else
			{
				float4 whiteColor = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
				unsigned int uiWhiteColor = rgbaFloatToInt(whiteColor);
				surf2Dwrite(uiWhiteColor, dstSurface[0], 4 * ux, uy);
			}
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

			FVector3 eye = { 0.0f, 0.0f, -5.0f };
			FVector3 center = { 0.0f, 0.0f, 1.0f };
			FVector3 up = { 0.0f, 1.0f, 0.0f };

			FVector3 forward = (center - eye).Normalized();               // +Z 방향
			FVector3 right = FVector3::Cross(up, forward).Normalized(); // +X 방향
			FVector3 newUp = FVector3::Cross(forward, right);           // +Y 방향

			float4x4 cameraMatrix;
			cameraMatrix._m00 = right.x;
			cameraMatrix._m01 = right.y;
			cameraMatrix._m02 = right.z;
			cameraMatrix._m03 = 0.0f;

			cameraMatrix._m10 = newUp.x;
			cameraMatrix._m11 = newUp.y;
			cameraMatrix._m12 = newUp.z;
			cameraMatrix._m13 = 0.0f;

			cameraMatrix._m20 = forward.x;
			cameraMatrix._m21 = forward.y;
			cameraMatrix._m22 = forward.z;
			cameraMatrix._m23 = 0.0f;

			cameraMatrix._m30 = eye.x;
			cameraMatrix._m31 = eye.y;
			cameraMatrix._m32 = eye.z;
			cameraMatrix._m33 = 1.0f;

			float fovYDeg = 90.0f;
			float aspectRatio = 16.0f / 9.0f;
			float nearZ = 0.1f;

			float halfFovRad = fovYDeg * 0.5f * (3.14159265f / 180.0f); // deg → rad
			float planeHeight = 2.0f * nearZ * tanf(halfFovRad);        // 수직 크기
			float planeWidth = planeHeight * aspectRatio;
			float3 viewParams = make_float3(planeWidth, planeHeight, nearZ);

			TraceRayKernel << <dimGrid, dimBlock, 0, mStreamToRun >> > (
				dev_mSurfaceObjectList, WIDTH, HEIGHT,
				cameraMatrix, viewParams, make_float3(0.0f, 0.0f, 5.0f), 0.1f);

			cudaVkSemaphoreSignal(mCudaExtCudaUpdateVkSemaphore);
		}
	}
}