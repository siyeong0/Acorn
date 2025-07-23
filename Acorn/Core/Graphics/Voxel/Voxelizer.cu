#include "Voxelizer.h"

#include <iostream>
#include <fstream>
#include <format>
#include <cmath>
#include <algorithm>
#include <set>
#include <unordered_map>

#include "Debug.h"

#include "Math/Math.h"
#include "Math/CuMath.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

namespace aco
{
	// --------------------------------------------------
	// Contants
	// --------------------------------------------------

	__constant__ float dev_voxelSize; // Size of a voxel in world space.
	__constant__ int dev_numVoxelsPerAxis; // 2^10
	__constant__ float dev_octreeSize; // Size of the octree in world space.

	// --------------------------------------------------
	// Bit Operations
	// --------------------------------------------------

	inline __host__ __device__ thrust::pair<uint32_t, uint32_t> splitBits(uint32_t code, uint32_t n)
	{
		// ASSERT(n >= 0 && n < 32, "Spliting pivot out of bounds.");
		uint32_t high = code >> (32 - n);
		uint32_t low = code & ((1u << (32 - n)) - 1);
		return { high, low };
	}

	inline __host__ __device__ uint32_t combineBits(uint32_t high, uint32_t low, uint32_t n)
	{
		// ASSERT(n >= 0 && n < 32, "Combining pivot out of bounds.");
		return (high << n) | low;
	}

	// --------------------------------------------------
	// Math Utilities
	// --------------------------------------------------

	// Compute incenter(내심) of a triangle defined by three vertices.
	inline __host__ __device__ float3 computeIncenter(float3 v0, float3 v1, float3 v2)
	{
		float a = cumath::Length(v1 - v2);
		float b = cumath::Length(v2 - v0);
		float c = cumath::Length(v0 - v1);

		float sum = a + b + c;
		// ASSERT(sum != 0.0f, "Invalid triangle. Cannot compute incenter.");
		return (a * v0 + b * v1 + c * v2) / sum;
	}

	// Compute intersection point of two lines defined by FVector4{pointA.x, pointA.y, pointB.x, pointB.y}.
	inline __host__ __device__ float2 computeLineIntersection(float4 line0, float4 line1)
	{
		float2 a{ line0.x, line0.y };
		float2 b{ line0.z, line0.w };
		float2 c{ line1.x, line1.y };
		float2 d{ line1.z, line1.w };

		float2 d1 = b - a;
		float2 d2 = d - c;

		float denom = d1.x * d2.y - d1.y * d2.x;

		// ASSERT(denom != 0.0f, "Lines are parallel or coincident, cannot compute intersection.");
		//if (denom == 0.0f)
		//	return FVector2::Zero(); // Lines are parallel or coincident, return zero vector.

		float2 ac = c - a;
		float t = (ac.x * d2.y - ac.y * d2.x) / denom;
		return a + d1 * t;
	}

	inline __host__ __device__ float2 project(float3 p, float3 center, float3 tangent, float3 bitangent)
	{
		float3 d = p - center;
		return make_float2(cumath::Dot(d, tangent), cumath::Dot(d, bitangent));
	}

	inline __host__ __device__ float3 backProject(float2 uv, float d, float3 center, float3 tangent, float3 bitangent, float3 normal)
	{
		return center + uv.x * tangent + uv.y * bitangent + d * normal;
	}

	// Check if point p is inside the triangle defined by vertices a, b, c in 2D space.
	inline __host__ __device__ bool isInsideTriangle2D(float2 p, float2 a, float2 b, float2 c)
	{
		float2 ab = b - a;
		float2 bc = c - b;
		float2 ca = a - c;
		float2 ap = p - a;
		float2 bp = p - b;
		float2 cp = p - c;

		float w1 = ab.x * ap.y - ab.y * ap.x;
		float w2 = bc.x * bp.y - bc.y * bp.x;
		float w3 = ca.x * cp.y - ca.y * cp.x;
		return (w1 >= 0 && w2 >= 0 && w3 >= 0) || (w1 <= 0 && w2 <= 0 && w3 <= 0);
	}

	// Compute barycentric coordinate of point p with respect to triangle defined by vertices a, b, c in 2D space.
	// What is "Barycentric Coordinate"? :https://en.wikipedia.org/wiki/Barycentric_coordinate_system
	inline __host__ __device__ float3 computeBarycentric2D(float2 p, float2 a, float2 b, float2 c)
	{
		float2 v0 = b - a, v1 = c - a, v2 = p - a;
		float d00 = cumath::Dot(v0, v0), d01 = cumath::Dot(v0, v1), d11 = cumath::Dot(v1, v1);
		float d20 = cumath::Dot(v2, v0), d21 = cumath::Dot(v2, v1);
		float denom = d00 * d11 - d01 * d01;

		if (denom == 0.0f) return make_float3(0, 0, 1);
		float v = (d11 * d20 - d01 * d21) / denom;
		float w = (d00 * d21 - d01 * d20) / denom;
		float u = 1.0f - v - w;
		return make_float3(u, v, w);
	}

	inline __host__ __device__ float2 outwardEdgeNormal(float2 p0, float2 p1, float2 opp)
	{
		float2 edge = p1 - p0;
		float2 perp = make_float2(-edge.y, edge.x); // Perpendicular.

		float2 center = (p0 + p1) * 0.5f;
		float2 toOpp = opp - center;

		perp = perp * (cumath::Dot(perp, toOpp) > 0 ? -1.0f : 1.0f);
		return cumath::Normalize(perp);
	};

	inline __host__ __device__ float4 expandedEdge(float2 p0, float2 p1, float dist, float2 opp)
	{
		float2 dir = outwardEdgeNormal(p0, p1, opp);
		float2 offset = dir * dist * 2.0f; // TODO: Why 2.0??????
		return make_float4(p0.x + offset.x, p0.y + offset.y, p1.x + offset.x, p1.y + offset.y);
	}

	// --------------------------------------------------
	// Morton Code
	// --------------------------------------------------

	inline __device__ uint32_t expandBits(uint32_t x)
	{
		x = (x | (x << 16)) & 0x030000FF;
		x = (x | (x << 8)) & 0x0300F00F;
		x = (x | (x << 4)) & 0x030C30C3;
		x = (x | (x << 2)) & 0x09249249;
		return x;
	}

	inline __device__ uint32_t encodeMortonCode(float3 v)
	{
		uint32_t mx = static_cast<uint32_t>((v.x + 0.5f * dev_octreeSize) * dev_numVoxelsPerAxis / dev_octreeSize);
		uint32_t my = static_cast<uint32_t>((v.y + 0.5f * dev_octreeSize) * dev_numVoxelsPerAxis / dev_octreeSize);
		uint32_t mz = static_cast<uint32_t>((v.z + 0.5f * dev_octreeSize) * dev_numVoxelsPerAxis / dev_octreeSize);
		uint32_t mortonCode = (expandBits(mx) << 2) | (expandBits(my) << 1) | (expandBits(mz) << 0);
		return mortonCode;
	}

	inline __device__ uint32_t compactBits(uint32_t x)
	{
		x &= 0x09249249;
		x = (x ^ (x >> 2)) & 0x030C30C3;
		x = (x ^ (x >> 4)) & 0x0300F00F;
		x = (x ^ (x >> 8)) & 0x030000FF;
		x = (x ^ (x >> 16)) & 0x000003FF;
		return x;
	}

	inline __device__ float3 decodeMortonCode(uint32_t code)
	{
		uint32_t x = compactBits(code >> 2);
		uint32_t y = compactBits(code >> 1);
		uint32_t z = compactBits(code >> 0);

		return make_float3(
			x * dev_voxelSize - 0.5f * dev_octreeSize,
			y * dev_voxelSize - 0.5f * dev_octreeSize,
			z * dev_voxelSize - 0.5f * dev_octreeSize
		);
	}

	// --------------------------------------------------
	// Fragment.
	// --------------------------------------------------

	struct Fragment
	{
		uint32_t MortonCode;
		uint16_t Color;
		uint16_t Dummy;
	};
	static_assert(sizeof(Fragment) == 8, "Fragment size must be 8 bytes.");

	struct FragmentLessByMortonCode
	{
		__host__ __device__ bool operator()(const Fragment& a, const Fragment& b) const { return a.MortonCode < b.MortonCode; }
	};

	struct FragmentEqualByMortonCode
	{
		__host__ __device__ bool operator()(const Fragment& a, const Fragment& b) const { return a.MortonCode == b.MortonCode; }
	};

	// --------------------------------------------------
	// Color Prividers for mesh rasterization.
	// --------------------------------------------------

	__host__ __device__ inline uint16_t packRGB565(const float3& color)
	{
		uint8_t r = static_cast<uint8_t>(fminf(fmaxf(color.x, 0.0f), 1.0f) * 31.0f);  // 5비트
		uint8_t g = static_cast<uint8_t>(fminf(fmaxf(color.y, 0.0f), 1.0f) * 63.0f);  // 6비트
		uint8_t b = static_cast<uint8_t>(fminf(fmaxf(color.z, 0.0f), 1.0f) * 31.0f);  // 5비트

		return (r << 11) | (g << 5) | b;
	}

	__host__ __device__ inline float3 unpackRGB565(uint16_t packed)
	{
		float r = ((packed >> 11) & 0x1F) / 31.0f;
		float g = ((packed >> 5) & 0x3F) / 63.0f;
		float b = (packed & 0x1F) / 31.0f;
		return make_float3(r, g, b);
	}

	std::pair<cudaArray_t, cudaTextureObject_t> createCudaTextureObject(const aco::Texture& tex)
	{
		// Create a CUDA texture object from a vsn::Texture.
		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();

		cudaArray_t cuArray;
		cudaMallocArray(&cuArray, &channelDesc, tex.Width, tex.Height);

		// Copy texture data to CUDA array.
		CUDA_CHECK(cudaMemcpy2DToArray(
			cuArray,
			0, 0,
			tex.Data.data(),
			tex.Width * sizeof(FVector4),
			tex.Width * sizeof(FVector4),
			tex.Height,
			cudaMemcpyHostToDevice));

		// Create resource descriptor.
		cudaResourceDesc resDesc = {};
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = cuArray;

		// Crate texture descriptor.
		cudaTextureDesc texDesc = {};
		texDesc.addressMode[0] = cudaAddressModeWrap;
		texDesc.addressMode[1] = cudaAddressModeWrap;
		texDesc.filterMode = cudaFilterModeLinear;
		texDesc.readMode = cudaReadModeElementType; // float4 그대로
		texDesc.normalizedCoords = 1;

		// Create texture object.
		cudaTextureObject_t texObj = 0;
		CUDA_CHECK(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr));

		return std::make_pair(cuArray, texObj);
	}

	struct SolidColorProvider
	{
		float3 Color{ 1.0f, 1.0f, 1.0f };
		__device__ float3 operator()(int triIdx, float3 barycentric) const
		{
			return Color; // Return solid color.
		}
	};

	struct VertexColorProvider
	{
		const float4* VertexColors;
		const uint32_t* Indices;

		__device__ float3 operator()(int triIdx, float3 barycentric) const
		{
			const uint32_t i0 = Indices[triIdx * 3 + 0];
			const uint32_t i1 = Indices[triIdx * 3 + 1];
			const uint32_t i2 = Indices[triIdx * 3 + 2];

			float3 c0 = make_float3(VertexColors[i0].x, VertexColors[i0].y, VertexColors[i0].z);
			float3 c1 = make_float3(VertexColors[i1].x, VertexColors[i1].y, VertexColors[i1].z);
			float3 c2 = make_float3(VertexColors[i2].x, VertexColors[i2].y, VertexColors[i2].z);

			float3 c = c0 * barycentric.x + c1 * barycentric.y + c2 * barycentric.z;
			return c;
		}
	};

	struct TextureColorProvider
	{
		const float2* VertexUVs = nullptr;
		const uint32_t* Indices = nullptr;
		cudaTextureObject_t TexureObject = 0;

		__device__ float3 operator()(int triIdx, float3 barycentric) const
		{
			const uint32_t i0 = Indices[triIdx * 3 + 0];
			const uint32_t i1 = Indices[triIdx * 3 + 1];
			const uint32_t i2 = Indices[triIdx * 3 + 2];

			float2 uv0 = VertexUVs[i0];
			float2 uv1 = VertexUVs[i1];
			float2 uv2 = VertexUVs[i2];

			float2 uv = uv0 * barycentric.x + uv1 * barycentric.y + uv2 * barycentric.z;

			float4 c = tex2D<float4>(TexureObject, uv.x, uv.y);
			return make_float3(c.x, c.y, c.z); // Ignore alpha channel.
		}
	};

	// --------------------------------------------------
	// VoxLize
	// --------------------------------------------------

	// Parallel for each triangle, rasterize and sample morton codes.
	template <typename ColorProvider>
	__global__ void rasterizeTrianglesKernel(
		const float3* vertices, const uint32_t* indices, int numTriangles,
		ColorProvider colorFunc, Fragment* outFragmentList, int* globalCounter)
	{
		int triIdx = blockIdx.x * blockDim.x + threadIdx.x;
		if (triIdx >= numTriangles) return;


		const float3 v0 = vertices[indices[3 * triIdx + 0]];
		const float3 v1 = vertices[indices[3 * triIdx + 1]];
		const float3 v2 = vertices[indices[3 * triIdx + 2]];

		const float3 normal = cumath::Normalize(cumath::Cross(v1 - v0, v2 - v0));
		const float3 center = computeIncenter(v0, v1, v2);

		// Projection basis
		const float3 tangent = (std::fabsf(normal.x) < 0.9f)
			? cumath::Normalize(cumath::Cross(normal, make_float3(1.0f, 0.0f, 0.0f)))
			: cumath::Normalize(cumath::Cross(normal, make_float3(0.0f, 1.0f, 0.0f)));
		const float3 bitangent = cumath::Normalize(cumath::Cross(normal, tangent)); // guaranteed orthogonal

		// Project vertices to dominant axis.
		float2 p0 = project(v0, center, tangent, bitangent);
		float2 p1 = project(v1, center, tangent, bitangent);
		float2 p2 = project(v2, center, tangent, bitangent);
		float d0 = cumath::Dot(v0 - center, normal);
		float d1 = cumath::Dot(v1 - center, normal);
		float d2 = cumath::Dot(v2 - center, normal);

		// Compute rasterization bounds.
		float2 minP = cumath::Min(cumath::Min(p0, p1), p2);
		float2 maxP = cumath::Max(cumath::Max(p0, p1), p2);

		float2 minGrid = make_float2(
			std::floor(minP.x / dev_voxelSize) * dev_voxelSize - dev_voxelSize * 0.5f, // Center of the voxel.
			std::floor(minP.y / dev_voxelSize) * dev_voxelSize - dev_voxelSize * 0.5f
		);
		float2 maxGrid = make_float2(
			std::ceil(maxP.x / dev_voxelSize) * dev_voxelSize + dev_voxelSize * 0.5f,
			std::ceil(maxP.y / dev_voxelSize) * dev_voxelSize + dev_voxelSize * 0.5f
		);

		// To ensure conservative rasterization,
		// exapnd projected triangle for checking voxel coverage.
		float4 edge01 = expandedEdge(p0, p1, dev_voxelSize, make_float2(0.0f, 0.0f));
		float4 edge12 = expandedEdge(p1, p2, dev_voxelSize, make_float2(0.0f, 0.0f));
		float4 edge20 = expandedEdge(p2, p0, dev_voxelSize, make_float2(0.0f, 0.0f));

		float2 ep0 = computeLineIntersection(edge01, edge20);
		float2 ep1 = computeLineIntersection(edge01, edge12);
		float2 ep2 = computeLineIntersection(edge12, edge20);

		// Rasterize the triangle in 2D space
		for (float y = minGrid.y; y <= maxGrid.y; y += dev_voxelSize)
		{
			for (float x = minGrid.x; x <= maxGrid.x; x += dev_voxelSize)
			{
				float2 p{ x, y };
				if (!isInsideTriangle2D(p, ep0, ep1, ep2)) continue;

				float3 barycentric = computeBarycentric2D(p, p0, p1, p2);
				float d = (barycentric.x * d0 + barycentric.y * d1 + barycentric.z * d2);

				for (int dz = -1; dz <= 1; ++dz) // For conservative rasterization, sample 3 layers.
				{
					float3 voxelPos = backProject(p, d + dz * dev_voxelSize, center, tangent, bitangent, normal);
					int index = atomicAdd(globalCounter, 1);
					uint32_t mortonCode = encodeMortonCode(voxelPos);
					float3 color = colorFunc(triIdx, barycentric);
					outFragmentList[index] = Fragment{ mortonCode, packRGB565(color), 0 };
				}
			}
		}
	}

	__global__ void markChildFlagsKernel(
		AvoxNode* nodePool, Fragment* outFragmentList, int numFragments, uint32_t parentOffset, int mortonCodeBitSize)
	{
		int fragmentIdx = blockIdx.x * blockDim.x + threadIdx.x;
		if (fragmentIdx >= numFragments) return;

		uint32_t code = outFragmentList[fragmentIdx].MortonCode;

		thrust::pair<uint32_t, uint32_t> parentData = splitBits(code, 32 - mortonCodeBitSize);
		uint32_t parentIndex = parentData.first;
		uint32_t parentCode = parentData.second;

		AvoxNode* parentNode = nodePool + parentOffset + parentIndex;
		uint32_t childIndex = parentCode >> (mortonCodeBitSize - 3); // Get first 3 bits for child index.
		atomicOr(reinterpret_cast<uint32_t*>(&(parentNode->ChildMask)), 1 << childIndex); // Be careful of overrun.
	}

	__global__ void allocChildNodesKernel(
		AvoxNode* nodePool, uint32_t* poolCounter, uint32_t parentOffset, uint32_t numChilds)
	{
		int nodeIdx = blockIdx.x * blockDim.x + threadIdx.x;
		if (nodeIdx >= numChilds) return;

		AvoxNode* parentNode = nodePool + parentOffset + nodeIdx;
		if (parentNode->ChildMask == 0) return; // Skip empty nodes.

		uint32_t poolIndex = atomicAdd(poolCounter, 8); // Allocate 8 child nodes.

		thrust::pair<uint32_t, uint32_t> childBits = splitBits(poolIndex, 16);
		uint32_t pageHeader = childBits.first;
		uint32_t childPointer = childBits.second;

		parentNode->PageHeader = static_cast<uint8_t>(pageHeader);
		parentNode->ChildPointer = static_cast<uint16_t>(childPointer);
	}
	
	__global__ void updateFragmentsKernel(
		AvoxNode* nodePool, Fragment* outFragmentList, int numFragments, uint32_t parentOffset, uint32_t childOffset, int mortonCodeBitSize)
	{
		int fragmentIdx = blockIdx.x * blockDim.x + threadIdx.x;
		if (fragmentIdx >= numFragments) return;

		uint32_t code = outFragmentList[fragmentIdx].MortonCode;

		thrust::pair<uint32_t, uint32_t> parentData = splitBits(code, 32 - mortonCodeBitSize);
		uint32_t parentIndex = parentData.first;
		uint32_t parentCode = parentData.second;

		AvoxNode* parentNode = nodePool + parentOffset + parentIndex;
		uint32_t childIndex = parentCode >> (mortonCodeBitSize - 3); // Get first 3 bits for child index.

		uint32_t nextIndex = combineBits(static_cast<uint32_t>(parentNode->PageHeader), static_cast<uint32_t>(parentNode->ChildPointer), 16) - childOffset + childIndex;
		uint32_t nextCode = parentCode & ((1u << (mortonCodeBitSize - 3)) - 1); // Remove first 3 bits.

		outFragmentList[fragmentIdx].MortonCode = combineBits(nextIndex, nextCode, mortonCodeBitSize - 3);
	}

	__global__ void paintLeafNodesKernel(
		AvoxNode* nodePool, Fragment* outFragmentList, int numFragments, uint32_t nodeOffset)
	{
		int fragmentIdx = blockIdx.x * blockDim.x + threadIdx.x;
		if (fragmentIdx >= numFragments) return;

		uint32_t nodeIndex = outFragmentList[fragmentIdx].MortonCode;

		AvoxNode* node = nodePool + nodeOffset + nodeIndex;
		node->Color = outFragmentList[fragmentIdx].Color; // Store color in the node.
		node->ColorDummy = 0;
	}

	// Voxeliz a mesh into a voxel tree.
	// 1. Loop for each triangle.
	//	1.1. Projects each triangle onto a plane defined by its normal.
	//	1.2. Rasterizes the triangle and sample pixels.
	//	1.3. Compute depth of the valid pixel.
	//	1.4. Stores sampled voxel(u,v,d) positions as Morton codes.
	// 2. Sort fragment list(morton code list) and remove duplicate morton codes.
	// 3. Loop for octree nodes on depth D.
	//	4.1. For each Morton code, compute the voxel position.
	//	4.2. Set flag for the octree node that contains the voxel position.
	//  4.3. Repeat until depth < MAX_DEPTH.
	// Reference: https://research.nvidia.com/labs/rtr/publication/crassin2012voxelization/
	Avox Voxelizer::BuildAvox(const Mesh& mesh)
	{
		Avox avox;
		constexpr int DEPTH = 10;

		int numTriangles = mesh.NumTriangles();
		const std::vector<FVector3>& vertices = mesh.Vertices;
		const std::vector<uint32_t>& indices = mesh.Indices;
		const std::vector<FVector4>& colors = mesh.Colors;
		const std::vector<FVector2>& uvs = mesh.UVs;
		const Texture* texture = mesh.Tex;
		const Mesh::EColorSourceType colorSourceType = mesh.ColorSourceType;

		const float host_voxelSize = 0.05f;
		const int host_numVoxelsPerAxis = 1024; // 2^10 = 1024
		const float host_octreeSize = host_voxelSize * host_numVoxelsPerAxis;
		ASSERT(std::fabsf(mesh.Bounds.Max.x) < host_octreeSize &&
			std::fabsf(mesh.Bounds.Max.y) < host_octreeSize &&
			std::fabsf(mesh.Bounds.Min.x) < host_octreeSize &&
			std::fabsf(mesh.Bounds.Min.y) < host_octreeSize,
			"Mesh bounds is bigger than max octree size");
		// TODO: 옥트리 크기보다 메쉬 크기가 크면, 옥트리 여러개로 구성 or 동적 뎁스

		// Kernel launch parameters.
		int numThreadsPerBlock = -1;
		int numBlocks = -1;

		// Allocate device memory for mesh data.
		float3* dev_vertices = nullptr;
		uint32_t* dev_indices = nullptr;
		float4* dev_colors = nullptr;
		float2* dev_uvs = nullptr;
		cudaArray_t dev_texureArray = nullptr;
		cudaTextureObject_t dev_textureObject = 0;

		CUDA_CHECK(cudaMalloc(&dev_vertices, vertices.size() * sizeof(float3)));
		CUDA_CHECK(cudaMalloc(&dev_indices, indices.size() * sizeof(uint32_t)));
		CUDA_CHECK(cudaMemcpy(dev_vertices, vertices.data(), vertices.size() * sizeof(float3), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(dev_indices, indices.data(), indices.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));

		if (colorSourceType == Mesh::EColorSourceType::VertexColor)
		{
			CUDA_CHECK(cudaMalloc(&dev_colors, colors.size() * sizeof(float4)));
			CUDA_CHECK(cudaMemcpy(dev_colors, colors.data(), colors.size() * sizeof(float4), cudaMemcpyHostToDevice));
		}
		else if (colorSourceType == Mesh::EColorSourceType::TextureUV)
		{
			CUDA_CHECK(cudaMalloc(&dev_uvs, uvs.size() * sizeof(float2)));
			CUDA_CHECK(cudaMemcpy(dev_uvs, uvs.data(), uvs.size() * sizeof(float2), cudaMemcpyHostToDevice));
			auto texDataPair = createCudaTextureObject(*texture);
			dev_texureArray = texDataPair.first;
			dev_textureObject = texDataPair.second;
		}

		// Parallel for each triangle, rasterize and sample morton codes.
		Fragment* dev_fragmentList = nullptr;
		int* dev_fragCounter = nullptr;
		int maxFragments = static_cast<int>(std::powf(host_numVoxelsPerAxis / 3.0f, 3.0f));

		// Constants.
		CUDA_CHECK(cudaMemcpyToSymbol((const void*)&dev_voxelSize, &host_voxelSize, sizeof(float), 0, cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpyToSymbol((const void*)&dev_numVoxelsPerAxis, &host_numVoxelsPerAxis, sizeof(int), 0, cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpyToSymbol((const void*)&dev_octreeSize, &host_octreeSize, sizeof(float), 0, cudaMemcpyHostToDevice));

		// Allocate device memory.
		CUDA_CHECK(cudaMalloc(&dev_fragmentList, maxFragments * sizeof(Fragment)));
		CUDA_CHECK(cudaMalloc(&dev_fragCounter, sizeof(int)));

		// Copy data to device memory.
		CUDA_CHECK(cudaMemset(dev_fragCounter, 0, sizeof(int)));

		// Rasterize triangles and sample morton codes(fragments).
		numThreadsPerBlock = 256;
		numBlocks = int(std::ceil(float(numTriangles) / numThreadsPerBlock));

		if (colorSourceType == Mesh::EColorSourceType::VertexColor)
		{
			rasterizeTrianglesKernel << <numBlocks, numThreadsPerBlock >> > (
				dev_vertices,
				dev_indices,
				numTriangles,
				VertexColorProvider{ dev_colors, dev_indices },
				dev_fragmentList,
				dev_fragCounter
				);
		}
		else if (colorSourceType == Mesh::EColorSourceType::TextureUV)
		{
			rasterizeTrianglesKernel << <numBlocks, numThreadsPerBlock >> > (
				dev_vertices,
				dev_indices,
				numTriangles,
				TextureColorProvider{ dev_uvs, dev_indices, dev_textureObject },
				dev_fragmentList,
				dev_fragCounter
				);
		}
		else
		{
			rasterizeTrianglesKernel << <numBlocks, numThreadsPerBlock >> > (
				dev_vertices,
				dev_indices,
				numTriangles,
				SolidColorProvider{},
				dev_fragmentList,
				dev_fragCounter
				);
		}

		int numDupFragments;
		CUDA_CHECK(cudaMemcpy(&numDupFragments, dev_fragCounter, sizeof(int), cudaMemcpyDeviceToHost));

		// Sort and unique the fragments by Morton code.
		thrust::device_ptr<Fragment> begin(dev_fragmentList);
		thrust::device_ptr<Fragment> end(dev_fragmentList + numDupFragments);

		thrust::sort(begin, end, FragmentLessByMortonCode());
		auto uniqueLast = thrust::unique(begin, end, FragmentEqualByMortonCode());
		int numFragments = static_cast<int>(thrust::distance(begin, uniqueLast));

		//// Copy back unique fragments to host.
		// TODO: erase
		//std::vector<Fragment> fragmentList(numFragments);
		//CUDA_CHECK(cudaMemcpy(fragmentList.data(), dev_fragmentList, numFragments * sizeof(uint32_t), cudaMemcpyDeviceToHost));

		// Free device memory.
		CUDA_CHECK(cudaFree(dev_vertices));
		CUDA_CHECK(cudaFree(dev_indices));
		CUDA_CHECK(cudaFree(dev_colors));
		CUDA_CHECK(cudaFree(dev_uvs));
		CUDA_CHECK(cudaFree(dev_colors));
		CUDA_CHECK(cudaDestroyTextureObject(dev_textureObject));
		CUDA_CHECK(cudaFreeArray(dev_texureArray));

		CUDA_CHECK(cudaFree(dev_fragCounter));

		// Build an octree from fragments.
		// GPU Memory pool for octree nodes.
		AvoxNode* dev_memPool = nullptr;
		uint32_t* dev_memPoolCounter = nullptr;

		const size_t MEM_POOL_SIZE = 1ull << 32; // 4GB memory pool.
		CUDA_CHECK(cudaMalloc(&dev_memPool, MEM_POOL_SIZE)); // Allocate 1M nodes.
		CUDA_CHECK(cudaMalloc(&dev_memPoolCounter, sizeof(uint32_t))); // Counter for allocated nodes.

		// Initialize memory pool and counter.
		cudaMemset(dev_memPool, 0, MEM_POOL_SIZE);
		// Initialize counter to 8. Root node is already allocated at offset 0. Dummy node is also allocated.
		// CUDA_CHECK(cudaMemset(dev_memPoolCounter, 8, sizeof(uint32_t)));
		uint32_t initialValue = 8;
		CUDA_CHECK(cudaMemcpy(dev_memPoolCounter, &initialValue, sizeof(uint32_t), cudaMemcpyHostToDevice));

		// Loop for octree depth.
		uint32_t parentOffset = 0;
		int mortonCodeBitSize = 30;

		std::vector<uint32_t> nodeOffsets; // Store offsets for each depth.
		nodeOffsets.clear();
		nodeOffsets.reserve(DEPTH + 1);
		nodeOffsets.emplace_back(parentOffset); // Add root node offset.
		for (int octreeDepth = 0; octreeDepth < DEPTH; ++octreeDepth)
		{
			// Parallel for each fragment, mark child flags.
			numThreadsPerBlock = 256;
			numBlocks = int(std::ceil(float(numFragments) / numThreadsPerBlock));
			markChildFlagsKernel << <numBlocks, numThreadsPerBlock >> > (dev_memPool, dev_fragmentList, numFragments, parentOffset, mortonCodeBitSize);

			CUDA_CHECK(cudaGetLastError());
			CUDA_CHECK(cudaDeviceSynchronize());

			// Parallel for each node, allocate child nodes.
			uint32_t childOffset;
			CUDA_CHECK(cudaMemcpy(&childOffset, dev_memPoolCounter, sizeof(uint32_t), cudaMemcpyDeviceToHost));
			uint32_t numChilds = childOffset - parentOffset;

			numThreadsPerBlock = 256;
			numBlocks = int(std::ceil(float(numChilds) / numThreadsPerBlock));
			allocChildNodesKernel << <numBlocks, numThreadsPerBlock >> > (dev_memPool, dev_memPoolCounter, parentOffset, numChilds);

			CUDA_CHECK(cudaGetLastError());
			CUDA_CHECK(cudaDeviceSynchronize());

			// Parallel for each fragment, update fragments.
			numThreadsPerBlock = 256;
			numBlocks = int(std::ceil(float(numFragments) / numThreadsPerBlock));
			updateFragmentsKernel << <numBlocks, numThreadsPerBlock >> > (dev_memPool, dev_fragmentList, numFragments, parentOffset, childOffset, mortonCodeBitSize);

			CUDA_CHECK(cudaGetLastError());
			CUDA_CHECK(cudaDeviceSynchronize());

			// Update parent offset and morton code bit size for next depth.
			parentOffset = childOffset;
			mortonCodeBitSize -= 3; // Decrease bit size for next depth.

			// Store child offset for this depth.
			nodeOffsets.emplace_back(childOffset);
		}
		numThreadsPerBlock = 256;
		numBlocks = int(std::ceil(float(numFragments) / numThreadsPerBlock));
		paintLeafNodesKernel << <numBlocks, numThreadsPerBlock >> > (dev_memPool, dev_fragmentList, numFragments, parentOffset);

		// Copy back the device memory pool to host.
		uint32_t poolCounter;
		CUDA_CHECK(cudaMemcpy(&poolCounter, dev_memPoolCounter, sizeof(uint32_t), cudaMemcpyDeviceToHost));
		avox.Nodes.resize(poolCounter);
		CUDA_CHECK(cudaMemcpy(avox.Nodes.data(), dev_memPool, poolCounter * sizeof(AvoxNode), cudaMemcpyDeviceToHost));

		// Free device memory.
		CUDA_CHECK(cudaFree(dev_fragmentList));
		CUDA_CHECK(cudaFree(dev_memPool));
		CUDA_CHECK(cudaFree(dev_memPoolCounter));

		avox.Offsets = std::move(nodeOffsets);
		return avox;
	}
}