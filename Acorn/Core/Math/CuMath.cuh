#pragma once
#include <cuda_runtime.h>
#include <math.h>

// --------------------------------------------------
// float2
// --------------------------------------------------
__host__ __device__ inline float2 operator+(float2 a, float2 b)
{
    return make_float2(a.x + b.x, a.y + b.y);
}

__host__ __device__ inline float2 operator-(float2 a, float2 b)
{
    return make_float2(a.x - b.x, a.y - b.y);
}

__host__ __device__ inline float2 operator*(float2 a, float2 b)
{
    return make_float2(a.x * b.x, a.y * b.y);
}

__host__ __device__ inline float2 operator*(float2 v, float s)
{
    return make_float2(v.x * s, v.y * s);
}

__host__ __device__ inline float2 operator*(float s, float2 v)
{
    return v * s;
}

__host__ __device__ inline float2 operator/(float2 v, float s)
{
    float inv = 1.0f / s;
    return make_float2(v.x * inv, v.y * inv);
}

__host__ __device__ inline float2 operator-(float2 v)
{
    return make_float2(-v.x, -v.y);
}

namespace cumath
{
    __host__ __device__ inline float Dot(float2 a, float2 b)
    {
        return a.x * b.x + a.y * b.y;
    }

    __host__ __device__ inline float Length(float2 v)
    {
        return sqrtf(Dot(v, v));
    }

    __host__ __device__ inline float2 Normalize(float2 v)
    {
        float len = Length(v);
        return (len > 0.0f) ? v / len : make_float2(0.0f, 0.0f);
    }

    __host__ __device__ inline float2 Abs(float2 v)
    {
        return make_float2(fabsf(v.x), fabsf(v.y));
    }

    __host__ __device__ inline float2 Min(float2 a, float2 b)
    {
        return make_float2(fminf(a.x, b.x), fminf(a.y, b.y));
    }

    __host__ __device__ inline float2 Max(float2 a, float2 b)
    {
        return make_float2(fmaxf(a.x, b.x), fmaxf(a.y, b.y));
    }

    __host__ __device__ inline float2 Clamp(float2 v, float2 minVal, float2 maxVal)
    {
        return Max(Min(v, maxVal), minVal);
    }

    __host__ __device__ inline float2 Lerp(float2 a, float2 b, float t)
    {
        return a + (b - a) * t;
    }
}

// --------------------------------------------------
// float3
// --------------------------------------------------

__host__ __device__ inline float3 operator+(float3 a, float3 b)
{
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ inline float3 operator-(float3 a, float3 b)
{
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ inline float3 operator*(float3 a, float3 b)
{
	return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__host__ __device__ inline float3 operator*(float3 v, float s)
{
	return make_float3(v.x * s, v.y * s, v.z * s);
}

__host__ __device__ inline float3 operator*(float s, float3 v)
{
	return v * s;
}

__host__ __device__ inline float3 operator/(float3 v, float s)
{
	float inv = 1.0f / s;
	return make_float3(v.x * inv, v.y * inv, v.z * inv);
}

__host__ __device__ inline float3 operator-(float3 v)
{
	return make_float3(-v.x, -v.y, -v.z);
}

namespace cumath
{
	__host__ __device__ inline float Dot(float3 a, float3 b)
	{
		return a.x * b.x + a.y * b.y + a.z * b.z;
	}

	__host__ __device__ inline float3 Cross(float3 a, float3 b)
	{
		return make_float3(
			a.y * b.z - a.z * b.y,
			a.z * b.x - a.x * b.z,
			a.x * b.y - a.y * b.x
		);
	}

	__host__ __device__ inline float Length(float3 v)
	{
		return sqrtf(Dot(v, v));
	}

	__host__ __device__ inline float3 Normalize(float3 v)
	{
		float len = Length(v);
		return (len > 0.0f) ? v / len : make_float3(0, 0, 0);
	}

	__host__ __device__ inline float3 Abs(float3 v)
	{
		return make_float3(fabsf(v.x), fabsf(v.y), fabsf(v.z));
	}

	__host__ __device__ inline float3 Min(float3 a, float3 b)
	{
		return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
	}

	__host__ __device__ inline float3 Max(float3 a, float3 b)
	{
		return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
	}

	__host__ __device__ inline float3 Clamp(float3 v, float3 minVal, float3 maxVal)
	{
		return Max(Min(v, maxVal), minVal);
	}

	__host__ __device__ inline float3 Lerp(float3 a, float3 b, float t)
	{
		return a + (b - a) * t;
	}
}