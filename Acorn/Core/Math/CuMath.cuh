#pragma once
#include <cuda_runtime.h>
#include <math.h>

// --------------------------------------------------
// float
// --------------------------------------------------
namespace cumath
{
	__host__ __device__ inline float Abs(float v)
	{
		return fabs(v);
	}

	__host__ __device__ inline float Min(float a, float b)
	{
		return fminf(a, b);
	}

	__host__ __device__ inline float Max(float a, float b)
	{
		return fmaxf(a, b);
	}

	__host__ __device__ inline float Clamp(float v, float minVal, float maxVal)
	{
		return Max(Min(v, maxVal), minVal);
	}

	__host__ __device__ inline float Lerp(float a, float b, float t)
	{
		return a + (b - a) * t;
	}
}
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

// --------------------------------------------------
// float4
// --------------------------------------------------

__host__ __device__ inline float4 operator+(float4 a, float4 b)
{
	return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__host__ __device__ inline float4 operator-(float4 a, float4 b)
{
	return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

__host__ __device__ inline float4 operator*(float4 a, float4 b)
{
	return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}

__host__ __device__ inline float4 operator*(float4 v, float s)
{
	return make_float4(v.x * s, v.y * s, v.z * s, v.w * s);
}

__host__ __device__ inline float4 operator*(float s, float4 v)
{
	return v * s;
}

__host__ __device__ inline float4 operator/(float4 v, float s)
{
	float inv = 1.0f / s;
	return make_float4(v.x * inv, v.y * inv, v.z * inv, v.w * inv);
}

__host__ __device__ inline float4 operator-(float4 v)
{
	return make_float4(-v.x, -v.y, -v.z, -v.w);
}

// --------------------------------------------------
// float4x4
// --------------------------------------------------

struct float4x4
{
	union
	{
		struct
		{
			float _m00, _m10, _m20, _m30;
			float _m01, _m11, _m21, _m31;
			float _m02, _m12, _m22, _m32;
			float _m03, _m13, _m23, _m33;
		};
		float data[4][4];
		float4 rows[4];
	};

	__host__ __device__ float4& operator[](int i) { return rows[i]; }
	__host__ __device__ const float4& operator[](int i) const { return rows[i]; }

	__host__ __device__ static float4x4 Identity()
	{
		float4x4 m;
		m._m00 = 1.0f; m._m01 = 0.0f; m._m02 = 0.0f; m._m03 = 0.0f;
		m._m10 = 0.0f; m._m11 = 1.0f; m._m12 = 0.0f; m._m13 = 0.0f;
		m._m20 = 0.0f; m._m21 = 0.0f; m._m22 = 1.0f; m._m23 = 0.0f;
		m._m30 = 0.0f; m._m31 = 0.0f; m._m32 = 0.0f; m._m33 = 1.0f;
		return m;
	}

	__host__ __device__ void Transpose()
	{
		float tmp;
		tmp = _m01; _m01 = _m10; _m10 = tmp;
		tmp = _m02; _m02 = _m20; _m20 = tmp;
		tmp = _m03; _m03 = _m30; _m30 = tmp;
		tmp = _m12; _m12 = _m21; _m21 = tmp;
		tmp = _m13; _m13 = _m31; _m31 = tmp;
		tmp = _m23; _m23 = _m32; _m32 = tmp;
	}
};

namespace cumath
{
	__host__ __device__ inline float Dot(float4 a, float4 b)
	{
		return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
	}

	__host__ __device__ inline float4 Mul(const float4& v, const float4x4& m)
	{
		return make_float4(
			Dot(v, m[0]),
			Dot(v, m[1]),
			Dot(v, m[2]),
			Dot(v, m[3])
		);
	}

	__host__ __device__ inline float4x4 Transpose(const float4x4& m)
	{
		float4x4 t;
		t[0] = make_float4(m[0].x, m[1].x, m[2].x, m[3].x);
		t[1] = make_float4(m[0].y, m[1].y, m[2].y, m[3].y);
		t[2] = make_float4(m[0].z, m[1].z, m[2].z, m[3].z);
		t[3] = make_float4(m[0].w, m[1].w, m[2].w, m[3].w);
		return t;
	}
}