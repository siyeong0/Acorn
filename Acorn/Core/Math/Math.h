#pragma once
#include "Bounds.h"
#include "FVector2.h"
#include "FVector3.h"
#include "FVector4.h"
#include "FMatrix4x4.h"

namespace aco
{
	inline void Mat4x4Translate(FMatrix4x4* out, float x, float y, float z)
	{
		FMatrix4x4 t = FMatrix4x4::Identity();
		t[0][3] = x;
		t[1][3] = y;
		t[2][3] = z;
		*out = t * *out;
	}

	inline FMatrix4x4 Mat4x4LookAt(const FVector3& eye, const FVector3& center, const FVector3& up)
	{
		FVector3 f = (center - eye).Normalized();
		FVector3 s = FVector3::Cross(f, up).Normalized();
		FVector3 u = FVector3::Cross(s, f);

		FMatrix4x4 out = FMatrix4x4::Identity();
		out[0][0] = s.x;
		out[0][1] = s.y;
		out[0][2] = s.z;
		out[1][0] = u.x;
		out[1][1] = u.y;
		out[1][2] = u.z;
		out[2][0] = -f.x;
		out[2][1] = -f.y;
		out[2][2] = -f.z;

		out[0][3] = -FVector3::Dot(s, eye);
		out[1][3] = -FVector3::Dot(u, eye);
		out[2][3] = FVector3::Dot(f, eye);

		return out;
	}

	inline FMatrix4x4 Mat4x4Camera(const FVector3& eye, const FVector3& center, const FVector3& up)
	{
		FVector3 f = (center - eye).Normalized();
		FVector3 s = FVector3::Cross(up, f).Normalized();
		FVector3 u = FVector3::Cross(f, s);

		FMatrix4x4 out = FMatrix4x4::Identity();

		out[0][0] = s.x;
		out[0][1] = s.y;
		out[0][2] = s.z;
		out[0][3] = 0.0f;

		out[1][0] = u.x;
		out[1][1] = u.y;
		out[1][2] = u.z;
		out[1][3] = 0.0f;

		out[2][0] = f.x;
		out[2][1] = f.y;
		out[2][2] = f.z;
		out[2][3] = 0.0f;

		out[3][0] = eye.x;
		out[3][1] = eye.y;
		out[3][2] = eye.z;
		out[3][3] = 1.0f;

		return out;
	}

	inline FMatrix4x4 Mat4x4Ortho(float left, float right, float bottom, float top, float zNear, float zFar)
	{
		FMatrix4x4 out = FMatrix4x4::Zero();
		out[0][0] = 2.0f / (right - left);
		out[1][1] = 2.0f / (top - bottom);
		out[2][2] = -2.0f / (zFar - zNear);
		out[3][3] = 1.0f;

		out[0][3] = -(right + left) / (right - left);
		out[1][3] = -(top + bottom) / (top - bottom);
		out[2][3] = -(zFar + zNear) / (zFar - zNear);

		return out;
	}

	inline FMatrix4x4 MakePerspectiveLH(float fovYRadians, float aspect, float zNear, float zFar)
	{
		float yScale = 1.0f / tanf(fovYRadians * 0.5f); // cotangent(fovY / 2)
		float xScale = yScale / aspect;

		FMatrix4x4 out = FMatrix4x4::Zero();

		/*out._m00 = xScale;
		out._m11 = yScale;
		out._m22 = zFar / (zFar - zNear);
		out._m23 = 1.0f;

		out._m32 = -(zNear * zFar) / (zFar - zNear);
		out._m33 = 0.0f;*/

		out[0][0] = xScale;
		out[1][1] = yScale;
		out[2][2] = zFar / (zFar - zNear);
		out[2][3] = 1.0f;

		out[3][2] = -(zNear * zFar) / (zFar - zNear);
		out[3][3] = 0.0f;

		return out;
	}
}
