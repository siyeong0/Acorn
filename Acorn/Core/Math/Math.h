#pragma once
#include "Bounds.h"
#include "FVector2.h"
#include "FVector3.h"
#include "FVector4.h"
#include "FMatrix4x4.h"

namespace aco
{
	inline void Mat4x4Translate(FMatrix4x4& mat, float x, float y, float z)
	{
		FMatrix4x4 t = FMatrix4x4::Identity();
		t[0][3] = x;
		t[1][3] = y;
		t[2][3] = z;
		mat = t * mat;
	}

	inline void Mat4x4LookAt(FMatrix4x4& out, const FVector3& eye, const FVector3& center, const FVector3& up)
	{
		FVector3 f = (center - eye).Normalized();
		FVector3 s = FVector3::Cross(f, up).Normalized();
		FVector3 u = FVector3::Cross(s, f);

		out = FMatrix4x4::Identity();
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
	}

	inline void Mat4x4Ortho(FMatrix4x4& out, float left, float right, float bottom, float top, float zNear, float zFar)
	{
		out = FMatrix4x4::Zero();
		out[0][0] = 2.0f / (right - left);
		out[1][1] = 2.0f / (top - bottom);
		out[2][2] = -2.0f / (zFar - zNear);
		out[3][3] = 1.0f;

		out[0][3] = -(right + left) / (right - left);
		out[1][3] = -(top + bottom) / (top - bottom);
		out[2][3] = -(zFar + zNear) / (zFar - zNear);
	}
}
