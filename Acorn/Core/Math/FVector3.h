#pragma once
#include <assert.h>
#include <cmath>
#include <numeric>
#include <limits>

#undef max

namespace aco
{
	using FLOAT = float;

	struct FVector3
	{
		FLOAT x;
		FLOAT y;
		FLOAT z;

		FVector3() = default;
		~FVector3() = default;

		FVector3(FLOAT x, FLOAT y, FLOAT z);
		FVector3(const FVector3& other);
		FVector3& operator=(const FVector3& other);
		inline FLOAT& operator[](size_t idx);
		inline const FLOAT& operator[](size_t idx) const;

		static inline FVector3 Zero() { return FVector3{ 0.f, 0.f, 0.f }; }
		static inline FVector3 One() { return FVector3{ 1.f, 1.f, 1.f }; }
		static inline FVector3 UnitX() { return FVector3{ 1.f, 0.f, 0.f }; }
		static inline FVector3 UnitY() { return FVector3{ 0.f, 1.f, 0.f }; }
		static inline FVector3 UnitZ() { return FVector3{ 0.f, 0.f, 1.f }; }
		static inline FVector3 FMaxValue() { constexpr FLOAT v = std::numeric_limits<FLOAT>::max();  return FVector3{ v,v,v, }; }
		static inline FVector3 FMinValue() { constexpr FLOAT v = std::numeric_limits<FLOAT>::lowest();  return FVector3{ v,v,v, }; }
		static inline FVector3 Up() { return FVector3{ 0.f, 1.f, 0.f }; }
		static inline FVector3 Down() { return FVector3{ 0.f, -1.f, 0.f }; }
		static inline FVector3 Right() { return FVector3{ 1.f, 0.f, 0.f }; }
		static inline FVector3 Left() { return FVector3{ -1.f, 0.f, 0.f }; }
		static inline FVector3 Forward() { return FVector3{ 0.f, 0.f, 1.f }; }
		static inline FVector3 Backward() { return FVector3{ 0.f, 0.f, -1.f }; }

		inline FLOAT Dot(const FVector3& other) const;
		inline FVector3 Cross(const FVector3& other) const;

		static inline float Dot(const FVector3& a, const FVector3& b);
		static inline FVector3 Cross(const FVector3& a, const FVector3& b);

		inline FLOAT Magnitude() const;
		inline FLOAT SqrMagnitude() const;
		inline FLOAT Length() const;

		inline FVector3 Normalized() const;
		inline void Normalize();

		inline void operator+=(const FVector3& other);
		inline void operator-=(const FVector3& other);
		inline void operator*=(FLOAT v);
		inline void operator/=(FLOAT v);

		static inline FVector3 Abs(const FVector3& vec);
		static inline FVector3 Min(const FVector3& a, const FVector3& b);
		static inline FVector3 Max(const FVector3& a, const FVector3& b);
		static inline FVector3 Clamp(const FVector3& value, const FVector3& min, const FVector3& max);
		static inline FVector3 Lerp(const FVector3& a, const FVector3& b, FLOAT t);
		static inline FVector3 SmoothStep(const FVector3& a, const FVector3& b, FLOAT t);
	};

	inline FVector3 operator-(const FVector3& vec);
	inline FVector3 operator+(const FVector3& lhs, const FVector3& rhs);
	inline FVector3 operator-(const FVector3& lhs, const FVector3& rhs);
	inline FVector3 operator*(const FVector3& lhs, const FVector3& rhs);
	inline FVector3 operator*(FLOAT v, const FVector3& vec);
	inline FVector3 operator*(const FVector3& vec, FLOAT v);
	inline FVector3 operator/(const FVector3& vec, FLOAT v);
	inline bool operator==(const FVector3& lhs, const FVector3& rhs);
	inline bool operator!=(const FVector3& lhs, const FVector3& rhs);

	// --- Implementation ---

	inline FVector3::FVector3(FLOAT x, FLOAT y, FLOAT z)
		: x(x), y(y), z(z) {
	}

	inline FVector3::FVector3(const FVector3& other)
		: x(other.x), y(other.y), z(other.z) {
	}

	inline FVector3& FVector3::operator=(const FVector3& other)
	{
		x = other.x;
		y = other.y;
		z = other.z;
		return *this;
	}

	inline FLOAT& FVector3::operator[](size_t idx)
	{
		assert(idx < 3);
		return (&x)[idx];
	}

	inline const FLOAT& FVector3::operator[](size_t idx) const
	{
		assert(idx < 3);
		return (&x)[idx];
	}

	inline float FVector3::Dot(const FVector3& a, const FVector3& b)
	{
		return a.Dot(b);
	}

	inline FVector3 FVector3::Cross(const FVector3& a, const FVector3& b)
	{
		return a.Cross(b);
	}

	inline FLOAT FVector3::Dot(const FVector3& other) const
	{
		return x * other.x + y * other.y + z * other.z;
	}

	inline FVector3 FVector3::Cross(const FVector3& other) const
	{
		return FVector3{
			y * other.z - z * other.y,
			z * other.x - x * other.z,
			x * other.y - y * other.x
		};
	}

	inline FLOAT FVector3::Magnitude() const
	{
		return std::sqrtf(SqrMagnitude());
	}

	inline FLOAT FVector3::SqrMagnitude() const
	{
		return Dot(*this);
	}

	inline FLOAT FVector3::Length() const
	{
		return Magnitude();
	}

	inline FVector3 FVector3::Normalized() const
	{
		FLOAT l = Length();
		if (l == 0)
			return FVector3{ 0.f, 0.f, 0.f };
		else
			return *this / l;
	}

	inline void FVector3::Normalize()
	{
		FLOAT l = Length();
		if (l != 0)
			*this /= l;
	}

	inline void FVector3::operator+=(const FVector3& other)
	{
		*this = *this + other;
	}

	inline void FVector3::operator-=(const FVector3& other)
	{
		*this = *this - other;
	}

	inline void FVector3::operator*=(FLOAT v)
	{
		*this = *this * v;
	}

	inline void FVector3::operator/=(FLOAT v)
	{
		*this = *this / v;
	}

	inline FVector3 FVector3::Abs(const FVector3& vec)
	{
		return FVector3{ std::fabs(vec.x), std::fabs(vec.y), std::fabs(vec.z) };
	};

	inline FVector3 FVector3::Min(const FVector3& a, const FVector3& b)
	{
		return FVector3{ std::fmin(a.x, b.x), std::fmin(a.y, b.y), std::fmin(a.z, b.z) };
	};

	inline FVector3 FVector3::Max(const FVector3& a, const FVector3& b)
	{
		return FVector3{ std::fmax(a.x, b.x), std::fmax(a.y, b.y), std::fmax(a.z, b.z) };
	};

	inline FVector3 FVector3::Clamp(const FVector3& value, const FVector3& min, const FVector3& max)
	{
		return FVector3{
			std::fmax(min.x, std::fmin(value.x, max.x)),
			std::fmax(min.y, std::fmin(value.y, max.y)),
			std::fmax(min.z, std::fmin(value.z, max.z))
		};
	}

	inline FVector3 FVector3::Lerp(const FVector3& a, const FVector3& b, FLOAT t)
	{
		t = std::fmax(0.f, std::fmin(t, 1.f)); // Clamp t between 0 and 1
		return FVector3{
			a.x + (b.x - a.x) * t,
			a.y + (b.y - a.y) * t,
			a.z + (b.z - a.z) * t
		};
	}

	inline FVector3 FVector3::SmoothStep(const FVector3& a, const FVector3& b, FLOAT t)
	{
		t = std::fmax(0.f, std::fmin(t, 1.f)); // Clamp t between 0 and 1
		FLOAT t2 = t * t;
		FLOAT t3 = t2 * t;
		return FVector3{
			a.x + (b.x - a.x) * (t2 * (3.f - 2.f * t)),
			a.y + (b.y - a.y) * (t2 * (3.f - 2.f * t)),
			a.z + (b.z - a.z) * (t2 * (3.f - 2.f * t))
		};
	}

	inline FVector3 operator-(const FVector3& vec)
	{
		return FVector3{ -vec.x, -vec.y, -vec.z };
	}

	inline FVector3 operator+(const FVector3& lhs, const FVector3& rhs)
	{
		return FVector3{ lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z };
	}

	inline FVector3 operator-(const FVector3& lhs, const FVector3& rhs)
	{
		return FVector3{ lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z };
	}

	inline FVector3 operator*(const FVector3& lhs, const FVector3& rhs)
	{
		return FVector3{ lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z };
	}

	inline FVector3 operator*(FLOAT v, const FVector3& vec)
	{
		return FVector3{ v * vec.x, v * vec.y, v * vec.z };
	}

	inline FVector3 operator*(const FVector3& vec, FLOAT v)
	{
		return FVector3{ v * vec.x, v * vec.y, v * vec.z };
	}

	inline FVector3 operator/(const FVector3& vec, FLOAT v)
	{
		return FVector3{ vec.x / v, vec.y / v, vec.z / v };
	}

	inline bool operator==(const FVector3& lhs, const FVector3& rhs)
	{
		return (lhs.x == rhs.x) && (lhs.y == rhs.y) && (lhs.z == rhs.z);
	}

	inline bool operator!=(const FVector3& lhs, const FVector3& rhs)
	{
		return !(lhs == rhs);
	}
}