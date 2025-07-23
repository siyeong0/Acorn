#pragma once
#include <cmath>
#include <limits>
#include <cassert>

namespace aco
{
	using FLOAT = float;

	struct FVector2
	{
		FLOAT x;
		FLOAT y;

		FVector2() = default;
		~FVector2() = default;

		FVector2(FLOAT x, FLOAT y);
		FVector2(const FVector2& other);
		FVector2& operator=(const FVector2& other);
		inline FLOAT& operator[](size_t idx);
		inline const FLOAT& operator[](size_t idx) const;

		static inline FVector2 Zero() { return FVector2{ 0.f, 0.f }; }
		static inline FVector2 One() { return FVector2{ 1.f, 1.f }; }
		static inline FVector2 Right() { return FVector2{ 1.f, 0.f }; }
		static inline FVector2 Left() { return FVector2{ -1.f, 0.f }; }
		static inline FVector2 Up() { return FVector2{ 0.f, 1.f }; }
		static inline FVector2 Down() { return FVector2{ 0.f, -1.f }; }
		static inline FVector2 FMaxValue() { constexpr FLOAT v = std::numeric_limits<FLOAT>::max(); return FVector2{ v, v }; }
		static inline FVector2 FMinValue() { constexpr FLOAT v = std::numeric_limits<FLOAT>::lowest(); return FVector2{ v, v }; }

		inline FLOAT Dot(const FVector2& other) const;
		static inline FLOAT Dot(const FVector2& a, const FVector2& b);

		inline FLOAT Magnitude() const;
		inline FLOAT SqrMagnitude() const;
		inline FLOAT Length() const;
		inline FVector2 Perpendicular() const { return FVector2{ -y, x }; }

		inline FVector2 Normalized() const;
		inline void Normalize();

		inline void operator+=(const FVector2& other);
		inline void operator-=(const FVector2& other);
		inline void operator*=(FLOAT v);
		inline void operator/=(FLOAT v);

		static inline FVector2 Abs(const FVector2& vec);
		static inline FVector2 Min(const FVector2& a, const FVector2& b);
		static inline FVector2 Max(const FVector2& a, const FVector2& b);
		static inline FVector2 Clamp(const FVector2& value, const FVector2& min, const FVector2& max);
		static inline FVector2 Lerp(const FVector2& a, const FVector2& b, FLOAT t);
		static inline FVector2 SmoothStep(const FVector2& a, const FVector2& b, FLOAT t);
	};

	inline FVector2 operator-(const FVector2& vec) { return FVector2{ -vec.x, -vec.y }; }
	inline FVector2 operator+(const FVector2& lhs, const FVector2& rhs) { return FVector2{ lhs.x + rhs.x, lhs.y + rhs.y }; }
	inline FVector2 operator-(const FVector2& lhs, const FVector2& rhs) { return FVector2{ lhs.x - rhs.x, lhs.y - rhs.y }; }
	inline FVector2 operator*(const FVector2& lhs, const FVector2& rhs) { return FVector2{ lhs.x * rhs.x, lhs.y * rhs.y }; }
	inline FVector2 operator*(FLOAT v, const FVector2& vec) { return FVector2{ v * vec.x, v * vec.y }; }
	inline FVector2 operator*(const FVector2& vec, FLOAT v) { return FVector2{ v * vec.x, v * vec.y }; }
	inline FVector2 operator/(const FVector2& vec, FLOAT v) { return FVector2{ vec.x / v, vec.y / v }; }
	inline bool operator==(const FVector2& lhs, const FVector2& rhs) { return lhs.x == rhs.x && lhs.y == rhs.y; }
	inline bool operator!=(const FVector2& lhs, const FVector2& rhs) { return !(lhs == rhs); }

	// --- Implementation ---

	inline FVector2::FVector2(FLOAT x, FLOAT y) : x(x), y(y) {}
	inline FVector2::FVector2(const FVector2& other) : x(other.x), y(other.y) {}
	inline FVector2& FVector2::operator=(const FVector2& other)
	{
		x = other.x;
		y = other.y;
		return *this;
	}

	inline FLOAT& FVector2::operator[](size_t idx)
	{
		assert(idx < 2);
		return (&x)[idx];
	}

	inline const FLOAT& FVector2::operator[](size_t idx) const
	{
		assert(idx < 2);
		return (&x)[idx];
	}

	inline FLOAT FVector2::Dot(const FVector2& other) const
	{
		return x * other.x + y * other.y;
	}

	inline FLOAT FVector2::Dot(const FVector2& a, const FVector2& b)
	{
		return a.Dot(b);
	}

	inline FLOAT FVector2::Magnitude() const
	{
		return std::sqrt(SqrMagnitude());
	}

	inline FLOAT FVector2::SqrMagnitude() const
	{
		return x * x + y * y;
	}

	inline FLOAT FVector2::Length() const
	{
		return Magnitude();
	}

	inline FVector2 FVector2::Normalized() const
	{
		FLOAT len = Length();
		return (len != 0) ? *this / len : FVector2{ 0.f, 0.f };
	}

	inline void FVector2::Normalize()
	{
		FLOAT len = Length();
		if (len != 0) *this /= len;
	}

	inline void FVector2::operator+=(const FVector2& other) { *this = *this + other; }
	inline void FVector2::operator-=(const FVector2& other) { *this = *this - other; }
	inline void FVector2::operator*=(FLOAT v) { *this = *this * v; }
	inline void FVector2::operator/=(FLOAT v) { *this = *this / v; }

	inline FVector2 FVector2::Abs(const FVector2& vec)
	{
		return FVector2{ std::fabs(vec.x), std::fabs(vec.y) };
	}

	inline FVector2 FVector2::Min(const FVector2& a, const FVector2& b)
	{
		return FVector2{ std::fmin(a.x, b.x), std::fmin(a.y, b.y) };
	}

	inline FVector2 FVector2::Max(const FVector2& a, const FVector2& b)
	{
		return FVector2{ std::fmax(a.x, b.x), std::fmax(a.y, b.y) };
	}

	inline FVector2 FVector2::Clamp(const FVector2& value, const FVector2& min, const FVector2& max)
	{
		return FVector2{
			std::fmax(min.x, std::fmin(value.x, max.x)),
			std::fmax(min.y, std::fmin(value.y, max.y))
		};
	}

	inline FVector2 FVector2::Lerp(const FVector2& a, const FVector2& b, FLOAT t)
	{
		t = std::fmax(0.f, std::fmin(t, 1.f));
		return FVector2{
			a.x + (b.x - a.x) * t,
			a.y + (b.y - a.y) * t
		};
	}

	inline FVector2 FVector2::SmoothStep(const FVector2& a, const FVector2& b, FLOAT t)
	{
		t = std::fmax(0.f, std::fmin(t, 1.f));
		FLOAT t2 = t * t;
		FLOAT t3 = t2 * t;
		FLOAT smoothT = t2 * (3.f - 2.f * t);
		return FVector2{
			a.x + (b.x - a.x) * smoothT,
			a.y + (b.y - a.y) * smoothT
		};
	}
}
