#pragma once
#include <cstdint>

#include "Math/Math.h"

namespace aco
{
	struct AvoxNode
	{
		union
		{
			uint32_t Data;
			struct
			{
				uint8_t ChildMask;
				uint8_t PageHeader;
				uint16_t ChildPointer;
			};
			struct
			{
				uint16_t Color;
				uint16_t ColorDummy;
			};
		};
	};
	static_assert(sizeof(AvoxNode) == 4, "SVO node size must be 8 bytes.");

	inline FVector3 UnpackRGB565(uint16_t packed)
	{
		float r = ((packed >> 11) & 0x1F) / 31.0f;
		float g = ((packed >> 5) & 0x3F) / 63.0f;
		float b = (packed & 0x1F) / 31.0f;
		return FVector3{ r,g,b };
	}
}