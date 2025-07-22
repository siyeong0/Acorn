#pragma once
#include <string>
#include <cassert>

#include "Math/FVector4.h"

namespace aco
{
	struct Texture
	{
		std::string Name;
		std::string Path;

		int Width = 0;
		int Height = 0;
		int Channels = 0;

		std::vector<FVector4> Data;

		const FVector4& operator()(int w, int h) const
		{
			assert(w >= 0 && w < Width && h >= 0 && h < Height);
			return Data[h * Width + w];
		}

		void FlipY()
		{
			for (int h = 0; h < Height / 2; ++h)
			{
				for (int w = 0; w < Width; ++w)
				{
					std::swap(Data[h * Width + w], Data[(Height - 1 - h) * Width + w]);
				}
			}
		}
	};
}