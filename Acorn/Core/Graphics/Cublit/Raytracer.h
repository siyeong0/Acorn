#pragma once
#include "Graphics/Renderer/CUDA/ICublit.h"

namespace aco
{
	namespace gfx
	{
		class Raytracer : public ICublit
		{
		public:
			Raytracer() = default;
			virtual ~Raytracer() = default;

			virtual void Blit(CudaRenderBuffer* renderTarget, int width, int height, cudaStream_t streamToRun) override;
		};
	}
}