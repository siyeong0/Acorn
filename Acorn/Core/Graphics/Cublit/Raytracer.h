#pragma once
#include "Graphics/Renderer/CUDA/ICublit.h"

namespace aco
{
	class Raytracer : public gfx::ICublit
	{
	public:
		Raytracer() = default;
		virtual ~Raytracer() = default;

		virtual void Blit(gfx::CudaRenderBuffer* renderTarget, int width, int height, cudaStream_t streamToRun) override;
	};
}