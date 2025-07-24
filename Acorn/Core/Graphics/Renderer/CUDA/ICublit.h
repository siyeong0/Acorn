#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "CudaRenderBuffer.h"

namespace aco
{
	namespace gfx
	{
		class ICublit
		{
		public:
			ICublit() = default;
			virtual ~ICublit() = default;

			virtual void Blit(CudaRenderBuffer* renderTarget, int width, int height, cudaStream_t streamToRun = 0) = 0;
		};
	}
}