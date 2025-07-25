#include "Raytracer.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Debug.h"
#include "Math/CuMath.cuh"
#include "Graphics/Renderer/WindowsSecurityAttributes/WindowsSecurityAttributes.h"
#include "Graphics/Voxel/Avox.h"

namespace aco
{
	namespace gfx
	{
		// convert floating point rgba color to 32-bit integer
		__device__ unsigned int rgbaFloatToInt(float4 rgba)
		{
			rgba.x = __saturatef(rgba.x); // clamp to [0.0, 1.0]
			rgba.y = __saturatef(rgba.y);
			rgba.z = __saturatef(rgba.z);
			rgba.w = __saturatef(rgba.w);
			return ((unsigned int)(rgba.w * 255.0f) << 24) | ((unsigned int)(rgba.z * 255.0f) << 16) |
				((unsigned int)(rgba.y * 255.0f) << 8) | ((unsigned int)(rgba.x * 255.0f));
		}

		__device__ float4 rgbaIntToFloat(unsigned int c)
		{
			float4 rgba;
			rgba.x = (c & 0xff) * 0.003921568627f;         //  /255.0f;
			rgba.y = ((c >> 8) & 0xff) * 0.003921568627f;  //  /255.0f;
			rgba.z = ((c >> 16) & 0xff) * 0.003921568627f; //  /255.0f;
			rgba.w = ((c >> 24) & 0xff) * 0.003921568627f; //  /255.0f;
			return rgba;
		}

		struct Material
		{
			float4 Color;
			float4 EmissionColor;
			float4 SpecularColor;
			float EmissionStrength;
			float Smoothness;
			float SpecularProbability;
			int Flag;
		};

		struct Sphere
		{
			float3 Center;
			float Radius;
			Material Material;
		};

		constexpr int NUM_OBJECTS = 84;

		__constant__ Sphere dev_SceneList[NUM_OBJECTS];
		__constant__ int dev_NumObjects;

		Sphere host_SceneList[NUM_OBJECTS] = {
			Sphere{ make_float3(0.000000, -1000.000000, 0.000000), 1000.000000f, Material{
				make_float4(0.500000, 0.500000, 0.500000, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(-7.995381, 0.200000, -7.478668), 0.200000f, Material{
				make_float4(0.380012, 0.506085, 0.762437, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(-7.696819, 0.200000, -5.468978), 0.200000f, Material{
				make_float4(0.596282, 0.140784, 0.017972, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(-7.824804, 0.200000, -3.120637), 0.200000f, Material{
				make_float4(0.288507, 0.465652, 0.665070, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(-7.132909, 0.200000, -1.701323), 0.200000f, Material{
				make_float4(0.101047, 0.293493, 0.813446, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(-7.569523, 0.200000, 0.494554), 0.200000f, Material{
				make_float4(0.365924, 0.221622, 0.058332, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(-7.730332, 0.200000, 2.358976), 0.200000f, Material{
				make_float4(0.051231, 0.430547, 0.454086, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(-7.892865, 0.200000, 4.753728), 0.200000f, Material{
				make_float4(0.826684, 0.820511, 0.908836, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				1.0f, 0.389611f, 1.0f, 0
			} },
			Sphere{ make_float3(-7.656691, 0.200000, 6.888913), 0.200000f, Material{
				make_float4(0.346542, 0.225385, 0.180132, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(-7.217835, 0.200000, 8.203466), 0.200000f, Material{
				make_float4(0.600463, 0.582386, 0.608277, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				1.0f, 0.427369f, 1.0f, 0
			} },
			Sphere{ make_float3(-5.115232, 0.200000, -7.980404), 0.200000f, Material{
				make_float4(0.256969, 0.138639, 0.080293, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(-5.323222, 0.200000, -5.113037), 0.200000f, Material{
				make_float4(0.193093, 0.510542, 0.613362, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(-5.410681, 0.200000, -3.527741), 0.200000f, Material{
				make_float4(0.352200, 0.191551, 0.115972, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(-5.460670, 0.200000, -1.166543), 0.200000f, Material{
				make_float4(0.029486, 0.249874, 0.077989, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(-5.457659, 0.200000, 0.363870), 0.200000f, Material{
				make_float4(0.395713, 0.762043, 0.108515, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(-5.798715, 0.200000, 2.161684), 0.200000f, Material{
				make_float4(0.000000, 0.000000, 0.000000, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(-5.116586, 0.200000, 4.470188), 0.200000f, Material{
				make_float4(0.059444, 0.404603, 0.171767, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(-5.273591, 0.200000, 6.795187), 0.200000f, Material{
				make_float4(0.499454, 0.131330, 0.158348, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(-5.120286, 0.200000, 8.731398), 0.200000f, Material{
				make_float4(0.267365, 0.136024, 0.300483, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(-3.601565, 0.200000, -7.895600), 0.200000f, Material{
				make_float4(0.027752, 0.155209, 0.330428, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(-3.735860, 0.200000, -5.163056), 0.200000f, Material{
				make_float4(0.576768, 0.884712, 0.993335, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				1.0f, 0.359385f, 1.0f, 0
			} },
			Sphere{ make_float3(-3.481116, 0.200000, -3.794556), 0.200000f, Material{
				make_float4(0.405104, 0.066436, 0.009339, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(-3.866858, 0.200000, -1.465965), 0.200000f, Material{
				make_float4(0.027570, 0.021652, 0.252798, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(-3.168870, 0.200000, 0.553099), 0.200000f, Material{
				make_float4(0.421992, 0.107577, 0.177504, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(-3.428552, 0.200000, 2.627547), 0.200000f, Material{
				make_float4(0.974029, 0.653443, 0.571877, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				1.0f, 0.312780f, 1.0f, 0
			} },
			Sphere{ make_float3(-3.771736, 0.200000, 4.324785), 0.200000f, Material{
				make_float4(0.685957, 0.000043, 0.181270, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(-3.768522, 0.200000, 6.384588), 0.200000f, Material{
				make_float4(0.025972, 0.082246, 0.138765, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(-3.286992, 0.200000, 8.441148), 0.200000f, Material{
				make_float4(0.186577, 0.560376, 0.367045, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(-1.552127, 0.200000, -7.728200), 0.200000f, Material{
				make_float4(0.202998, 0.002459, 0.015350, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(-1.360796, 0.200000, -5.346098), 0.200000f, Material{
				make_float4(0.690820, 0.028470, 0.179907, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(-1.287209, 0.200000, -3.735321), 0.200000f, Material{
				make_float4(0.345974, 0.672353, 0.450180, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(-1.344859, 0.200000, -1.726654), 0.200000f, Material{
				make_float4(0.209209, 0.431116, 0.164732, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(-1.974774, 0.200000, 0.183260), 0.200000f, Material{
				make_float4(0.006736, 0.675637, 0.622067, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(-1.542872, 0.200000, 2.067868), 0.200000f, Material{
				make_float4(0.192247, 0.016661, 0.010109, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(-1.743856, 0.200000, 4.752810), 0.200000f, Material{
				make_float4(0.295270, 0.108339, 0.276513, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(-1.955621, 0.200000, 6.493702), 0.200000f, Material{
				make_float4(0.270527, 0.270494, 0.202029, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(-1.350449, 0.200000, 8.068503), 0.200000f, Material{
				make_float4(0.646942, 0.501660, 0.573693, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				1.0f, 0.346551f, 1.0f, 0
			} },
			Sphere{ make_float3(0.706123, 0.200000, -7.116040), 0.200000f, Material{
				make_float4(0.027695, 0.029917, 0.235781, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(0.897766, 0.200000, -5.938681), 0.200000f, Material{
				make_float4(0.114934, 0.046258, 0.039647, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(0.744113, 0.200000, -3.402960), 0.200000f, Material{
				make_float4(0.513631, 0.335578, 0.204787, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(0.867750, 0.200000, -1.311908), 0.200000f, Material{
				make_float4(0.400246, 0.000956, 0.040513, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(0.082480, 0.200000, 0.838206), 0.200000f, Material{
				make_float4(0.594141, 0.215068, 0.025718, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(0.649692, 0.200000, 2.525103), 0.200000f, Material{
				make_float4(0.602157, 0.797249, 0.614694, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				1.0f, 0.341860f, 1.0f, 0
			} },
			Sphere{ make_float3(0.378574, 0.200000, 4.055579), 0.200000f, Material{
				make_float4(0.005086, 0.003349, 0.064403, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(0.425844, 0.200000, 6.098526), 0.200000f, Material{
				make_float4(0.266812, 0.016602, 0.000853, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(0.261365, 0.200000, 8.661150), 0.200000f, Material{
				make_float4(0.150201, 0.007353, 0.152506, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(2.814218, 0.200000, -7.751227), 0.200000f, Material{
				make_float4(0.570094, 0.610319, 0.584192, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				1.0f, 0.018611f, 1.0f, 0
			} },
			Sphere{ make_float3(2.050073, 0.200000, -5.731364), 0.200000f, Material{
				make_float4(0.109886, 0.029498, 0.303265, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(2.020130, 0.200000, -3.472627), 0.200000f, Material{
				make_float4(0.216908, 0.216448, 0.221775, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(2.884277, 0.200000, -1.232662), 0.200000f, Material{
				make_float4(0.483428, 0.027275, 0.113898, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(2.644454, 0.200000, 0.596324), 0.200000f, Material{
				make_float4(0.005872, 0.860718, 0.561933, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(2.194283, 0.200000, 2.880603), 0.200000f, Material{
				make_float4(0.452710, 0.824152, 0.045179, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(2.281000, 0.200000, 4.094307), 0.200000f, Material{
				make_float4(0.002091, 0.145849, 0.032535, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(2.080841, 0.200000, 6.716384), 0.200000f, Material{
				make_float4(0.468539, 0.032772, 0.018071, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(2.287131, 0.200000, 8.583242), 0.200000f, Material{
				make_float4(0.000000, 0.000000, 0.000000, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(4.329136, 0.200000, -7.497218), 0.200000f, Material{
				make_float4(0.030865, 0.071452, 0.016051, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(4.502115, 0.200000, -5.941060), 0.200000f, Material{
				make_float4(0.000000, 0.000000, 0.000000, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(4.750631, 0.200000, -3.836759), 0.200000f, Material{
				make_float4(0.702578, 0.084798, 0.141374, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(4.082084, 0.200000, -1.180746), 0.200000f, Material{
				make_float4(0.043052, 0.793077, 0.018707, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(4.429173, 0.200000, 2.069721), 0.200000f, Material{
				make_float4(0.179009, 0.147750, 0.617371, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(4.277152, 0.200000, 4.297482), 0.200000f, Material{
				make_float4(0.422693, 0.011222, 0.211945, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(4.012743, 0.200000, 6.225072), 0.200000f, Material{
				make_float4(0.986275, 0.073358, 0.133628, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(4.047066, 0.200000, 8.419360), 0.200000f, Material{
				make_float4(0.878749, 0.677170, 0.684995, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				1.0f, 0.243932f, 1.0f, 0
			} },
			Sphere{ make_float3(6.441846, 0.200000, -7.700798), 0.200000f, Material{
				make_float4(0.309255, 0.342524, 0.489512, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(6.047810, 0.200000, -5.519369), 0.200000f, Material{
				make_float4(0.532361, 0.008200, 0.077522, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(6.779211, 0.200000, -3.740542), 0.200000f, Material{
				make_float4(0.161234, 0.539314, 0.016667, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(6.430776, 0.200000, -1.332107), 0.200000f, Material{
				make_float4(0.641951, 0.661402, 0.326114, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(6.476387, 0.200000, 0.329973), 0.200000f, Material{
				make_float4(0.033000, 0.648388, 0.166911, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(6.568686, 0.200000, 2.116949), 0.200000f, Material{
				make_float4(0.590952, 0.072292, 0.125672, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(6.371189, 0.200000, 4.609841), 0.200000f, Material{
				make_float4(0.870345, 0.753830, 0.933118, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				1.0f, 0.233489f, 1.0f, 0
			} },
			Sphere{ make_float3(6.011877, 0.200000, 6.569579), 0.200000f, Material{
				make_float4(0.044868, 0.651697, 0.086779, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(6.096087, 0.200000, 8.892333), 0.200000f, Material{
				make_float4(0.588587, 0.078723, 0.044928, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(8.185763, 0.200000, -7.191109), 0.200000f, Material{
				make_float4(0.989702, 0.886784, 0.540759, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				1.0f, 0.104229f, 1.0f, 0
			} },
			Sphere{ make_float3(8.411960, 0.200000, -5.285309), 0.200000f, Material{
				make_float4(0.139604, 0.022029, 0.461688, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(8.047109, 0.200000, -3.427552), 0.200000f, Material{
				make_float4(0.815002, 0.631228, 0.806757, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				1.0f, 0.150782f, 1.0f, 0
			} },
			Sphere{ make_float3(8.119639, 0.200000, -1.652587), 0.200000f, Material{
				make_float4(0.177852, 0.429797, 0.042251, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(8.818120, 0.200000, 0.401292), 0.200000f, Material{
				make_float4(0.065416, 0.087694, 0.040518, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(8.754155, 0.200000, 2.152549), 0.200000f, Material{
				make_float4(0.230659, 0.035665, 0.435895, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(8.595298, 0.200000, 4.802001), 0.200000f, Material{
				make_float4(0.188493, 0.184933, 0.040215, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(8.036216, 0.200000, 6.739752), 0.200000f, Material{
				make_float4(0.023192, 0.364636, 0.464844, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(8.256561, 0.200000, 8.129115), 0.200000f, Material{
				make_float4(0.002612, 0.598319, 0.435378, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(0.000000, 1.000000, 0.000000), 1.000000f, Material{
				make_float4(0.000000, 0.000000, 0.000000, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(-4.000000, 1.000000, 0.000000), 1.000000f, Material{
				make_float4(0.400000, 0.200000, 0.100000, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(4.000000, 1.000000, 0.000000), 1.000000f, Material{
				make_float4(0.700000, 0.600000, 0.500000, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				make_float4(1.0f, 1.0f, 1.0f, 1.0f),
				1.0f, 0.000000f, 1.0f, 0
			} }
		};

		struct Ray
		{
			float3 Origin;
			float3 Dir;
		};

		struct HitInfo
		{
			bool bHit;
			float Distance;
			float3 HitPoint;
			float3 Normal;
			Material Material;
		};

		__host__ __device__ float randNormalDistribution(unsigned int& state)
		{
			float theta = 2.0 * 3.14159265358979323846 * cumath::Rand(state);
			float rho = sqrt(-2.0 * log(cumath::Rand(state)));
			return rho * cos(theta);
		}

		__host__ __device__ float3 randDirection(unsigned int& state)
		{
			float x = randNormalDistribution(state);
			float y = randNormalDistribution(state);
			float z = randNormalDistribution(state);
			return cumath::Normalize(make_float3(x, y, z));
		}

		__host__ __device__ float3 getEnvironmentLight(const Ray& ray)
		{
			float t = 0.5f * (ray.Dir.y + 1.0f);

			return cumath::Lerp(make_float3(1.0f, 1.0f, 1.0f), make_float3(0.5f, 0.7f, 1.0f), t);
		}

		__host__ __device__ HitInfo RaySphere(const Ray& ray, const Sphere& sphere)
		{
			HitInfo hitInfo = {};
			float3 offsetToRayOrigin = ray.Origin - sphere.Center;

			float a = cumath::Dot(ray.Dir, ray.Dir);
			float b = 2.0 * cumath::Dot(offsetToRayOrigin, ray.Dir);
			float c = cumath::Dot(offsetToRayOrigin, offsetToRayOrigin) - sphere.Radius * sphere.Radius;
			float discriminant = b * b - 4 * a * c;

			if (discriminant >= 0.0)
			{
				float dist = (-b - sqrt(discriminant)) / (2.0 * a);

				if (dist > 0.0)
				{
					hitInfo.bHit = true;
					hitInfo.Distance = dist;
					hitInfo.HitPoint = ray.Origin + ray.Dir * dist;
					hitInfo.Material = sphere.Material;
					hitInfo.Normal = cumath::Normalize(hitInfo.HitPoint - sphere.Center);
				}
			}

			return hitInfo;
		}

		__host__ __device__ HitInfo RayScene(const Ray& ray)
		{
			HitInfo closestHit = {};
			closestHit.Distance = 9999.9f;
			for (int i = 0; i < dev_NumObjects; ++i)
			{
				const Sphere& sphere = dev_SceneList[i];
				HitInfo hitinfo = RaySphere(ray, sphere);

				if (hitinfo.bHit && hitinfo.Distance < closestHit.Distance)
				{
					closestHit = hitinfo;
				}
			}
			return closestHit;
		}

#define MAX_BOUNCE 4
#ifdef _DEBUG
#define SAMPLE_PIR_PIX 2
#else
#define SAMPLE_PIR_PIX 16
#endif

		__host__ __device__ float3 Trace(Ray ray, unsigned int& state)
		{
			float3 incomingLight = make_float3(0.0f, 0.0f, 0.0f);
			float3 rayColor = make_float3(1.0f, 1.0f, 1.0f);

			for (int bounce = 0; bounce < MAX_BOUNCE; ++bounce)
			{
				HitInfo hitinfo = RayScene(ray);
				if (hitinfo.bHit)
				{
					Material material = hitinfo.Material;

					bool bSpecularBounce = material.SpecularProbability >= cumath::Rand(state);

					ray.Origin = hitinfo.HitPoint;
					float3 diffuseDir = cumath::Normalize(hitinfo.Normal + randDirection(state));
					float3 specularDir = cumath::Reflect(ray.Dir, hitinfo.Normal);
					ray.Dir = cumath::Normalize(cumath::Lerp(diffuseDir, specularDir, material.Smoothness * bSpecularBounce));

					float3 emittedLight = to_float3(material.EmissionColor * material.EmissionStrength);
					incomingLight += emittedLight * rayColor;
					rayColor *= cumath::Lerp(to_float3(material.Color), to_float3(material.SpecularColor), (float)bSpecularBounce);

					float p = cumath::Max(rayColor.x, cumath::Max(rayColor.y, rayColor.z));
					if (cumath::Rand(state) >= p) break;
					rayColor *= 1.0f / p;
				}
				else
				{
					incomingLight += getEnvironmentLight(ray) * rayColor;
					break;
				}
			}

			return rayColor;
		}

		__global__ void TraceRayKernel(
			cudaSurfaceObject_t* dstSurface, size_t baseWidth, size_t baseHeight,
			float4x4 cameraMatrix, float3 viewParams)
		{
			unsigned int ux = blockIdx.x * blockDim.x + threadIdx.x;
			unsigned int uy = blockIdx.y * blockDim.y + threadIdx.y;

			if (ux >= baseWidth || uy >= baseHeight) return;

			float fu = ((float)ux / (float)baseWidth);
			float fv = 1.0f - ((float)uy / (float)baseHeight);

			// Row-major
			float3 cameraPos = make_float3(cameraMatrix._m30, cameraMatrix._m31, cameraMatrix._m32);

			Ray ray;
			ray.Origin = cameraPos;
			float3 viewPointLocal = make_float3((fu - 0.5f) * viewParams.x, (fv - 0.5f) * viewParams.y, viewParams.z);
			float4 viewPointTemp = cumath::Mul(make_float4(viewPointLocal.x, viewPointLocal.y, viewPointLocal.z, 1.0f), cameraMatrix);
			float3 viewPoint = make_float3(viewPointTemp.x, viewPointTemp.y, viewPointTemp.z);
			ray.Dir = cumath::Normalize(viewPoint - ray.Origin);

			// Render color
			unsigned int randState = uy * baseWidth + ux;
			float3 totalIncomingLight = make_float3(0.0f, 0.0f, 0.0f);
			for (int i = 0; i < SAMPLE_PIR_PIX; ++i)
			{
				totalIncomingLight += Trace(ray, randState);
			}

			float3 pixelColor = totalIncomingLight / SAMPLE_PIR_PIX;

			surf2Dwrite(rgbaFloatToInt(make_float4(pixelColor.x, pixelColor.y, pixelColor.z, 1.0f)), dstSurface[0], 4 * ux, uy);
		}

		void Raytracer::Blit(CudaRenderBuffer* renderTarget, int width, int height, cudaStream_t streamToRun)
		{
			static bool bInit = true;
			if (bInit)
			{
				CUDA_CHECK(cudaMemcpyToSymbol(dev_SceneList, host_SceneList, sizeof(Sphere) * NUM_OBJECTS, 0, cudaMemcpyHostToDevice));
				CUDA_CHECK(cudaMemcpyToSymbol(dev_NumObjects, &NUM_OBJECTS, sizeof(int), 0, cudaMemcpyHostToDevice));
				bInit = false;
			}

			int nthreads = 16;
			dim3 dimBlock(nthreads, nthreads, 1); // dimBlock.x * dimBlock.y * dimBlock.z <= 1024
			dim3 dimGrid(
				(uint32_t)ceil(float(width) / dimBlock.x),
				(uint32_t)ceil(float(height) / dimBlock.y),
				1);

			FVector3 eye = { 13.0f, 2.0f, 4.0f };
			FVector3 center = { 0.0f, 0.0f, 0.0f };
			FVector3 up = { 0.0f, 1.0f, 0.0f };

			float4x4 cameraMatrix;
			FMatrix4x4 cameraMatrixFMat = Mat4x4Camera(eye, center, up);
			memcpy(&cameraMatrix, &cameraMatrixFMat.Transposed(), sizeof(float) * 16);

			float fovYDeg = 40.0f;
			float aspectRatio = 16.0f / 9.0f;
			float clipZ = 0.3f;

			float halfFovRad = fovYDeg * 0.5f * (3.14159265f / 180.0f); // deg → rad
			float planeHeight = 2.0f * clipZ * tanf(halfFovRad);        // 수직 크기
			float planeWidth = planeHeight * aspectRatio;
			float3 viewParams = make_float3(planeWidth, planeHeight, clipZ);

			TraceRayKernel << <dimGrid, dimBlock, 0, streamToRun >> > (
				renderTarget->GetDeviceSurfacePointer(),
				width, height, cameraMatrix, viewParams);
		}
	}
}