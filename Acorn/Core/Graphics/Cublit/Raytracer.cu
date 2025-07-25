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
			float Roughness;
			float SpecularProbability;
			int Flag;
		};

		struct Sphere
		{
			float3 Center;
			float Radius;
			Material Material;
		};

		constexpr int NUM_OBJECTS = 74;

		__constant__ Sphere dev_SceneList[NUM_OBJECTS];
		__constant__ int dev_NumObjects;

		Sphere host_SceneList[NUM_OBJECTS] = {
			Sphere{ make_float3(0.000000f, -1000.000000f, 0.000000f), 1000.000000f, Material{
				make_float4(0.500000f, 0.500000f, 0.500000f, 1.0f),
				make_float4(0.0f, 0.0f, 0.0f, 1.0f),
				make_float4(0.500000f, 0.500000f, 0.500000f, 1.0f),
				0.0f, 0.2f, 0.4f, 0
			} },
			Sphere{ make_float3(-7.995381f, 0.200000f, -7.478668f), 0.200000f, Material{
				make_float4(0.380012f, 0.506085f, 0.762437f, 1.0f),
				make_float4(0.96f, 0.96f, 0.96f, 1.0f),
				make_float4(0.96f, 0.96f, 0.96f, 1.0f),
				0.0f, 4.5f, 0.0f, 0
			} },
			Sphere{ make_float3(-7.696819f, 0.200000f, -5.468978f), 0.200000f, Material{
				make_float4(0.596282f, 0.140784f, 0.017972f, 1.0f),
				make_float4(0.0f, 0.0f, 0.0f, 1.0f),
				make_float4(0.96f, 0.96f, 0.96f, 1.0f),
				0.0f, 2.4f, 0.0f, 0
			} },
			Sphere{ make_float3(-7.824804f, 0.200000f, -3.120637f), 0.200000f, Material{
				make_float4(0.288507f, 0.465652f, 0.665070f, 1.0f),
				make_float4(0.0f, 0.0f, 0.0f, 1.0f),
				make_float4(0.96f, 0.96f, 0.96f, 1.0f),
				0.0f, 9.2f, 0.0f, 0
			} },
			Sphere{ make_float3(-7.132909f, 0.200000f, -1.701323f), 0.200000f, Material{
				make_float4(0.101047f, 0.293493f, 0.813446f, 1.0f),
				make_float4(0.0f, 0.0f, 0.0f, 1.0f),
				make_float4(0.96f, 0.96f, 0.96f, 1.0f),
				0.0f, 1.0f, 0.0f, 0
			} },
			Sphere{ make_float3(-7.569523f, 0.200000f, 0.494554f), 0.200000f, Material{
				make_float4(0.365924f, 0.221622f, 0.058332f, 1.0f),
				make_float4(0.0f, 0.0f, 0.0f, 1.0f),
				make_float4(0.96f, 0.96f, 0.96f, 1.0f),
				0.0f, 0.2f, 0.0f, 0
			} },
			Sphere{ make_float3(-7.730332f, 0.200000f, 2.358976f), 0.200000f, Material{
				make_float4(0.051231f, 0.430547f, 0.454086f, 1.0f),
				make_float4(0.0f, 0.0f, 0.0f, 1.0f),
				make_float4(0.96f, 0.96f, 0.96f, 1.0f),
				0.0f, 4.7f, 0.0f, 0
			} },
			Sphere{ make_float3(-7.892865f, 0.200000f, 4.753728f), 0.200000f, Material{
				make_float4(0.826684f, 0.820511f, 0.908836f, 1.0f),
				make_float4(0.0f, 0.0f, 0.0f, 1.0f),
				make_float4(0.96f, 0.96f, 0.96f, 1.0f),
				1.0f, 0.389611f, 1.0f, 0
			} },
			Sphere{ make_float3(-7.656691f, 0.200000f, 6.888913f), 0.200000f, Material{
				make_float4(0.346542f, 0.225385f, 0.180132f, 1.0f),
				make_float4(0.0f, 0.0f, 0.0f, 1.0f),
				make_float4(0.96f, 0.96f, 0.96f, 1.0f),
				0.0f, 0.02f, 0.0f, 0
			} },
			Sphere{ make_float3(-7.217835f, 0.200000f, 8.203466f), 0.200000f, Material{
				make_float4(0.600463f, 0.582386f, 0.608277f, 1.0f),
				make_float4(0.4f, 0.2f, 0.95f, 1.0f),
				make_float4(0.96f, 0.96f, 0.96f, 1.0f),
				1.0f, 1.0f, 1.0f, 0
			} },
			Sphere{ make_float3(-5.115232f, 0.200000f, -7.980404f), 0.200000f, Material{
				make_float4(0.256969f, 0.138639f, 0.080293f, 1.0f),
				make_float4(0.0f, 0.0f, 0.0f, 1.0f),
				make_float4(0.96f, 0.96f, 0.96f, 1.0f),
				0.0f, 0.24f, 0.0f, 0
			} },
			Sphere{ make_float3(-5.323222f, 0.200000f, -5.113037f), 0.200000f, Material{
				make_float4(0.193093f, 0.510542f, 0.613362f, 1.0f),
				make_float4(0.0f, 0.0f, 0.0f, 1.0f),
				make_float4(0.96f, 0.96f, 0.96f, 1.0f),
				0.0f, 0.89f, 0.0f, 0
			} },
			Sphere{ make_float3(-5.410681f, 0.200000f, -3.527741f), 0.200000f, Material{
				make_float4(0.352200f, 0.191551f, 0.115972f, 1.0f),
				make_float4(0.0f, 0.0f, 0.0f, 1.0f),
				make_float4(0.96f, 0.96f, 0.96f, 1.0f),
				0.0f, 0.77f, 0.0f, 0
			} },
			Sphere{ make_float3(-5.460670f, 0.200000f, -1.166543f), 0.200000f, Material{
				make_float4(0.029486f, 0.249874f, 0.077989f, 1.0f),
				make_float4(0.0f, 0.0f, 0.0f, 1.0f),
				make_float4(0.96f, 0.96f, 0.96f, 1.0f),
				0.0f, 0.3f, 0.0f, 0
			} },
			Sphere{ make_float3(-5.457659f, 0.200000f, 0.363870f), 0.200000f, Material{
				make_float4(0.395713f, 0.762043f, 0.108515f, 1.0f),
				make_float4(0.0f, 0.0f, 0.0f, 1.0f),
				make_float4(0.96f, 0.96f, 0.96f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(-5.798715f, 0.200000f, 2.161684f), 0.200000f, Material{
				make_float4(0.000000f, 0.000000f, 0.000000f, 1.0f),
				make_float4(0.0f, 0.0f, 0.0f, 1.0f),
				make_float4(0.96f, 0.96f, 0.96f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(-5.116586f, 0.200000f, 4.470188f), 0.200000f, Material{
				make_float4(0.059444f, 0.404603f, 0.171767f, 1.0f),
				make_float4(0.0f, 0.0f, 0.0f, 1.0f),
				make_float4(0.96f, 0.96f, 0.96f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(-5.273591f, 0.200000f, 6.795187f), 0.200000f, Material{
				make_float4(0.499454f, 0.131330f, 0.158348f, 1.0f),
				make_float4(0.0f, 0.0f, 0.0f, 1.0f),
				make_float4(0.96f, 0.96f, 0.96f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(-5.120286f, 0.200000f, 8.731398f), 0.200000f, Material{
				make_float4(0.267365f, 0.136024f, 0.300483f, 1.0f),
				make_float4(0.0f, 0.0f, 0.0f, 1.0f),
				make_float4(0.96f, 0.96f, 0.96f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(-3.601565f, 0.200000f, -7.895600f), 0.200000f, Material{
				make_float4(0.027752f, 0.155209f, 0.330428f, 1.0f),
				make_float4(0.0f, 0.0f, 0.0f, 1.0f),
				make_float4(0.96f, 0.96f, 0.96f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(-3.735860f, 0.200000f, -5.163056f), 0.200000f, Material{
				make_float4(0.576768f, 0.884712f, 0.993335f, 1.0f),
				make_float4(0.576768f, 0.884712f, 0.993335f, 1.0f),
				make_float4(0.96f, 0.96f, 0.96f, 1.0f),
				1.0f, 0.359385f, 1.0f, 0
			} },
			Sphere{ make_float3(-3.481116f, 0.200000f, -3.794556f), 0.200000f, Material{
				make_float4(0.405104f, 0.066436f, 0.009339f, 1.0f),
				make_float4(0.0f, 0.0f, 0.0f, 1.0f),
				make_float4(0.96f, 0.96f, 0.96f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(-3.866858f, 0.200000f, -1.465965f), 0.200000f, Material{
				make_float4(0.027570f, 0.021652f, 0.252798f, 1.0f),
				make_float4(0.0f, 0.0f, 0.0f, 1.0f),
				make_float4(0.96f, 0.96f, 0.96f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(-3.168870f, 0.200000f, 0.553099f), 0.200000f, Material{
				make_float4(0.421992f, 0.107577f, 0.177504f, 1.0f),
				make_float4(0.0f, 0.0f, 0.0f, 1.0f),
				make_float4(0.96f, 0.96f, 0.96f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(-3.428552f, 0.200000f, 2.627547f), 0.200000f, Material{
				make_float4(0.974029f, 0.653443f, 0.571877f, 1.0f),
				make_float4(0.974029f, 0.653443f, 0.571877f, 1.0f),
				make_float4(0.96f, 0.96f, 0.96f, 1.0f),
				1.0f, 0.312780f, 1.0f, 0
			} },
			Sphere{ make_float3(-3.771736f, 0.200000f, 4.324785f), 0.200000f, Material{
				make_float4(0.685957f, 0.000043f, 0.181270f, 1.0f),
				make_float4(0.0f, 0.0f, 0.0f, 1.0f),
				make_float4(0.96f, 0.96f, 0.96f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(-3.768522f, 0.200000f, 6.384588f), 0.200000f, Material{
				make_float4(0.025972f, 0.082246f, 0.138765f, 1.0f),
				make_float4(0.0f, 0.0f, 0.0f, 1.0f),
				make_float4(0.96f, 0.96f, 0.96f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(-3.286992f, 0.200000f, 8.441148f), 0.200000f, Material{
				make_float4(0.186577f, 0.560376f, 0.367045f, 1.0f),
				make_float4(0.0f, 0.0f, 0.0f, 1.0f),
				make_float4(0.96f, 0.96f, 0.96f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(-1.552127f, 0.200000f, -7.728200f), 0.200000f, Material{
				make_float4(0.202998f, 0.002459f, 0.015350f, 1.0f),
				make_float4(0.0f, 0.0f, 0.0f, 1.0f),
				make_float4(0.96f, 0.96f, 0.96f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(-1.360796f, 0.200000f, -5.346098f), 0.200000f, Material{
				make_float4(0.690820f, 0.028470f, 0.179907f, 1.0f),
				make_float4(0.0f, 0.0f, 0.0f, 1.0f),
				make_float4(0.96f, 0.96f, 0.96f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(-1.287209f, 0.200000f, -3.735321f), 0.200000f, Material{
				make_float4(0.345974f, 0.672353f, 0.450180f, 1.0f),
				make_float4(0.0f, 0.0f, 0.0f, 1.0f),
				make_float4(0.96f, 0.96f, 0.96f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(-1.344859f, 0.200000f, -1.726654f), 0.200000f, Material{
				make_float4(0.209209f, 0.431116f, 0.164732f, 1.0f),
				make_float4(0.0f, 0.0f, 0.0f, 1.0f),
				make_float4(0.96f, 0.96f, 0.96f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(-1.974774f, 0.200000f, 0.183260f), 0.200000f, Material{
				make_float4(0.006736f, 0.675637f, 0.622067f, 1.0f),
				make_float4(0.0f, 0.0f, 0.0f, 1.0f),
				make_float4(0.96f, 0.96f, 0.96f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(-1.542872f, 0.200000f, 2.067868f), 0.200000f, Material{
				make_float4(0.192247f, 0.016661f, 0.010109f, 1.0f),
				make_float4(0.0f, 0.0f, 0.0f, 1.0f),
				make_float4(0.96f, 0.96f, 0.96f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(-1.743856f, 0.200000f, 4.752810f), 0.200000f, Material{
				make_float4(0.295270f, 0.108339f, 0.276513f, 1.0f),
				make_float4(0.0f, 0.0f, 0.0f, 1.0f),
				make_float4(0.96f, 0.96f, 0.96f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(-1.955621f, 0.200000f, 6.493702f), 0.200000f, Material{
				make_float4(0.270527f, 0.270494f, 0.202029f, 1.0f),
				make_float4(0.0f, 0.0f, 0.0f, 1.0f),
				make_float4(0.96f, 0.96f, 0.96f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(-1.350449f, 0.200000f, 8.068503f), 0.200000f, Material{
				make_float4(0.646942f, 0.501660f, 0.573693f, 1.0f),
				make_float4(0.0f, 0.0f, 0.0f, 1.0f),
				make_float4(0.96f, 0.96f, 0.96f, 1.0f),
				1.0f, 0.346551f, 1.0f, 0
			} },
			Sphere{ make_float3(0.706123f, 0.200000f, -7.116040f), 0.200000f, Material{
				make_float4(0.027695f, 0.029917f, 0.235781f, 1.0f),
				make_float4(0.0f, 0.0f, 0.0f, 1.0f),
				make_float4(0.96f, 0.96f, 0.96f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(0.897766f, 0.200000f, -5.938681f), 0.200000f, Material{
				make_float4(0.114934f, 0.046258f, 0.039647f, 1.0f),
				make_float4(0.0f, 0.0f, 0.0f, 1.0f),
				make_float4(0.96f, 0.96f, 0.96f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(0.744113f, 0.200000f, -3.402960f), 0.200000f, Material{
				make_float4(0.513631f, 0.335578f, 0.204787f, 1.0f),
				make_float4(0.0f, 0.0f, 0.0f, 1.0f),
				make_float4(0.96f, 0.96f, 0.96f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(0.867750f, 0.200000f, -1.311908f), 0.200000f, Material{
				make_float4(0.400246f, 0.000956f, 0.040513f, 1.0f),
				make_float4(0.0f, 0.0f, 0.0f, 1.0f),
				make_float4(0.96f, 0.96f, 0.96f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(0.082480f, 0.200000f, 0.838206f), 0.200000f, Material{
				make_float4(0.594141f, 0.215068f, 0.025718f, 1.0f),
				make_float4(0.0f, 0.0f, 0.0f, 1.0f),
				make_float4(0.96f, 0.96f, 0.96f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(0.649692f, 0.200000f, 2.525103f), 0.200000f, Material{
				make_float4(0.602157f, 0.797249f, 0.614694f, 1.0f),
				make_float4(0.96f, 0.96f, 0.96f, 1.0f),
				make_float4(0.96f, 0.96f, 0.96f, 1.0f),
				1.0f, 0.341860f, 1.0f, 0
			} },
			Sphere{ make_float3(0.378574f, 0.200000f, 4.055579f), 0.200000f, Material{
				make_float4(0.005086f, 0.003349f, 0.064403f, 1.0f),
				make_float4(0.0f, 0.0f, 0.0f, 1.0f),
				make_float4(0.96f, 0.96f, 0.96f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(0.425844f, 0.200000f, 6.098526f), 0.200000f, Material{
				make_float4(0.266812f, 0.016602f, 0.000853f, 1.0f),
				make_float4(0.0f, 0.0f, 0.0f, 1.0f),
				make_float4(0.96f, 0.96f, 0.96f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(0.261365f, 0.200000f, 8.661150f), 0.200000f, Material{
				make_float4(0.150201f, 0.007353f, 0.152506f, 1.0f),
				make_float4(0.0f, 0.0f, 0.0f, 1.0f),
				make_float4(0.96f, 0.96f, 0.96f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(2.814218f, 0.200000f, -7.751227f), 0.200000f, Material{
				make_float4(0.570094f, 0.610319f, 0.584192f, 1.0f),
				make_float4(0.570094f, 0.610319f, 0.584192f, 1.0f),
				make_float4(0.96f, 0.96f, 0.96f, 1.0f),
				1.0f, 0.018611f, 1.0f, 0
			} },
			Sphere{ make_float3(2.050073f, 0.200000f, -5.731364f), 0.200000f, Material{
				make_float4(0.109886f, 0.029498f, 0.303265f, 1.0f),
				make_float4(0.0f, 0.0f, 0.0f, 1.0f),
				make_float4(0.96f, 0.96f, 0.96f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(2.020130f, 0.200000f, -3.472627f), 0.200000f, Material{
				make_float4(0.216908f, 0.216448f, 0.221775f, 1.0f),
				make_float4(0.0f, 0.0f, 0.0f, 1.0f),
				make_float4(0.96f, 0.96f, 0.96f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(2.884277f, 0.200000f, -1.232662f), 0.200000f, Material{
				make_float4(0.483428f, 0.027275f, 0.113898f, 1.0f),
				make_float4(0.0f, 0.0f, 0.0f, 1.0f),
				make_float4(0.96f, 0.96f, 0.96f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(2.644454f, 0.200000f, 0.596324f), 0.200000f, Material{
				make_float4(0.005872f, 0.860718f, 0.561933f, 1.0f),
				make_float4(0.0f, 0.0f, 0.0f, 1.0f),
				make_float4(0.96f, 0.96f, 0.96f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(2.194283f, 0.200000f, 2.880603f), 0.200000f, Material{
				make_float4(0.452710f, 0.824152f, 0.045179f, 1.0f),
				make_float4(0.0f, 0.0f, 0.0f, 1.0f),
				make_float4(0.96f, 0.96f, 0.96f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(2.281000f, 0.200000f, 4.094307f), 0.200000f, Material{
				make_float4(0.002091f, 0.145849f, 0.032535f, 1.0f),
				make_float4(0.0f, 0.0f, 0.0f, 1.0f),
				make_float4(0.96f, 0.96f, 0.96f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(2.080841f, 0.200000f, 6.716384f), 0.200000f, Material{
				make_float4(0.468539f, 0.032772f, 0.018071f, 1.0f),
				make_float4(0.0f, 0.0f, 0.0f, 1.0f),
				make_float4(0.96f, 0.96f, 0.96f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(2.287131f, 0.200000f, 8.583242f), 0.200000f, Material{
				make_float4(0.000000f, 0.000000f, 0.000000f, 1.0f),
				make_float4(0.0f, 0.0f, 0.0f, 1.0f),
				make_float4(0.96f, 0.96f, 0.96f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(4.329136f, 0.200000f, -7.497218f), 0.200000f, Material{
				make_float4(0.030865f, 0.071452f, 0.016051f, 1.0f),
				make_float4(0.0f, 0.0f, 0.0f, 1.0f),
				make_float4(0.96f, 0.96f, 0.96f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(4.502115f, 0.200000f, -5.941060f), 0.200000f, Material{
				make_float4(0.000000f, 0.000000f, 0.000000f, 1.0f),
				make_float4(0.0f, 0.0f, 0.0f, 1.0f),
				make_float4(0.96f, 0.96f, 0.96f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(4.750631f, 0.200000f, -3.836759f), 0.200000f, Material{
				make_float4(0.702578f, 0.084798f, 0.141374f, 1.0f),
				make_float4(0.0f, 0.0f, 0.0f, 1.0f),
				make_float4(0.96f, 0.96f, 0.96f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(4.277152f, 0.200000f, 4.297482f), 0.200000f, Material{
				make_float4(0.422693f, 0.011222f, 0.211945f, 1.0f),
				make_float4(0.0f, 0.0f, 0.0f, 1.0f),
				make_float4(0.96f, 0.96f, 0.96f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(4.012743f, 0.200000f, 6.225072f), 0.200000f, Material{
				make_float4(0.986275f, 0.073358f, 0.133628f, 1.0f),
				make_float4(0.0f, 0.0f, 0.0f, 1.0f),
				make_float4(0.96f, 0.96f, 0.96f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(4.047066f, 0.200000f, 8.419360f), 0.200000f, Material{
				make_float4(0.878749f, 0.677170f, 0.684995f, 1.0f),
				make_float4(0.96f, 0.96f, 0.96f, 1.0f),
				make_float4(0.96f, 0.96f, 0.96f, 1.0f),
				1.0f, 0.243932f, 1.0f, 0
			} },
			Sphere{ make_float3(6.441846f, 0.200000f, -7.700798f), 0.200000f, Material{
				make_float4(0.309255f, 0.342524f, 0.489512f, 1.0f),
				make_float4(0.0f, 0.0f, 0.0f, 1.0f),
				make_float4(0.96f, 0.96f, 0.96f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(6.430776f, 0.200000f, -1.332107f), 0.200000f, Material{
				make_float4(0.641951f, 0.661402f, 0.326114f, 1.0f),
				make_float4(0.0f, 0.0f, 0.0f, 1.0f),
				make_float4(0.96f, 0.96f, 0.96f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(6.476387f, 0.200000f, 0.329973f), 0.200000f, Material{
				make_float4(0.033000f, 0.648388f, 0.166911f, 1.0f),
				make_float4(0.0f, 0.0f, 0.0f, 1.0f),
				make_float4(0.96f, 0.96f, 0.96f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(6.011877f, 0.200000f, 6.569579f), 0.200000f, Material{
				make_float4(0.044868f, 0.651697f, 0.086779f, 1.0f),
				make_float4(0.0f, 0.0f, 0.0f, 1.0f),
				make_float4(0.96f, 0.96f, 0.96f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(6.096087f, 0.200000f, 8.892333f), 0.200000f, Material{
				make_float4(0.588587f, 0.078723f, 0.044928f, 1.0f),
				make_float4(0.0f, 0.0f, 0.0f, 1.0f),
				make_float4(0.96f, 0.96f, 0.96f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(8.185763f, 0.200000f, -7.191109f), 0.200000f, Material{
				make_float4(0.989702f, 0.886784f, 0.540759f, 1.0f),
				make_float4(0.96f, 0.96f, 0.96f, 1.0f),
				make_float4(0.96f, 0.96f, 0.96f, 1.0f),
				1.0f, 0.104229f, 1.0f, 0
			} },
			Sphere{ make_float3(8.411960f, 0.200000f, -5.285309f), 0.200000f, Material{
				make_float4(0.139604f, 0.022029f, 0.461688f, 1.0f),
				make_float4(0.0f, 0.0f, 0.0f, 1.0f),
				make_float4(0.96f, 0.96f, 0.96f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(8.047109f, 0.200000f, -3.427552f), 0.200000f, Material{
				make_float4(0.815002f, 0.631228f, 0.806757f, 1.0f),
				make_float4(0.815002f, 0.631228f, 0.806757f, 1.0f),
				make_float4(0.96f, 0.96f, 0.96f, 1.0f),
				1.0f, 0.150782f, 1.0f, 0
			} },
			Sphere{ make_float3(8.119639f, 0.200000f, -1.652587f), 0.200000f, Material{
				make_float4(0.177852f, 0.429797f, 0.042251f, 1.0f),
				make_float4(0.0f, 0.0f, 0.0f, 1.0f),
				make_float4(0.96f, 0.96f, 0.96f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(8.256561f, 0.200000f, 8.129115f), 0.200000f, Material{
				make_float4(0.002612f, 0.598319f, 0.435378f, 1.0f),
				make_float4(0.0f, 0.0f, 0.0f, 1.0f),
				make_float4(0.96f, 0.96f, 0.96f, 1.0f),
				0.0f, 1.000000f, 0.0f, 0
			} },
			Sphere{ make_float3(0.000000f, 1.000000f, 0.000000f), 1.000000f, Material{
				make_float4(0.000000f, 0.000000f, 0.000000f, 1.0f),
				make_float4(0.96f, 0.96f, 0.96f, 1.0f),
				make_float4(0.96f, 0.96f, 0.96f, 1.0f),
				0.0f, 0.114f, 0.85f, 0
			} },
			Sphere{ make_float3(-4.000000f, 1.000000f, 0.000000f), 1.000000f, Material{
				make_float4(0.400000f, 0.200000f, 0.100000f, 1.0f),
				make_float4(0.96f, 0.96f, 0.96f, 1.0f),
				make_float4(0.96f, 0.96f, 0.96f, 1.0f),
				0.0f, 0.2f, 0.95f, 0
			} },
			Sphere{ make_float3(4.000000f, 1.000000f, 0.000000f), 1.000000f, Material{
				make_float4(0.700000f, 0.600000f, 0.500000f, 1.0f),
				make_float4(0.96f, 0.96f, 0.96f, 1.0f),
				make_float4(0.96f, 0.96f, 0.96f, 1.0f),
				0.0f, 0.0f, 0.99f, 0 // TODO: 머터리얼 조정
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
			float theta = 2.0f * PI * cumath::Rand(state);
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
				float dist = (-b - sqrt(discriminant)) / (2.0f * a);

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

					ray.Origin = hitinfo.HitPoint + hitinfo.Normal * 0.01f;
					float3 diffuseDir = cumath::Normalize(hitinfo.Normal + randDirection(state));
					float3 specularDir = cumath::Reflect(ray.Dir, hitinfo.Normal);
					ray.Dir = cumath::Normalize(cumath::Lerp(diffuseDir, specularDir, (1.0f - material.Roughness) * bSpecularBounce));

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

			const FVector3 eye = { 13.0f, 2.0f, 4.0f };
			const FVector3 center = { 0.0f, 0.0f, 0.0f };
			const FVector3 up = { 0.0f, 1.0f, 0.0f };

			// Rotation
			static float time = 0.0f; // 누적 시간 (프레임당 증가)
			time += 0.02f; // 프레임당 회전 속도 조절
			float radius = sqrtf(eye.x * eye.x + eye.z * eye.z);
			float angle = time;

			FVector3 rotEye = {
				center.x + radius * cosf(angle),
				eye.y,
				center.z + radius * sinf(angle)
			};


			float4x4 cameraMatrix;
			FMatrix4x4 cameraMatrixFMat = Mat4x4Camera(rotEye, center, up);
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