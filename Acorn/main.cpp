#include <iostream>

#include "Graphics/Mesh/Mesh.h"
#include "Graphics/Voxel/Avox.h"
#include "Graphics/Voxel/Voxelizer.h"

#include "Graphics/Renderer/Vulkan/VulkanRenderer.h"
#include "Graphics/Cublit/Raytracer.h"

#include "Utils/Path.h"

#include "Debug.h"

int main(void)
{
	std::cout << "Acorn is running!" << std::endl;

	aco::gfx::VulkanRenderer app;
	aco::gfx::Raytracer raytracer;
	app.SetBlit(&raytracer);
	try 
	{
		app.Run();
	}
	catch (const std::exception& e) 
	{
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}

//int main(void)
//{
//	std::string filepath = "../Resources/Mesh/Boguchi.glb";
//	float scale = 30.0f;
//
//	aco::Mesh mesh;
//	if (!mesh.Load(filepath, scale))
//	{
//		std::cerr << "Failed to load mesh!" << std::endl;
//		return -1;
//	}
//
//	auto voxelizer = aco::Voxelizer();
//	aco::Avox avox = voxelizer.BuildAvox(mesh);
//
//	std::string savepath = "../Resources/Avox/" + aco::ExtractName(filepath) + ".avox";
//	avox.Save(savepath);
//}