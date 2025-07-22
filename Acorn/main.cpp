#include <iostream>

#include "Core/Mesh/Mesh.h"
#include "Core/Voxel/Avox.h"
#include "Core/Voxelizer.h"

#include "Graphics\Vulkan\VulkanRenderer.h"

#include "Utils/Path.h"

int main(void)
{
	std::cout << "Acorn is running!" << std::endl;

	aco::gfx::VulkanRenderer app;

	std::string image_filename = "image.jpg";

	try 
	{
		// This app only works on ppm images
		app.LoadImageData(image_filename);
		app.Run();
	}
	catch (const std::exception& e) 
	{
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
	/*std::string filepath = "../Resources/Mesh/Boguchi.glb";
	float scale = 30.0f;

	aco::Mesh mesh;
	if (!mesh.Load(filepath, scale))
	{
		std::cerr << "Failed to load mesh!" << std::endl;
		return -1;
	}

	auto voxelizer = aco::Voxelizer();
	aco::Avox avox = voxelizer.BuildAvox(mesh);

	std::string savepath = "../Resources/Avox/" + extractName(filepath) + ".avox";
	avox.Save(savepath);*/