#include <iostream>

#include "Core/Mesh/Mesh.h"
#include "Core/Voxel/Avox.h"
#include "Core/Voxelizer.h"

#include "Utils/Path.h"

int main(void)
{
	std::cout << "Acorn is running!" << std::endl;

	std::string filepath = "../Resources/Mesh/Boguchi.glb";
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
	avox.Save(savepath);

	return 0;
}