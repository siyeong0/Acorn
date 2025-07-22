#pragma once
#include "Mesh/Mesh.h"
#include "Voxel/Avox.h"

namespace aco
{
	class Voxelizer
	{
	public:
		// Voxeliz a mesh into a avox(sparse voxel octree).
		Avox BuildAvox(const Mesh& mesh);
	};
}