#include "Avox.h"

#include <string>
#include <fstream>

namespace aco
{
	void Avox::Save(const std::string& filename) const
	{
		std::ofstream file(filename);
		if (!file.is_open())
		{
			printf("Failed to open file: %s\n", filename.c_str());
			goto END;
		}

		for (size_t i = 0; i < Offsets.back(); ++i)
		{
			const AvoxNode& node = Nodes[i];
			file << "I " << static_cast<int>(node.ChildMask) << " " << static_cast<int>(node.PageHeader) << " " << node.ChildPointer << "\n";
		}

		for (size_t i = Offsets.back(); i < Nodes.size(); ++i)
		{
			const AvoxNode& node = Nodes[i];
			FVector3 color = UnpackRGB565(node.Color);
			file << "L " << int(color.x * 255) << " " << int(color.y * 255) << " " << int(color.z * 255) << "\n";
		}

	END:
		file.close();
	}
}