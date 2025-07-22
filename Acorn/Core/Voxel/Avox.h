#pragma once
#include <cstdint>
#include <cassert>
#include <vector>
#include <string>

#include "AvoxNode.h"

namespace aco
{
	struct Avox
	{
		std::vector<AvoxNode> Nodes;
		std::vector<uint32_t> Offsets;

		Avox() = default;
		~Avox() = default;

		const AvoxNode& GetRoot() const
		{
			assert(!Nodes.empty() && "Avox has no nodes.");
			return Nodes[0];
		}

		AvoxNode& operator[](size_t index)
		{
			assert(index < Nodes.size() && "Index out of bounds.");
			return Nodes[index];
		}

		void Save(const std::string& filename) const;
	};
}