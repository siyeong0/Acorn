#pragma once
#include <vector>
#include <string>

#include "Math/Math.h"
#include "Texture.h"

namespace aco
{
	struct Mesh
	{
		std::vector<FVector3> Vertices;
		std::vector<uint32_t> Indices;
		Bounds Bounds;

		std::vector<FVector3> Normals;
		std::vector<FVector4> Colors;
		std::vector<FVector2> UVs;
		Texture* Tex = nullptr;

		enum class EColorSourceType
		{
			None = 0,
			VertexColor = 1,
			TextureUV = 2,
		};
		EColorSourceType ColorSourceType = EColorSourceType::None;

		std::string Name;

		Mesh() = default;
		Mesh(const std::string& path);
		~Mesh() = default;
		Mesh(const Mesh& other);
		Mesh(Mesh&& other) noexcept;
		Mesh& operator=(const Mesh& other);

		inline int NumVertices() const { return static_cast<int>(Vertices.size()); }
		inline int NumTriangles() const { return static_cast<int>(Indices.size() / 3); }

		bool Load(const std::string& path, float scale = 1.0f);
		void ComputeNormals();
		void ComputeBounds();
	};
}