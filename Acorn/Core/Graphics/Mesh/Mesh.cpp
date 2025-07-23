#include "Mesh.h"

#include <iostream>
#include <filesystem>

#include <assimp/Importer.hpp>
#include <assimp/Exporter.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <stb_image.h>

namespace aco
{
	Mesh::Mesh(const std::string& path)
	{
		std::filesystem::path fsPath(path);
		std::string filename = fsPath.stem().string();
		Name = filename;
	}

	Mesh::Mesh(const Mesh& other)
	{

	}

	Mesh::Mesh(Mesh&& other) noexcept
		: Vertices(other.Vertices)
		, Indices(other.Indices)
	{

	}

	Mesh& Mesh::operator=(const Mesh& other)
	{
		Vertices = other.Vertices;
		Indices = other.Indices;
		return *this;
	}

	bool Mesh::Load(const std::string& path, float scale)
	{
		Assimp::Importer importer;
		const aiScene* scene = importer.ReadFile(path,
			aiProcess_Triangulate |
			aiProcess_DropNormals |
			aiProcess_JoinIdenticalVertices);

		if (!scene || !scene->HasMeshes())
		{
			std::cerr << "Failed to load model \"" << path << "\"\n";
			return false;
		}

		const aiMesh* mesh = scene->mMeshes[0];
		const int numVertices = mesh->mNumVertices;
		const int numFaces = mesh->mNumFaces;
		const int numIndices = numFaces * 3;

		// Vertices.
		Vertices.clear();
		Vertices.reserve(numVertices);
		for (int i = 0; i < numVertices; ++i)
		{
			aiVector3D vertex = mesh->mVertices[i];
			Vertices.emplace_back(vertex.x * scale, vertex.y * scale, vertex.z * scale);
		}

		// Indices.
		Indices.clear();
		Indices.reserve(numIndices);
		for (int i = 0; i < numFaces; ++i)
		{
			aiFace face = mesh->mFaces[i];
			Indices.emplace_back(face.mIndices[0]);
			Indices.emplace_back(face.mIndices[1]);
			Indices.emplace_back(face.mIndices[2]);
		}

		// Normals.
		if (mesh->HasNormals())
		{
			Normals.clear();
			Normals.reserve(numVertices);
			for (int i = 0; i < numVertices; ++i)
			{
				aiVector3D normal = mesh->mNormals[i];
				Normals.emplace_back(normal.x, normal.y, normal.z);
			}
		}
		else
		{
			ComputeNormals();
		}

		// Colors.
		if (mesh->HasVertexColors(0))
		{
			Colors.clear();
			Colors.reserve(numVertices);
			for (int i = 0; i < numVertices; ++i)
			{
				aiColor4D color = mesh->mColors[0][i];
				Colors.emplace_back(color.r, color.g, color.b, color.a);
			}

			ColorSourceType = EColorSourceType::VertexColor;
		}
		// UVs and Textures.
		else if (mesh->HasTextureCoords(0))
		{
			// UVs.
			UVs.clear();
			UVs.reserve(numVertices);
			for (int i = 0; i < numVertices; ++i)
			{
				aiVector3D uv = mesh->mTextureCoords[0][i];
				UVs.emplace_back(uv.x, uv.y);
			}

			// Textures.
			Tex = new Texture();
			const aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];
			aiString aiTexPath;

			if (material->GetTexture(aiTextureType_DIFFUSE, 0, &aiTexPath) == AI_FAILURE)
			{
				std::cerr << "Failed to load texture from material for mesh \"" << Name << "\"\n";
				ColorSourceType = EColorSourceType::None;
				return false;
			}

			std::string texPath = aiTexPath.C_Str();
			unsigned char* pixels = nullptr;
			int width = 0, height = 0, channels = 0;

			if (!texPath.empty() && texPath[0] == '*')
			{
				// Embedded texture
				int texIndex = std::atoi(texPath.c_str() + 1); // "*0" → 0
				const aiTexture* texture = scene->mTextures[texIndex];

				if (texture->mHeight == 0)
				{
					// Compressed image (e.g., PNG, JPEG)
					unsigned char* data = reinterpret_cast<unsigned char*>(texture->pcData);
					pixels = stbi_load_from_memory(data, texture->mWidth, &width, &height, &channels, 4);
					if (!pixels)
					{
						std::cerr << "Failed to load texture from embedded data.\n";
						ColorSourceType = EColorSourceType::None;
						return false;
					}
				}
				else
				{
					// Uncompressed image (RGBA) — optional handling
					width = texture->mWidth;
					height = texture->mHeight;
					channels = 4;
					pixels = new unsigned char[width * height * channels];

					std::memcpy(pixels, texture->pcData, width * height * channels);
				}
			}
			else
			{
				// External texture
				std::string texDir = std::filesystem::path(path).parent_path().string();
				pixels = stbi_load((texDir + "/" + texPath).c_str(), &width, &height, &channels, 4);
				if (!pixels)
				{
					std::cerr << "Failed to load texture image from path: " << texPath << "\n";
					ColorSourceType = EColorSourceType::None;
					return false;
				}
			}

			Tex->Width = width;
			Tex->Height = height;
			Tex->Channels = 4;
			Tex->Data.resize(width * height);

			for (int i = 0; i < width * height; ++i)
			{
				int offset = i * 4;
				float r = pixels[offset + 0] / 255.0f;
				float g = pixels[offset + 1] / 255.0f;
				float b = pixels[offset + 2] / 255.0f;
				float a = pixels[offset + 3] / 255.0f;

				Tex->Data[i] = FVector4(r, g, b, a);
			}

			Tex->FlipY();

			ColorSourceType = EColorSourceType::TextureUV;

			stbi_image_free(pixels); // or delete[] if new[] was used
		}

		// AABB Bounds.
		ComputeBounds();

		return true;

	}

	void Mesh::ComputeNormals()
	{
		Normals.clear();
		Normals.resize(Vertices.size());
		for (int triIdx = 0; triIdx < NumTriangles(); ++triIdx)
		{
			const FVector3& v0 = Vertices[Indices[3 * triIdx + 0]];
			const FVector3& v1 = Vertices[Indices[3 * triIdx + 1]];
			const FVector3& v2 = Vertices[Indices[3 * triIdx + 2]];
			FVector3 normal = (v1 - v0).Cross(v2 - v0).Normalized();
			Normals[Indices[3 * triIdx + 0]] += normal;
			Normals[Indices[3 * triIdx + 1]] += normal;
			Normals[Indices[3 * triIdx + 2]] += normal;
		}

		for (auto& normal : Normals)
		{
			normal.Normalize();
		}
	}

	void Mesh::ComputeBounds()
	{
		Bounds.Min = FVector3::FMaxValue();
		Bounds.Max = FVector3::FMinValue();
		for (int triIdx = 0; triIdx < NumTriangles(); ++triIdx)
		{
			Bounds.Encapsulate(Vertices[Indices[3 * triIdx + 0]]);
			Bounds.Encapsulate(Vertices[Indices[3 * triIdx + 1]]);
			Bounds.Encapsulate(Vertices[Indices[3 * triIdx + 2]]);
		}
	}
}