#pragma once
#include <string>

std::string extractName(const std::string& path)
{
	size_t lastSlash = path.find_last_of("/\\");
	std::string filename = (lastSlash == std::string::npos) ? path : path.substr(lastSlash + 1);
	size_t lastDot = filename.find_last_of('.');
	std::string name = (lastDot == std::string::npos) ? filename : filename.substr(0, lastDot);
	return name;
}