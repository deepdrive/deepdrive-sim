
#include "socket/IP4Address.hpp"

#include <string>
#include <sstream>
#include <vector>
#include <iterator>

#include <iostream>

template<typename Out>
void split(const std::string &s, char delim, Out result)
{
	std::stringstream ss(s);
	std::string item;
	while (std::getline(ss, item, delim))
	{
		*(result++) = item;
	}
}

std::vector<std::string> split(const std::string &s, char delim)
{
	std::vector<std::string> elems;
	split(s, delim, std::back_inserter(elems));
	return elems;
}

IP4Address::IP4Address()
{

}

bool IP4Address::set(const char *addressStr, uint16 _port)
{
	bool res = false;
	std::vector<std::string> partsStr = split(addressStr, '.');
	if(partsStr.size() == 4)
	{
		int32 parts[4] = { std::stoi(partsStr[0]), std::stoi(partsStr[1]), std::stoi(partsStr[2]), std::stoi(partsStr[3]) };

		res = parts[0] > 0;
		for(unsigned i = 0; res && i < 4; ++i)
		{
			if(parts[i] >= 0 && parts[i] < 256)
				address[i] = static_cast<uint8> (parts[i]);
			else
				res = false;
		}
		port = _port;
	}
	return res;
}

std::string IP4Address::toStr(bool appendPort) const
{
	std::ostringstream ss;

	ss << static_cast<int32> (address[0]) << "." << static_cast<int32> (address[1]) << "." << static_cast<int32> (address[2]) << "." << static_cast<int32> (address[3]);
	if(appendPort)
		ss << ":" << port;
	return ss.str();
}
