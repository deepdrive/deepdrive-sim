
#pragma once

#include "Engine.h"

struct IP4Address
{
	IP4Address();

	bool set(const char *addressStr, uint16 _port);

	std::string toStr(bool appendPort) const;

	uint8		address[4];
	uint16		port;
};