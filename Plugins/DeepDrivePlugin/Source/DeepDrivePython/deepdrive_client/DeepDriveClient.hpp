
#pragma once

#include "Engine.h"

#include "socket/IP4Address.hpp"
#include "socket/IP4ClientSocket.hpp"

class DeepDriveClient
{

public:

	DeepDriveClient(const IP4Address &ip4Address);

	~DeepDriveClient();

	uint32 registerClient();

	void close();

	bool isConnected() const;

	uint32 registerCamera(float hFoV, uint16 captureWidth, uint16 captureHeight, float relPos[3], float relRot[3]);

	bool requestAgentControl();
	void releaseAgentControl();

	const char* getSharedMemoryName() const;
	uint32 getSharedMemorySize() const;

	void setControlValues(float steering, float throttle, float brake, uint32 handbrake);

private:

	IP4ClientSocket					m_Socket;

	uint32							m_ClientId = 0;
	bool							m_isMaster = false;

	uint32							m_ServerProtocolVersion = 0;

	std::string						m_SharedMemoryName;
	uint32							m_SharedMemorySize = 0;

	uint16							m_MaxSupportedCameras = 0;
	uint16							m_MaxCaptureResolution = 0;

	uint32							m_InactivityTimeout = 0;

};


inline const char* DeepDriveClient::getSharedMemoryName() const
{
	return m_SharedMemoryName.c_str();
}

inline uint32 DeepDriveClient::getSharedMemorySize() const
{
	return m_SharedMemorySize;
}
