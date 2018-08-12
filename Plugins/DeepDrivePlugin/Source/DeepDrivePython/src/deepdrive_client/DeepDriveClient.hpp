
#pragma once

#include "Engine.h"

#include "socket/IP4Address.hpp"
#include "socket/IP4ClientSocket.hpp"

#include "Public/Server/Messages/DeepDriveServerConnectionMessages.h"

class DeepDriveClient
{

public:

	DeepDriveClient(const IP4Address &ip4Address);

	~DeepDriveClient();

	int32 registerClient(deepdrive::server::RegisterClientResponse &response, bool requestMasterRole);

	void close();

	bool isConnected() const;

	int32 registerCamera(float hFoV, uint16 captureWidth, uint16 captureHeight, float relPos[3], float relRot[3], const char *label);

	int32 requestAgentControl();
	int32 releaseAgentControl();
	int32 resetAgent();

	int32 setControlValues(float steering, float throttle, float brake, uint32 handbrake);
	
	int32 activateSynchronousStepping();
	int32 deactivateSynchronousStepping();

	int32 advanceSynchronousStepping(float dT, float steering, float throttle, float brake, uint32 handbrake);

	int32 setViewMode(int32 cameraId, const char *viewMode);

	const char* getSharedMemoryName() const;
	uint32 getSharedMemorySize() const;

	uint32 getClientId() const;

	IP4ClientSocket& getSocket();

	uint32							m_ClientId = 0;
	bool							m_isMaster = false;

	std::string						m_ServerProtocolVersion;

	std::string						m_SharedMemoryName;
	uint32							m_SharedMemorySize = 0;

	uint16							m_MaxSupportedCameras = 0;
	uint16							m_MaxCaptureResolution = 0;

	uint32							m_InactivityTimeout = 0;

private:

	IP4ClientSocket					m_Socket;

};


inline const char* DeepDriveClient::getSharedMemoryName() const
{
	return m_SharedMemoryName.c_str();
}

inline uint32 DeepDriveClient::getSharedMemorySize() const
{
	return m_SharedMemorySize;
}

inline uint32 DeepDriveClient::getClientId() const
{
	return m_ClientId;
}

inline IP4ClientSocket& DeepDriveClient::getSocket()
{
	return m_Socket;
}
