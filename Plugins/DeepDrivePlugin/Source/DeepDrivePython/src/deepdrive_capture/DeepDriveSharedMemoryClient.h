
#pragma once

#include "Engine.h"

class SharedMemory;
struct PyCaptureCameraObject;
struct PyCaptureLastCollisionObject;
struct PyCaptureSnapshotObject;

struct DeepDriveCaptureCamera;
struct DeepDriveLastCollisionData;
struct DeepDriveCaptureMessage;

class DeepDriveSharedMemoryClient
{
public:

	DeepDriveSharedMemoryClient();
	~DeepDriveSharedMemoryClient();

	bool connect(const std::string &name, uint32 maxSize);
	
	PyCaptureSnapshotObject* readMessage();

	void close();

	bool isConnected() const;

private:

	PyCaptureCameraObject* buildCamera(const DeepDriveCaptureCamera &srcCam);
	void setupLastCollision(const DeepDriveLastCollisionData &srcCollision, PyCaptureLastCollisionObject &dstCollision);

	void dumpSharedMemContent(const DeepDriveCaptureMessage *data);

	SharedMemory			*m_SharedMemory = 0;
	bool					m_isConnected = false;

	uint32					m_lastMsgId = 0;

	uint32					m_maxSize = 0;

	uint32					m_DumpIndex = 0;
};


inline bool DeepDriveSharedMemoryClient::isConnected() const
{
	return m_isConnected;
}