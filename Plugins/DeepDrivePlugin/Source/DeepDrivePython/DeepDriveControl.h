
#pragma once

#include "Engine.h"

class SharedMemory;

struct PyDeepDriveControlObject;

class DeepDriveControl
{
public:

	DeepDriveControl();
	~DeepDriveControl();

	bool create(const std::string &name, uint32 maxSize);

	bool isCreated() const;
	
	void close();

	void sendControl(const PyDeepDriveControlObject &control);

	void disconnect();

private:

	SharedMemory			*m_SharedMemory = 0;
	bool					m_isCreated = false;

	uint32					m_nextMsgId = 1;
};


inline bool DeepDriveControl::isCreated() const
{
	return m_isCreated;
}