
#include "DeepDriveControl.h"
#include "Public/SharedMemory/SharedMemory.h"

#include "Public/Messages/DeepDriveControlMessages.h"

#include "PyDeepDriveControlObject.h"

#include <iostream>

DeepDriveControl::DeepDriveControl()
	:	m_SharedMemory(new SharedMemory)
{
}

DeepDriveControl::~DeepDriveControl()
{
	delete m_SharedMemory;
}

void DeepDriveControl::disconnect()
{
	if(m_isCreated)
	{
		DeepDriveDisconnectControl *disconnectMsg = reinterpret_cast<DeepDriveDisconnectControl*> (m_SharedMemory->lockForWriting(-1));

		if(disconnectMsg)
		{
			disconnectMsg = new (disconnectMsg) DeepDriveDisconnectControl();

			disconnectMsg->setMessageId();
			m_SharedMemory->unlock(disconnectMsg->message_size);
			std::cout << "ControlDisconnect message sent\n";
		}
	}
}

bool DeepDriveControl::create(const std::string &name, uint32 maxSize)
{
	if(m_SharedMemory)
	{
		m_isCreated = m_SharedMemory->create(FString(name), maxSize);
		if(m_isCreated)
			std::cout << "Successfully created " << name << " with max size of " << maxSize << "\n";
		else
			std::cout << "Failed to open " << name << "\n";			
	}

	return m_isCreated;
}

void DeepDriveControl::close()
{
	
}

void DeepDriveControl::sendControl(const PyDeepDriveControlObject &control)
{
	if(m_isCreated)
	{
		DeepDriveControlMessage *ctrlMsg = reinterpret_cast<DeepDriveControlMessage*> (m_SharedMemory->lockForWriting(-1));

		if(ctrlMsg)
		{
			ctrlMsg = new (ctrlMsg) DeepDriveControlMessage();

			ctrlMsg->capture_timestamp = control.capture_timestamp;
			ctrlMsg->capture_sequence_number = control.capture_sequence_number;
			ctrlMsg->steering = control.steering;
			ctrlMsg->throttle = control.throttle;
			ctrlMsg->brake = control.brake;
			ctrlMsg->handbrake = control.handbrake;
			ctrlMsg->is_game_driving = control.is_game_driving;
			ctrlMsg->should_reset = control.should_reset;
//			std::cout << "Should reset: " << control.should_reset << "\n";
//			std::cout << "Is game driving: " << control.is_game_driving << "\n";
			ctrlMsg->setMessageId();
			m_SharedMemory->unlock(ctrlMsg->message_size);
		}
	}
}
