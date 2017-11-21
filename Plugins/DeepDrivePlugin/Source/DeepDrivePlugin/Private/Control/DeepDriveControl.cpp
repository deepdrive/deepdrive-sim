
#include "DeepDrivePluginPrivatePCH.h"
#include "Private/Control/DeepDriveControl.h"
#include "Private/Control/DeepDriveControlWorker.h"

#include "Public/Control/DeepDriveControlProxy.h"

DEFINE_LOG_CATEGORY(LogDeepDriveControl);


DeepDriveControl* DeepDriveControl::theInstance = 0;


DeepDriveControl& DeepDriveControl::GetInstance()
{
	if (theInstance == 0)
	{
		theInstance = new DeepDriveControl;
	}

	return *theInstance;
}

void DeepDriveControl::Destroy()
{
	delete theInstance;
	theInstance = 0;
}


DeepDriveControl::DeepDriveControl()
{
}

void DeepDriveControl::RegisterProxy(ADeepDriveControlProxy &proxy, const FString &sharedMemName, uint32 sharedMemSize)
{
	m_Proxy = &proxy;

	if (m_Worker)
	{
		delete m_Worker;
		m_Worker = 0;
	}

	m_Worker = new DeepDriveControlWorker(sharedMemName, sharedMemSize);
}

void DeepDriveControl::UnregisterProxy(ADeepDriveControlProxy &proxy)
{
	if (&proxy == m_Proxy)
	{
		m_Proxy = 0;
		if (m_Worker)
		{
			m_Worker->Stop();
			m_Worker = 0;
		}
	}
}

const DeepDriveMessageHeader* DeepDriveControl::getMessage()
{
	return m_Worker ? m_Worker->getMessage() : 0;
}
