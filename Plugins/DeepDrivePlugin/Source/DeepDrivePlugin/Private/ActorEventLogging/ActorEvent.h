
#pragma once

#include "EngineMinimal.h"


struct ActorEvent
{
	uint32						EventType;
	double						Timestamp;

	TSharedPtr<FJsonObject>		EventData;
};
