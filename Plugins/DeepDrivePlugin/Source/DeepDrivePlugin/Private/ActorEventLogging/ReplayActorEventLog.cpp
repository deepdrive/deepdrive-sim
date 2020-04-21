
#include "ReplayActorEventLog.h"
#include "ActorEventLogReplayManager.h"

DEFINE_LOG_CATEGORY(LogReplayActorEventLog);

ReplayActorEventLog::ReplayActorEventLog(AActorEventLogReplayManager &replayMgr, int32 id)
	:	m_ReplayManager(replayMgr)
	,	m_Id(id)
{

}

bool ReplayActorEventLog::load(const FString &fileName)
{
	bool res = false;

	FString jsonStr;
	if(FFileHelper::LoadFileToString(jsonStr, *fileName))
	{
		UE_LOG(LogReplayActorEventLog, Log, TEXT("Successfully loaded actor event logfile %s"), *(fileName));

		TSharedRef< TJsonReader<> > reader = TJsonReaderFactory<>::Create( jsonStr );
		if ( FJsonSerializer::Deserialize(reader, m_JsonObject ) )
		{
			UE_LOG(LogReplayActorEventLog, Log, TEXT("Successfully deserialized log file"));

			FString nameStr;
			
			if	(	m_JsonObject->TryGetStringField("Name", m_ActorName)
				&&	m_JsonObject->HasTypedField<EJson::Array>("Frames")
				)
			{
				m_JsonObject->TryGetStringField("Description", m_ActorDescription);
				m_JsonObject->TryGetStringField("Class", m_ActorClassName);
				m_JsonObject->TryGetBoolField("KeepTransformFixed", m_KeepTransformFixed);
				m_curTransform = extractTransform(m_JsonObject, "InitialTransform");

				m_FileName = fileName;

				res = true;
			}

		}
		else
			UE_LOG(LogReplayActorEventLog, Log, TEXT("Failed to deserialized log file"));
	}

	return res;
}

void ReplayActorEventLog::updateOnIdle(float deltaSeconds)
{
	if(m_KeepTransformFixed)
	{
		m_Actor->SetActorTransform(m_curTransform, false, 0, ETeleportType::TeleportPhysics);
	}
}

void ReplayActorEventLog::replayFrame(int32 frameIndex)
{
	const TArray< TSharedPtr< FJsonValue > > &frames = m_JsonObject->GetArrayField("Frames");
	if(frameIndex >= frames.Num())
		frameIndex = frames.Num() - 1;

	if(frameIndex >= 0)
	{
		TSharedPtr<FJsonValue> curFrame = frames[frameIndex];
		const TSharedPtr<FJsonObject> *frameObj = 0;
		if	(	curFrame->TryGetObject(frameObj)
			&&	frameObj
			&&	(*frameObj).IsValid()
			)
		{
			const TArray< TSharedPtr< FJsonValue > > *events = 0;
			if	(	(*frameObj)->TryGetArrayField("Events", events)
				&&	events
				&&	events->Num() > 0
				)
			{
				for(auto &event : (*events))
				{
					const TSharedPtr<FJsonObject> *eventObj = 0;
					if	(	event->TryGetObject(eventObj)
						&&	eventObj
						&&	(*eventObj).IsValid()
						)
					{
						handleEvent(*eventObj);
					}
				}
			}
		}
	}

}


void ReplayActorEventLog::handleEvent(const TSharedPtr<FJsonObject> &event)
{
	FString eventType;
	if(event->TryGetStringField("Type", eventType))
	{
		if(eventType == "ActorTransform")
		{
			handleActorTransform(event);
		}
		else if(eventType == "Message")
		{
			handleMessage(event);
		}
	}
}

void ReplayActorEventLog::handleMessage(const TSharedPtr<FJsonObject> &event)
{
	FString message;
	if(event->TryGetStringField("Data", message))
	{
		if (m_ReplayManager.OnMessageEvent.IsBound())
		{
			m_ReplayManager.OnMessageEvent.Broadcast(m_Id, message);
		}
		
	}
}


void ReplayActorEventLog::handleActorTransform(const TSharedPtr<FJsonObject> &event)
{
	m_curTransform = extractTransform(event, "Data");
	m_Actor->SetActorTransform(m_curTransform, false, 0, ETeleportType::TeleportPhysics);
}

const FTransform& ReplayActorEventLog::getInitialTransform() const
{
	return m_curTransform;
}

int32 ReplayActorEventLog::getFrameCount() const
{
	const TArray< TSharedPtr< FJsonValue > > &frames = m_JsonObject->GetArrayField("Frames");
	return frames.Num();
}

void ReplayActorEventLog::setPhysicsEnabled(bool enablePhysics, bool enableGravity)
{
	TArray<UActorComponent*> components;
	m_Actor->GetComponents(components, false);
	for (auto &c : components)
	{
		UPrimitiveComponent *comp = Cast<UPrimitiveComponent>(c);
		if (comp)
		{
			comp->SetSimulatePhysics(enablePhysics);
			comp->SetEnableGravity(enableGravity);
		}
	}
}

FTransform ReplayActorEventLog::extractTransform(const TSharedPtr<FJsonObject> &jsonObj, const FString &fieldName) const
{
	FVector location(0.0f, 0.0f, 0.0f);
	FRotator rotation(0.0f, 0.0f, 0.0f);
	FVector scale(1.0f, 1.0f, 1.0f);

	const TArray< TSharedPtr< FJsonValue > > *initialTransform;
	if (jsonObj->TryGetArrayField(fieldName, initialTransform))
	{
		location = FVector((*initialTransform)[0]->AsNumber(), (*initialTransform)[1]->AsNumber(), (*initialTransform)[2]->AsNumber());
		rotation = FRotator((*initialTransform)[3]->AsNumber(), (*initialTransform)[4]->AsNumber(), (*initialTransform)[5]->AsNumber());
	}

	return FTransform(rotation, location, scale);

}
