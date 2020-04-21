
#pragma once

#include "EngineMinimal.h"


DECLARE_LOG_CATEGORY_EXTERN(LogReplayActorEventLog, Log, All);

class AActorEventLogReplayManager;

class ReplayActorEventLog
{
public:

	ReplayActorEventLog(AActorEventLogReplayManager &replayMgr, int32 id);

	void updateOnIdle(float deltaSeconds);

	bool load(const FString &fileName);

	void setActor(AActor *actor);
	
	void setPhysicsEnabled(bool enablePhysics, bool enableGravity);

	void replayFrame(int32 frameIndex);

	const FString& getFileName() const;
	const FString& getActorName() const;
	const FString& getActorDescription() const;
	const FString& getActorClassName() const;

	const FTransform& getInitialTransform() const;

	int32 getFrameCount() const;

private:

	void handleEvent(const TSharedPtr<FJsonObject> &event);

	void handleActorTransform(const TSharedPtr<FJsonObject> &event);
	void handleMessage(const TSharedPtr<FJsonObject> &event);

	FTransform extractTransform(const TSharedPtr<FJsonObject> &jsonObj, const FString &fieldName) const;

	AActorEventLogReplayManager		&m_ReplayManager;
	int32							m_Id;

	TSharedPtr<FJsonObject>			m_JsonObject;

	FString							m_FileName;
	FString							m_ActorName;
	FString							m_ActorDescription;
	FString							m_ActorClassName;

	AActor							*m_Actor = 0;

	FTransform						m_curTransform;
	bool 							m_KeepTransformFixed = true;

};

inline void ReplayActorEventLog::setActor(AActor *actor)
{
	m_Actor = actor;
}

inline const FString& ReplayActorEventLog::getFileName() const
{
	return m_FileName;
}

inline const FString& ReplayActorEventLog::getActorName() const
{
	return m_ActorName;
}

inline const FString& ReplayActorEventLog::getActorDescription() const
{
	return m_ActorDescription;
}

inline const FString& ReplayActorEventLog::getActorClassName() const
{
	return m_ActorClassName;
}

