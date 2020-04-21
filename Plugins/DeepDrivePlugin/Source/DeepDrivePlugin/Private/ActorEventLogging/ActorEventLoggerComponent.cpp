

#include "DeepDrivePluginPrivatePCH.h"
#include "ActorEventLoggerComponent.h"

#include "ActorEventLogManager.h"

DEFINE_LOG_CATEGORY(LogActorEventLogger);

// Sets default values for this component's properties
UActorEventLoggerComponent::UActorEventLoggerComponent()
{
	// Set this component to be initialized when the game starts, and to be ticked every frame.  You can turn these features
	// off to improve performance if you don't need them.
	PrimaryComponentTick.bCanEverTick = true;

	// ...
}


// Called when the game starts
void UActorEventLoggerComponent::BeginPlay()
{
	Super::BeginPlay();

	TArray<AActor*> actors;
	UGameplayStatics::GetAllActorsOfClass(GetWorld(), AActorEventLogManager::StaticClass(), actors);

	for (auto &actor : actors)
	{
		m_ActorEventLogMgr = Cast<AActorEventLogManager>(actor);
		if (m_ActorEventLogMgr)
		{
			m_ActorEventLogMgr->RegisterActorEventLogger(*this);
			break;
		}
	}

}


// Called every frame
void UActorEventLoggerComponent::TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
	Super::TickComponent(DeltaTime, TickType, ThisTickFunction);

	// ...
}

void UActorEventLoggerComponent::StartLogging(const FString &BasePath)
{
	FString relPath("ActorEventLogs");
	FString path = FPaths::Combine(BasePath, relPath);
	m_EventLogFileName = FPaths::Combine(path, m_UniqueActorName);
	m_EventLogFileName += ".ael";

	IPlatformFile& platformFile = FPlatformFileManager::Get().GetPlatformFile();
	if (platformFile.DirectoryExists(*path) == false)
	{
		platformFile.CreateDirectory(*path);
	}

	m_LogStream.open(*m_EventLogFileName, std::ios::out | std::ios::trunc );
	if(m_LogStream.is_open())
	{
		UE_LOG(LogActorEventLogger, Log, TEXT("Start Event logging to %s for %s"), *(m_EventLogFileName), *(AActor::GetDebugName(GetOwner())));

		FVector pos = GetOwner()->GetActorLocation();
		FRotator rot = GetOwner()->GetActorRotation();
		FVector scale = GetOwner()->GetActorScale3D();

		m_LogStream << "{" << std::endl;
		m_LogStream << "\"Name\": \"" << TCHAR_TO_ANSI(*ActorName) << "\"," << std::endl;
		m_LogStream << "\"Description\": \"" << TCHAR_TO_ANSI(*ActorDescription) << "\"," << std::endl;
		if(ActorClassName.IsEmpty() == false)
			m_LogStream << "\"Class\": \"" << TCHAR_TO_ANSI(*ActorClassName) << "\"," << std::endl;
		m_LogStream << "\"KeepTransformFixed\": \"" << (KeepTransformFixed ? "true" : "false") << "\"," << std::endl;
		m_LogStream << "\"InitialTransform\": [" << pos.X << "," << pos.Y << "," << pos.Z << "," << rot.Pitch << "," << rot.Yaw << "," << rot.Roll << "," << scale.X << "," << scale.Y << "," << scale.Z << "],";
		m_LogStream << "\"Frames\": [" << std::endl;

		m_isFirstFrame = true;
	}
	else
	{
		UE_LOG(LogActorEventLogger, Log, TEXT("Event logging failed to %s for %s"), *(m_EventLogFileName), *(AActor::GetDebugName(GetOwner())));
	}
}

void UActorEventLoggerComponent::StopLogging()
{
	if(m_LogStream.is_open())
	{
		m_LogStream << std::endl << "]" << std::endl << "}" << std::endl;
		m_LogStream.close();
	}
}


void UActorEventLoggerComponent::BeginFrame(uint32 FrameCounter, double Timestamp)
{
	/*
	{
	"FrameBegin":	123,
	"Events":	[
					{
						"Type":		"Message",
						"Timestamp": 123,
						"Data":		"Route calculated"
					}
				],
	"FrameEnd":		125
	},
	*/

	if(m_LogStream.is_open())
	{
		if(m_isFirstFrame)
		{
			m_LogStream << "{";
			m_isFirstFrame = false;
		}
		else
			m_LogStream << "," << std::endl << "{";

		m_hasSeenBeginFrame = true;
		m_isFirstEvent = true;

		m_LogStream << "\"FrameBegin\": " << Timestamp << ",";
		m_LogStream << "\"Events\": [" << std::endl;

	}
}


void UActorEventLoggerComponent::EndFrame(double Timestamp)
{
	if	(	m_hasSeenBeginFrame
		&&	m_LogStream.is_open()
		)
	{
		m_LogStream << "], \"FrameEnd\": " << Timestamp << "}";
	}

	m_hasSeenBeginFrame = false;
}

void UActorEventLoggerComponent::LogMessage(const FString &Message)
{
	/*
	{
	"Type":		"Message",
	"Timestamp": 124,
	"Data":	"My nice log message"
	}
	*/
	if	(	m_hasSeenBeginFrame
		&&	m_LogStream.is_open()
		)
	{
		if (m_isFirstEvent == false)
			m_LogStream << ",";
		m_LogStream << "{\"Type\":\"Message\",\"Timestamp\":" << m_ActorEventLogMgr->getTimestamp() <<",";
		m_LogStream << "\"Data\":\"" << TCHAR_TO_ANSI(*Message) << "\"}";

		m_isFirstEvent = false;
	}
}

void UActorEventLoggerComponent::LogActorTransform(const FTransform &Transform)
{
	/*
	{
	"Type":		"ActorTransform",
	"Timestamp": 124,
	"Data":		[0,0,0,0,0,0,1,1,1]
	}
	*/
	if	(	m_hasSeenBeginFrame
		&&	m_LogStream.is_open()
		)
	{
		FVector pos = Transform.GetLocation();
		FRotator rot = Transform.GetRotation().Rotator();
		FVector scale = Transform.GetScale3D();
		if (m_isFirstEvent == false)
			m_LogStream << ",";
		m_LogStream << "{\"Type\":\"ActorTransform\",\"Timestamp\":" << m_ActorEventLogMgr->getTimestamp() <<",";
		m_LogStream << "\"Data\":[" << pos.X << "," << pos.Y << "," << pos.Z << "," << rot.Pitch << "," << rot.Yaw << "," << rot.Roll << "," << scale.X << "," << scale.Y << "," << scale.Z << "]}";

		m_isFirstEvent = false;
	}
}

void UActorEventLoggerComponent::LogEvent(const ActorEvent &Event)
{
}
