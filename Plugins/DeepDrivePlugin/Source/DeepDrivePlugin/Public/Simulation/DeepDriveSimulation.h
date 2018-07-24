

#pragma once

#include "GameFramework/Actor.h"

#include "Public/Simulation/DeepDriveSimulationDefines.h"

#include "DeepDriveSimulation.generated.h"

DECLARE_LOG_CATEGORY_EXTERN(LogDeepDriveSimulation, Log, All);

class DeepDriveSimulationCaptureProxy;
class DeepDriveSimulationServerProxy;
class DeepDriveSimulationStateMachine;

class DeepDriveSimulationRunningState;
class DeepDriveSimulationReseState;

class ADeepDriveAgent;
class ADeepDriveAgentControllerCreator;
class ADeepDriveAgentControllerBase;
class UCaptureSinkComponentBase;
class ADeepDriveSimulationFreeCamera;
class ADeepDriveSplineTrack;
class UDeepDriveRandomStream;

namespace deepdrive { namespace server {
struct SimulationConfiguration;
struct SimulationGraphicsSettings;
} }


USTRUCT(BlueprintType)
struct FDeepDriveRandomStreamData
{
	GENERATED_USTRUCT_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	bool	ReSeedOnReset = true;

	FDeepDriveRandomStreamData()
		:	ReSeedOnReset(true)
		,	RandomStream(0)
	{	}

	FDeepDriveRandomStreamData(UDeepDriveRandomStream *randomStream, bool reseedOnReset)
		:	ReSeedOnReset(reseedOnReset)
		,	RandomStream(randomStream)
	{	}

	void setRandomStream(UDeepDriveRandomStream *randomStream)
	{
		RandomStream = randomStream;
	}

	UDeepDriveRandomStream* getRandomStream()
	{
		return RandomStream;
	}

private:

	UPROPERTY()
	UDeepDriveRandomStream		*RandomStream = 0;
};

USTRUCT(BlueprintType)
struct FDeepDriveAdditionalAgentData
{
	GENERATED_USTRUCT_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	TSubclassOf<ADeepDriveAgent>	Agent;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	EDeepDriveAgentControlMode		Mode;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	int32	ConfigurationSlot;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Default)
	int32	StartPositionSlot;
};

UCLASS()
class DEEPDRIVEPLUGIN_API ADeepDriveSimulation	:	public AActor
{
	friend class DeepDriveSimulationRunningState;
	friend class DeepDriveSimulationResetState;

	GENERATED_BODY()

public:	
	// Sets default values for this actor's properties
	ADeepDriveSimulation();

	~ADeepDriveSimulation();

	// Called when the game starts or when spawned
	virtual void PreInitializeComponents() override;

	// Called when the game starts or when spawned
	virtual void BeginPlay() override;
	
	// Called every frame
	virtual void Tick( float DeltaSeconds ) override;

	virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Capture)
	float CaptureInterval = 0.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Server)
	FString		IPAddress;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Server)
	int32		Port = 9876;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Agents)
	TSubclassOf<ADeepDriveAgent>	Agent;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Agents)
	EDeepDriveAgentControlMode	InitialControllerMode = EDeepDriveAgentControlMode::NONE;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Agents)
	int32	InitialConfigurationSlot;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Agents)
	int32	StartPositionSlot;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Agents)
	TArray<FDeepDriveAdditionalAgentData>	AdditionalAgents;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Agents)
	TMap<EDeepDriveAgentControlMode, ADeepDriveAgentControllerCreator*>	ControllerCreators;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = FreeCamera)
	ADeepDriveSimulationFreeCamera	*FreeCamera = 0;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Randomness)
	int32	Seed;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Randomness)
	TMap<FName, FDeepDriveRandomStreamData>	RandomStreams;

	UFUNCTION(BlueprintCallable, Category = "Simulation")
	void ResetSimulation();

	UFUNCTION(BlueprintCallable, Category = "Input")
	void MoveForward(float AxisValue);

	UFUNCTION(BlueprintCallable, Category = "Input")
	void MoveRight(float AxisValue);

	UFUNCTION(BlueprintCallable, Category = "Input")
	void LookUp(float AxisValue);

	UFUNCTION(BlueprintCallable, Category = "Input")
	void Turn(float AxisValue);

	UFUNCTION(BlueprintCallable, Category = "Input")
	void SelectCamera(EDeepDriveAgentCameraType CameraType);

	UFUNCTION(BlueprintCallable, Category = "Input")
	void SelectMode(EDeepDriveAgentControlMode Mode);

	UFUNCTION(BlueprintCallable, Category = "Spectator")
	void NextAgent();

	UFUNCTION(BlueprintCallable, Category = "Spectator")
	void PreviousAgent();

	UFUNCTION(BlueprintImplementableEvent, Category = "Agents")
	void OnAgentSpawned(ADeepDriveAgent *SpawnedAgent);

	UFUNCTION(BlueprintImplementableEvent, Category = "Agents")
	void OnCurrentAgentChanged(ADeepDriveAgent *CurrentAgent);

	UFUNCTION(BlueprintCallable, Category = "Misc")
	void RegisterRandomStream(const FName &RandomStreamId, bool ReseedOnReset);

	UFUNCTION(BlueprintCallable, Category = "Misc")
	UDeepDriveRandomStream* GetRandomStream(const FName &RandomStreamId);

	UFUNCTION(BlueprintCallable, Category = "Agents")
	void OnDebugTrigger();

	void configure(const deepdrive::server::SimulationConfiguration &configuration, const deepdrive::server::SimulationGraphicsSettings &graphicsSettings, bool initialConfiguration);
	bool resetAgent();
	
	ADeepDriveAgent* getCurrentAgent() const;
	ADeepDriveAgentControllerBase* getCurrentAgentController() const;
	TArray<UCaptureSinkComponentBase*>& getCaptureSinks();

	void initializeAgents();

private:

	bool isActive() const;

	ADeepDriveAgent* spawnAgent(EDeepDriveAgentControlMode mode, int32 configSlot, int32 startPosSlot);

	void spawnAdditionalAgents();

	ADeepDriveAgentControllerBase* spawnController(EDeepDriveAgentControlMode mode, int32 configSlot, int32 startPosSlot);

	void switchToAgent(int32 index);
	void switchToCamera(EDeepDriveAgentCameraType type);

	void applyGraphicsSettings(const deepdrive::server::SimulationGraphicsSettings &gfxSettings);

	DeepDriveSimulationStateMachine			*m_StateMachine = 0;
	DeepDriveSimulationServerProxy			*m_ServerProxy = 0;
	DeepDriveSimulationCaptureProxy			*m_CaptureProxy = 0;
	TArray<UCaptureSinkComponentBase*>		m_CaptureSinks;

	TArray<ADeepDriveAgent*>				m_Agents;
	int32									m_curAgentIndex = 0;
	ADeepDriveAgent							*m_curAgent = 0;
	EDeepDriveAgentControlMode				m_curAgentMode = EDeepDriveAgentControlMode::NONE;


	ADeepDriveAgentControllerBase			*m_curAgentController = 0;

	EDeepDriveAgentCameraType				m_curCameraType = EDeepDriveAgentCameraType::NONE;
	float									m_OrbitCameraPitch = 0.0f;
	float									m_OrbitCameraYaw = 0.0f;
};

inline bool ADeepDriveSimulation::isActive() const
{
	return m_StateMachine != 0;
}

inline ADeepDriveAgent* ADeepDriveSimulation::getCurrentAgent() const
{
	return m_curAgent;
}

inline ADeepDriveAgentControllerBase* ADeepDriveSimulation::getCurrentAgentController() const
{
	return m_curAgentController;
}

inline TArray<UCaptureSinkComponentBase*>& ADeepDriveSimulation::getCaptureSinks()
{
	return m_CaptureSinks;
}
