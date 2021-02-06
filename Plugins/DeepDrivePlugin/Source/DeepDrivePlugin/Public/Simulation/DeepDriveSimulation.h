

#pragma once

#include "GameFramework/Actor.h"

#include "Simulation/DeepDriveSimulationDefines.h"

#include <map>
#include <functional>

#include "DeepDriveSimulation.generated.h"

DECLARE_LOG_CATEGORY_EXTERN(LogDeepDriveSimulation, Log, All);

class DeepDriveSimulationCaptureProxy;
class DeepDriveSimulationServerProxy;
class DeepDriveSimulationStateMachine;
class DeepDriveSimulationServer;
class DeepDriveSimulationRequestHandler;
class DeepDriveSimulationConfigureState;
class DeepDriveSimulationResetState;
class DeepDriveManeuverCalculator;

class DeepDriveSimulationRunningState;
class DeepDriveSimulationReseState;

class ADeepDriveAgent;
class ADeepDriveAgentControllerCreator;
class ADeepDriveAgentControllerBase;
class UCaptureSinkComponentBase;
class ADeepDriveSimulationFreeCamera;
class ADeepDriveSplineTrack;
class UDeepDriveRandomStream;

class UDeepDriveRoadNetworkComponent;

struct SimulationGraphicsSettings;

namespace deepdrive { namespace server {
struct MessageHeader;
} }

UCLASS(Abstract)
class DEEPDRIVEPLUGIN_API ADeepDriveSimulation	:	public AActor
{
	friend class DeepDriveSimulationRunningState;
	friend class DeepDriveSimulationConfigureState;
	friend class DeepDriveSimulationResetState;
	friend class DeepDriveSimulationRequests;
	friend class DeepDriveMultiAgentRequests;

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

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = SimulationServer)
	FString		SimulationIPAddress = "127.0.0.1";

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = SimulationServer)
	int32		SimulationPort = 9009;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = ClientConnections)
	FString		ClientsIPAddress = "127.0.0.1";

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = ClientConnections)
	int32		ClientsPort = 9876;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Agents)
	TSubclassOf<ADeepDriveAgent>	Agent;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Agents)
	ADeepDriveAgentControllerCreator	*ControllerCreator = 0;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Agents)
	int32	InitialConfigurationSlot;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Agents)
	int32	StartPositionSlot;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Agents)
	TArray<FDeepDriveAdditionalAgentData>	AdditionalAgents;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Agents)
	ADeepDriveSplineTrack	*OneOffAgentsTrack = 0;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = FreeCamera)
	ADeepDriveSimulationFreeCamera	*FreeCamera = 0;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Randomness)
	int32	Seed;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Randomness)
	TMap<FName, FDeepDriveRandomStreamData>	RandomStreams;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = ViewModes)
	TMap<FString, FDeepDriveViewMode>	ViewModes;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = RoadNetwork)
	UDeepDriveRoadNetworkComponent	*RoadNetwork = 0;

	UPROPERTY(BlueprintReadOnly, Category = Scenario)
	bool	ScenarioMode = false;

#if WITH_EDITORONLY_DATA
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Debug)
	bool	InitialScenarioMode = false;
#endif

	UFUNCTION(BlueprintCallable, Category = "Simulation")
	void ConfigureSimulation(FDeepDriveScenarioConfiguration Configuration);

	UFUNCTION(BlueprintCallable, Category = "Simulation")
	void ResetSimulation(bool ActivateAdditionalAgents);

	UFUNCTION(BlueprintCallable, Category = "Input")
	void MoveForward(float AxisValue);

	UFUNCTION(BlueprintCallable, Category = "Input")
	void MoveRight(float AxisValue);

	UFUNCTION(BlueprintCallable, Category = "Input")
	void LookUp(float AxisValue);

	UFUNCTION(BlueprintCallable, Category = "Input")
	void Turn(float AxisValue);

	UFUNCTION(BlueprintCallable, Category = "Input")
	void Brake(float AxisValue);

	UFUNCTION(BlueprintCallable, Category = "Input")
	void SelectCamera(EDeepDriveAgentCameraType CameraType);

	UFUNCTION(BlueprintCallable, Category = "Spectator")
	void NextAgent();

	UFUNCTION(BlueprintCallable, Category = "Spectator")
	void PreviousAgent();

	UFUNCTION(BlueprintImplementableEvent, Category = "Simulation")
	void OnSimulationReady();

	UFUNCTION(BlueprintImplementableEvent, Category = "Agents")
	void OnAgentSpawned(ADeepDriveAgent *SpawnedAgent);

	UFUNCTION(BlueprintImplementableEvent, Category = "Agents")
	void OnCurrentAgentChanged(ADeepDriveAgent *CurrentAgent);

	UFUNCTION(BlueprintImplementableEvent, Category = "Simulation")
	void SetDateAndTime(int32 Year, int32 Month, int32 Day, int32 Hour, int32 Minute);

	UFUNCTION(BlueprintImplementableEvent, Category = "Simulation")
	void SetSunSimulationSpeed(int32 Speed);

	UFUNCTION(BlueprintCallable, Category = "Misc")
	void RegisterRandomStream(const FName &RandomStreamId, bool ReseedOnReset);

	UFUNCTION(BlueprintCallable, Category = "Misc")
	UDeepDriveRandomStream* GetRandomStream(const FName &RandomStreamId);

	UFUNCTION(BlueprintCallable, Category = "Agents")
	void ToggleAgentRenderMode();

	UFUNCTION(BlueprintCallable, Category = "Agents")
	void ToggleAgentCollisionMode();

	UFUNCTION(BlueprintCallable, Category = "Agents")
	void ToggleCollisionVisibility();

	UFUNCTION(BlueprintCallable, Category = "Agents")
	int32 AddOneOffAgent(TSubclassOf<ADeepDriveAgent> AgentClass, const FTransform &Transform, bool BindToRoad);

	UFUNCTION(BlueprintCallable, Category = "Agents")
	void SetOneOffAgentControlValue(int32 AgentId, float Steering, float Throttle, float Brake, bool Handbrake);

	UFUNCTION(BlueprintCallable, Category = "Agents")
	void OnDebugTrigger();

	UFUNCTION(BlueprintCallable, Category = "Agents")
	TArray<ADeepDriveAgent*> GetAgentsList(EDeepDriveAgentsListFilter Filter);

	UFUNCTION(BlueprintCallable, Category = "Path")
	FDeepDrivePath GetEgoPath();

	void enqueueMessage(deepdrive::server::MessageHeader *message);

	bool resetAgent();

	bool requestControl();
	void releaseControl();

	ADeepDriveAgent* getCurrentAgent() const;
	ADeepDriveAgentControllerBase* getCurrentAgentController() const;
	TArray<UCaptureSinkComponentBase*>& getCaptureSinks();

	DeepDriveManeuverCalculator* getManeuverCalculator();

	TArray<ADeepDriveAgent*> getAgents(const FBox2D &area, ADeepDriveAgent *excludeAgent);

	void initializeAgents();
	void removeAgents(bool removeEgo);
	void spawnAdditionalAgents();
	bool hasAdditionalAgents();

	bool hasEgoAgent() const;
	void onEgoAgentChanged(bool added);

	void removeOneOffAgents();

	bool isLocationOccupied(const FVector &location, float radius);

	static FDateTime getSimulationStartTime();

private:

	bool isActive() const;

	ADeepDriveAgent *spawnAgent(const FDeepDriveAgentScenarioConfiguration &scenarioCfg, bool remotelyControlled);

	ADeepDriveAgent *spawnAgent(ADeepDriveAgentControllerCreator *ctrlCreator, int32 configSlot, int32 startPosSlot);

	void switchToAgent(int32 index);
	void switchToCamera(EDeepDriveAgentCameraType type);

	void applyGraphicsSettings(const SimulationGraphicsSettings &gfxSettings);

	DeepDriveSimulationStateMachine			*m_StateMachine = 0;
	DeepDriveSimulationConfigureState		*m_ConfigureState = 0;
	DeepDriveSimulationResetState			*m_ResetState = 0;
	DeepDriveSimulationServer				*m_SimulationServer = 0;
	DeepDriveSimulationRequestHandler		*m_RequestHandler = 0;
	DeepDriveSimulationServerProxy			*m_ServerProxy = 0;
	DeepDriveSimulationCaptureProxy			*m_CaptureProxy = 0;
	TArray<UCaptureSinkComponentBase*>		m_CaptureSinks;

	FDeepDriveRandomStreamData				m_DefaultRandomStream;

	DeepDriveManeuverCalculator				*m_ManeuverCalculator = 0;

	TArray<ADeepDriveAgent*>				m_Agents;
	int32									m_curAgentIndex = 0;
	ADeepDriveAgent							*m_curAgent = 0;

	TMap<int32, ADeepDriveAgent*>			m_OneOffAgents;
	int32									m_nextOneOffAgentId = 1;

	ADeepDriveAgentControllerBase			*m_curAgentController = 0;

	EDeepDriveAgentCameraType				m_curCameraType = EDeepDriveAgentCameraType::NONE;
	float									m_OrbitCameraPitch = 0.0f;
	float									m_OrbitCameraYaw = 0.0f;

	bool									m_SimpleRenderMode = false;
	bool									m_SimpleCollisionMode = false;
	bool									m_CollisionVisibility = false;

	int32									m_numEgoAgents = 0;
	
	static FDateTime						m_SimulationStartTime;
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

inline FDateTime ADeepDriveSimulation::getSimulationStartTime()
{
	return ADeepDriveSimulation::m_SimulationStartTime;
}

inline bool ADeepDriveSimulation::hasAdditionalAgents()
{
	return m_Agents.Num() > 1;
}
