

#pragma once

#include "GameFramework/Actor.h"

#include "Public/Server/IDeepDriveServerProxy.h"
#include "Public/Simulation/DeepDriveSimulationDefines.h"

#include "DeepDriveSimulation.generated.h"

DECLARE_LOG_CATEGORY_EXTERN(LogDeepDriveSimulation, Log, All);

class ADeepDriveAgent;
class ADeepDriveAgentControllerBase;


UCLASS()
class DEEPDRIVEPLUGIN_API ADeepDriveSimulation	:	public AActor
												,	public IDeepDriveServerProxy
{
	GENERATED_BODY()
	
public:	
	// Sets default values for this actor's properties
	ADeepDriveSimulation();

	// Called when the game starts or when spawned
	virtual void PreInitializeComponents() override;

	// Called when the game starts or when spawned
	virtual void BeginPlay() override;
	
	// Called every frame
	virtual void Tick( float DeltaSeconds ) override;

	virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

	/**
	*		IDeepDriveServerProxy methods
	*/

	virtual void RegisterClient(int32 ClientId, bool IsMaster);

	virtual void UnregisterClient(int32 ClientId, bool IsMaster);

	virtual int32 RegisterCaptureCamera(float FieldOfView, int32 CaptureWidth, int32 CaptureHeight, FVector RelativePosition, FVector RelativeRotation, const FString &Label);

	virtual bool RequestAgentControl();

	virtual void ReleaseAgentControl();

	virtual void ResetAgent();

	virtual void SetAgentControlValues(float steering, float throttle, float brake, bool handbrake);

	virtual const FString& getIPAddress() const;

	virtual uint16 getPort() const;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Server)
	FString		IPAddress;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Server)
	int32		Port = 9876;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Agents)
	TSubclassOf<ADeepDriveAgent>	Agent;

	UFUNCTION(BlueprintCallable, Category = "Input")
	void MoveForward(float AxisValue);

	UFUNCTION(BlueprintCallable, Category = "Input")
	void MoveRight(float AxisValue);

	UFUNCTION(BlueprintCallable, Category = "Input")
	void LookUp(float AxisValue);

	UFUNCTION(BlueprintCallable, Category = "Input")
	void Turn(float AxisValue);

	UFUNCTION(BlueprintCallable, Category = "Input")
	void OnCameraSelect(EDeepDriveAgentCameraType CameraType);

	UFUNCTION(BlueprintCallable, Category = "Input")
	void OnSelectMode(EDeepDriveAgentControlMode Mode);
	

private:

	ADeepDriveAgent* spawnAgent(EDeepDriveAgentControlMode mode);

	ADeepDriveAgentControllerBase* spawnController(EDeepDriveAgentControlMode mode);

	bool									m_isActive = false;

	ADeepDriveAgent							*m_curAgent = 0;
	EDeepDriveAgentControlMode				m_curAgentMode = EDeepDriveAgentControlMode::DDACM_NONE;

	ADeepDriveAgentControllerBase			*m_curAgentController = 0;

	EDeepDriveAgentCameraType				m_curCameraType = EDeepDriveAgentCameraType::DDAC_CHASE_CAMERA;

};


inline const FString& ADeepDriveSimulation::getIPAddress() const
{
	return IPAddress;
}

inline uint16 ADeepDriveSimulation::getPort() const
{
	return static_cast<uint16> (Port);
}
