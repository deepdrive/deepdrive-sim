

#include "DeepDrivePluginPrivatePCH.h"
#include "DeepDriveSimulation.h"


// Sets default values
ADeepDriveSimulation::ADeepDriveSimulation()
{
 	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;

}

// Called when the game starts or when spawned
void ADeepDriveSimulation::BeginPlay()
{
	Super::BeginPlay();
	
}

// Called every frame
void ADeepDriveSimulation::Tick( float DeltaTime )
{
	Super::Tick( DeltaTime );

}

void ADeepDriveSimulation::RegisterClient(int32 ClientId, bool IsMaster)
{
}

void ADeepDriveSimulation::UnregisterClient(int32 ClientId, bool IsMaster)
{
}

int32 ADeepDriveSimulation::RegisterCaptureCamera(float FieldOfView, int32 CaptureWidth, int32 CaptureHeight, FVector RelativePosition, FVector RelativeRotation, const FString &Label)
{
	int32 camId = 0;

	return camId;
}

bool ADeepDriveSimulation::RequestAgentControl()
{
	bool res = false;

	return res;
}

void ADeepDriveSimulation::ReleaseAgentControl()
{
}

void ADeepDriveSimulation::ResetAgent()
{
}

void ADeepDriveSimulation::SetAgentControlValues(float steering, float throttle, float brake, bool handbrake)
{
}


void ADeepDriveSimulation::MoveForward(float AxisValue)
{

}

void ADeepDriveSimulation::MoveRight(float AxisValue)
{
	
}

void ADeepDriveSimulation::LookUp(float AxisValue)
{
	
}

void ADeepDriveSimulation::Turn(float AxisValue)
{
	
}

void ADeepDriveSimulation::OnCameraSelect(EDeepDriveAgentCameraType CameraType)
{
	
}

void ADeepDriveSimulation::OnSelectMode(EDeepDriveAgentControlMode Mode)
{
	
}

