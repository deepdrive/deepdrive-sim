

#include "DeepDrivePluginPrivatePCH.h"
#include "DeepDrivePlugin.h"
#include "DeepDriveRoadSegmentProxy.h"

// Sets default values
ADeepDriveRoadSegmentProxy::ADeepDriveRoadSegmentProxy()
{
 	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;

	Root = CreateDefaultSubobject<USceneComponent>(TEXT("Root"));
	SetRootComponent(Root);

	StartPoint = CreateDefaultSubobject<UArrowComponent>(TEXT("StartPoint"));
	StartPoint->SetupAttachment(Root);

	EndPoint = CreateDefaultSubobject<UArrowComponent>(TEXT("EndPoint"));
	EndPoint->SetupAttachment(Root);

	Spline = CreateDefaultSubobject<USplineComponent>(TEXT("Spline"));
	Spline->SetupAttachment(Root);
}

// Called when the game starts or when spawned
void ADeepDriveRoadSegmentProxy::BeginPlay()
{
	Super::BeginPlay();

	m_IsGameRunning = true;
	
}

// Called every frame
void ADeepDriveRoadSegmentProxy::Tick(float DeltaTime)
{
#if WITH_EDITOR

	Super::Tick(DeltaTime);

	if (!m_IsGameRunning)
	{
		if(Spline->GetNumberOfSplinePoints() > 1)
		{
			bool updateRequired = false;
			FTransform startTransform = StartPoint->GetComponentTransform();
			if (startTransform.GetLocation() != m_LastStartLocation)
			{
				m_LastStartLocation = startTransform.GetLocation();
				Spline->SetLocationAtSplinePoint(0, m_LastStartLocation, ESplineCoordinateSpace::World, false);
				updateRequired = true;
			}

			FTransform endTransform = EndPoint->GetComponentTransform();
			if (endTransform.GetLocation() != m_LastEndLocation)
			{
				m_LastEndLocation = endTransform.GetLocation();
				Spline->SetLocationAtSplinePoint(Spline->GetNumberOfSplinePoints() - 1, m_LastEndLocation, ESplineCoordinateSpace::World, false);
				updateRequired = true;
			}

			if (updateRequired)
				Spline->UpdateSpline();

			const uint8 prio = 10;
			DrawDebugPoint(GetWorld(), getSplinePoint(0), 15.0f, Color, false, 0.0f, prio);
			for (signed i = 1; i < Spline->GetNumberOfSplinePoints(); ++i)
			{
				DrawDebugPoint(GetWorld(), getSplinePoint(i), 15.0f, Color, false, 0.0f, prio);
				DrawDebugLine(GetWorld(), getSplinePoint(i - 1), getSplinePoint(i), Color, false, 0.0f, prio, 8.0f);
			}

			FVector textPos;
			if (Spline->GetNumberOfSplinePoints() > 2)
			{
				const int32 index = Spline->GetNumberOfSplinePoints() >> 1;
				textPos = getSplinePoint(index);
			}
			else
				textPos = 0.5f * (m_LastStartLocation + m_LastEndLocation) + FVector(0.0f, 0.0f, 10.0f);

			DrawDebugString(GetWorld(), textPos, UKismetSystemLibrary::GetObjectName(this), 0, Color, 0.0f, true);
		}
	}

#endif
}

bool ADeepDriveRoadSegmentProxy::ShouldTickIfViewportsOnly() const
{
	return true;
}

FVector ADeepDriveRoadSegmentProxy::getSplinePoint(int32 index)
{
	FVector location = Spline->GetLocationAtSplinePoint(index, ESplineCoordinateSpace::World);
	FHitResult hitRes;
	if (GetWorld()->LineTraceSingleByChannel(hitRes, location + FVector(0.0f, 0.0f, 50.0f), location - FVector(0.0f, 0.0f, 50.0f), ECC_Visibility, FCollisionQueryParams(), FCollisionResponseParams()))
	{
		location.Z = hitRes.ImpactPoint.Z;
	}
	location.Z += 5.0f;
	return location;
}