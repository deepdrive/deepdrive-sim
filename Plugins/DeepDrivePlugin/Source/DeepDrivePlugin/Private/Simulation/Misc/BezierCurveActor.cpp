
#include "DeepDrivePluginPrivatePCH.h"
#include "BezierCurveActor.h"
#include "BezierCurveComponent.h"
#include "Runtime/Engine/Classes/Components/SphereComponent.h"

// Sets default values
ABezierCurveActor::ABezierCurveActor()
{
 	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;

	Root = CreateDefaultSubobject<USceneComponent>(TEXT("Root"));
	SetRootComponent(Root);

	m_BezierCurve = CreateDefaultSubobject<UBezierCurveComponent>(TEXT("BezierCurve"));
}

void ABezierCurveActor::ClearControlPoints()
{
	m_ControlPoints.Empty();
	m_BezierCurve->ClearControlPoints();
}

void ABezierCurveActor::AddControlPoint(USceneComponent *ControlPoint)
{
	m_ControlPoints.Add(ControlPoint);
	m_BezierCurve->AddControlPoint(ControlPoint->GetComponentLocation());
}

// Called when the game starts or when spawned
void ABezierCurveActor::BeginPlay()
{
	Super::BeginPlay();

}

// Called every frame
void ABezierCurveActor::Tick(float DeltaTime)
{
#if WITH_EDITOR

	Super::Tick(DeltaTime);

	if (m_ControlPoints.Num() > 2)
	{
		int32 index = 0;
		for (auto &p : m_ControlPoints)
		{
			m_BezierCurve->UpdateControlPoint(index++, p->GetComponentLocation());
		}

		int32 numSteps = 20;
		const float deltaT = 1.0f / static_cast<int32> (numSteps);
		float curT = deltaT;

		const uint8 prio = 200;

		FVector pos0 = m_BezierCurve->Evaluate(0.0f);
		DrawDebugPoint(GetWorld(), pos0, 15.0f, Color, false, 0.0f, 200);
		for (int32 i = 1; i <= numSteps; ++i)
		{
			const FVector pos1 = m_BezierCurve->Evaluate(curT);

			DrawDebugPoint(GetWorld(), pos1, 15.0f, Color, false, 0.0f, prio);
			DrawDebugLine(GetWorld(), pos0, pos1, Color, false, 0.0f, prio, 8.0f);

			pos0 = pos1;
			curT += deltaT;
		}
	}

#endif
}

bool ABezierCurveActor::ShouldTickIfViewportsOnly() const
{
	return true;
}

