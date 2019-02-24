

#pragma once

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "DeepDriveRandomStream.generated.h"

/**
 * 
 */
UCLASS()
class DEEPDRIVEPLUGIN_API UDeepDriveRandomStream : public UObject
{
	GENERATED_BODY()
	
public:

	void initialize(int32 seed);

	UFUNCTION(BlueprintCallable, Category = "Misc")
	int32 RandomInteger(int32 Max);

	UFUNCTION(BlueprintCallable, Category = "Misc")
	int32 RandomIntegerInRange(int32 Min, int32 Max);

	UFUNCTION(BlueprintCallable, Category = "Misc")
	float RandomFloat();

	UFUNCTION(BlueprintCallable, Category = "Misc")
	float RandomFloatInRange(float Min, float Max);

private:
	
	FRandomStream				m_RandomStream;

};
