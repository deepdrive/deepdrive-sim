// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "Kismet/BlueprintFunctionLibrary.h"
#include "DeepDrivePluginBPFunctionLibrary.generated.h"


/**
 * 
 */
UCLASS()
class DEEPDRIVEPLUGIN_API UDeepDrivePluginBPFunctionLibrary : public UBlueprintFunctionLibrary
{
	GENERATED_BODY()

public:

	/**
		Register a capture camera component with DeepDriveCapture
	*/	
	UFUNCTION(BlueprintCallable, Category="DeepDrivePlugin")
	static void Capture();

	
	
};
