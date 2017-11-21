// Fill out your copyright notice in the Description page of Project Settings.

#include "DeepDrivePluginPrivatePCH.h"
#include "DeepDrivePluginBPFunctionLibrary.h"

#include "Capture/DeepDriveCapture.h"


void UDeepDrivePluginBPFunctionLibrary::Capture()
{
	DeepDriveCapture::GetInstance().Capture();
}
