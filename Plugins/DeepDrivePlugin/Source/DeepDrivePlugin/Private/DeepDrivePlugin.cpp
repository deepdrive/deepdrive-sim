// Copyright 1998-2016 Epic Games, Inc. All Rights Reserved.

#include "DeepDrivePluginPrivatePCH.h"

#define LOCTEXT_NAMESPACE "FDeepDrivePluginModule"

DEFINE_LOG_CATEGORY(LogDeepDrivePlugin);

void FDeepDrivePluginModule::StartupModule()
{
	// This code will execute after your module is loaded into memory; the exact timing is specified in the .uplugin file per-module
	UE_LOG(LogDeepDrivePlugin, Log, TEXT(">>>>>> DeepDrivePlugin loaded"));
}

void FDeepDrivePluginModule::ShutdownModule()
{
	// This function may be called during shutdown to clean up your module.  For modules that support dynamic reloading,
	// we call this function before unloading the module.
	UE_LOG(LogDeepDrivePlugin, Log, TEXT("<<<<<<< DeepDrivePlugin unloaded"));
}

#undef LOCTEXT_NAMESPACE
	
IMPLEMENT_MODULE(FDeepDrivePluginModule, DeepDrivePlugin)
