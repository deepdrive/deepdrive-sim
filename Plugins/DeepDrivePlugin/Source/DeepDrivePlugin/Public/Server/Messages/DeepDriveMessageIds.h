
#pragma once

#include "Engine.h"

namespace deepdrive { namespace server {

enum class MessageId	:	uint32
{
	/*
		Connection handling
	*/
	Undefined,
	RegisterClientRequest,
	RegisterClientResponse,
	UnregisterClientRequest,
	UnregisterClientResponse,
	KeepAliveRequest,
	KeepAliveResponse,


	/*
		Configuration
	*/
	InitializeSimulationRequest,
	InitializeSimulationResponse,
	SetSunSimulationRequest,
	SetSunSimulationResponse,
	RegisterCaptureCameraRequest,
	RegisterCaptureCameraResponse,
	SetCapturePatternRequest,
	SetCapturePatternResponse,
	SetCaptureConfigurationRequest,
	SetCaptureConfigurationResponse,


	/*
		Control
	*/
	RequestAgentControlRequest,
	RequestAgentControlResponse,
	SetAgentControlValuesRequest,
	ResetAgentRequest,
	ResetAgentResponse,
	ReleaseAgentControlRequest,
	ReleaseAgentControlResponse,
	ActivateSynchronousSteppingRequest,
	ActivateSynchronousSteppingResponse,
	DeactivateSynchronousSteppingRequest,
	DeactivateSynchronousSteppingResponse,
	AdvanceSynchronousSteppingRequest,
	AdvanceSynchronousSteppingResponse

};
	
} }		//	namespaces
