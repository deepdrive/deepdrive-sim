
#pragma once

#include "Public/Server/Messages/DeepDriveServerMessageHeader.h"


namespace deepdrive { namespace server {


struct RegisterCaptureCameraRequest	:	public MessageHeader
{
	RegisterCaptureCameraRequest(uint32 clientId, float hFoV, uint16 captureWidth, uint16 captureHeight)
		:	MessageHeader(MessageId::RegisterCaptureCameraRequest, sizeof(RegisterCaptureCameraRequest))
		,	client_id(clientId)
		,	horizontal_field_of_view(hFoV)
		,	capture_width(captureWidth)
		,	capture_height(captureHeight)
	{	}

	uint32				client_id;
	float				horizontal_field_of_view;
	uint16				capture_width;
	uint16				capture_height;
 	float				relative_position[3];
	float				relative_rotation[3];

};

struct RegisterCaptureCameraResponse	:	public MessageHeader
{
	RegisterCaptureCameraResponse(uint32 camId = 0)
		:	MessageHeader(MessageId::RegisterCaptureCameraResponse, sizeof(RegisterCaptureCameraResponse))
		,	camera_id(camId)
	{	}

	uint32		camera_id;

};



} }	// namespaces
