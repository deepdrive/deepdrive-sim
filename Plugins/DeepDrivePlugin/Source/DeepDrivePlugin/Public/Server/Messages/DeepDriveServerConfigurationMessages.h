
#pragma once

#include "Public/Server/Messages/DeepDriveServerMessageHeader.h"

#include <cstring>

namespace deepdrive { namespace server {

struct RegisterCaptureCameraRequest	:	public MessageHeader
{
	RegisterCaptureCameraRequest(uint32 clientId, float hFoV, uint16 captureWidth, uint16 captureHeight, const char *label)
		:	MessageHeader(MessageId::RegisterCaptureCameraRequest, sizeof(RegisterCaptureCameraRequest))
		,	client_id(clientId)
		,	horizontal_field_of_view(hFoV)
		,	capture_width(captureWidth)
		,	capture_height(captureHeight)
	{
		if(label)
		{
			strncpy(camera_label, label, MessageHeader::StringSize - 1);
			camera_label[MessageHeader::StringSize - 1] = 0;
		}
		else
			camera_label[0] = 0;
	}

	uint32				client_id;
	float				horizontal_field_of_view;
	uint16				capture_width;
	uint16				capture_height;
 	float				relative_position[3];
	float				relative_rotation[3];

	char				camera_label[MessageHeader::StringSize];
};

struct RegisterCaptureCameraResponse	:	public MessageHeader
{
	RegisterCaptureCameraResponse(uint32 camId = 0)
		:	MessageHeader(MessageId::RegisterCaptureCameraResponse, sizeof(RegisterCaptureCameraResponse))
		,	camera_id(camId)
	{	}

	uint32		camera_id;

};

struct UnregisterCaptureCameraRequest	:	public MessageHeader
{
	UnregisterCaptureCameraRequest(uint32 clientId, uint32 cameraId)
		:	MessageHeader(MessageId::UnregisterCaptureCameraRequest, sizeof(UnregisterCaptureCameraRequest))
		,	client_id(clientId)
		,	camera_id(cameraId)
	{
	}

	uint32				client_id;
	uint32				camera_id;
};

struct UnregisterCaptureCameraResponse	:	public MessageHeader
{
	UnregisterCaptureCameraResponse(bool _unregistered = false)
		:	MessageHeader(MessageId::UnregisterCaptureCameraResponse, sizeof(UnregisterCaptureCameraResponse))
		,	unregistered(_unregistered ? 1 : 0)
	{	}

	uint32		unregistered;

};



} }	// namespaces
