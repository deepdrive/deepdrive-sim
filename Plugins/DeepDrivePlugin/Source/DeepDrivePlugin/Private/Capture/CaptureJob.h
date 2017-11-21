#pragma once

#include "Engine.h"

class CaptureBuffer;
class CaptureBufferPool;

struct SCaptureDestinationData
{
	void					*destination_buffer = 0;
	EPixelFormat			pixel_format = PF_Unknown;
	uint32					buffer_size = 0;
	uint32					buffer_stride = 0;
	uint32					width = 0;
	uint32					height = 0;
};

struct SCaptureRequest
{
	EDeepDriveCameraType			camera_type;
	int32							camera_id = 0;
	FTextureRenderTargetResource	*capture_source = 0;
	CaptureBuffer					*capture_buffer = 0;
};

struct SCaptureJob;

struct SCaptureJob
{
	double						timestamp;
	int32						sequence_number = 0;
	TArray<SCaptureRequest>		capture_requests;

	CaptureBufferPool			*capture_buffer_pool = 0;
	TQueue<SCaptureJob*>		*result_queue = 0;
};
