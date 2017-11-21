
#pragma once

#include "Engine.h"
#include "Public/Messages/DeepDriveMessageHeader.h"

struct DeepDriveCaptureCamera
{
	uint32						type;
	uint32						id;
	uint32						offset_to_next_camera;			// from beginning of this data structure, set to 0 for last camera
	uint32						padding_0;

	double						horizontal_field_of_view;
	double						aspect_ratio;


	uint32						capture_width;
	uint32						capture_height;
	uint32						bytes_per_pixel;
	uint32						bytes_per_depth_value;
	uint32						depth_offset;					// byte offset of depth data relative to data
	uint32						padding_1;

	uint8						data[1];

};

struct DeepDriveCaptureMessage	:	public DeepDriveMessageHeader
{
	DeepDriveCaptureMessage()
		:	DeepDriveMessageHeader(DeepDriveMessageType::Capture, sizeof(DeepDriveCaptureMessage) - sizeof(DeepDriveMessageHeader))
	{	}

	void addCameraSize(uint32 size)
	{
		message_size += size;
	}

	double						creation_timestamp;

	uint32						sequence_number;

	uint32						padding_0;

	DeepDriveVector3			position;

	DeepDriveVector3			rotation;

	DeepDriveVector3			velocity;

	DeepDriveVector3			acceleration;

	DeepDriveVector3			angular_velocity;

	DeepDriveVector3			angular_acceleration;

	DeepDriveVector3			forward_vector;

	DeepDriveVector3			up_vector;

	DeepDriveVector3			right_vector;

	DeepDriveVector3			dimension;

	double						distance_along_route;

	double						distance_to_center_of_lane;

	int32						lap_number;

	uint32						padding_1;

	double						speed;

	double						steering;

	double						throttle;

	double						brake;

	uint32						handbrake;

	uint32						is_game_driving;

	uint32						is_resetting;

	uint32						num_cameras;

	DeepDriveCaptureCamera		cameras[1];

};
