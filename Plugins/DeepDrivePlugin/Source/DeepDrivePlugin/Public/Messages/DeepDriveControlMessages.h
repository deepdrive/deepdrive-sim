
#pragma once

#include "Engine.h"
#include "Public/Messages/DeepDriveMessageHeader.h"

struct DeepDriveControlMessage	:	public DeepDriveMessageHeader
{
	DeepDriveControlMessage()
		: DeepDriveMessageHeader(DeepDriveMessageType::Control, sizeof(DeepDriveControlMessage) - sizeof(DeepDriveMessageHeader))
	{	}

	double						capture_timestamp;

	uint32						capture_sequence_number;

	uint32						should_reset;

	double						steering;

	double						throttle;

	double						brake;

	double						handbrake;

	uint32						is_game_driving;

};


struct DeepDriveCameraConfig
{
	uint32						id;

	uint32						is_active;

	double						horizontal_field_of_view;
	double						aspect_ratio;

	DeepDriveVector3			relative_position;
	DeepDriveVector3			relative_rotation;

};

struct DeepDriveDisconnectControl : public DeepDriveMessageHeader
{
	DeepDriveDisconnectControl()
		: DeepDriveMessageHeader(DeepDriveMessageType::DisconnectControl, sizeof(DeepDriveDisconnectControl) - sizeof(DeepDriveMessageHeader))
	{	}

};

struct DeepDriveCamereConfiguration : public DeepDriveMessageHeader
{
	DeepDriveCamereConfiguration()
		: DeepDriveMessageHeader(DeepDriveMessageType::Control, sizeof(DeepDriveCamereConfiguration) - sizeof(DeepDriveMessageHeader))
	{	}

	uint32						num_cameras;

	DeepDriveCameraConfig		camera_configs[1];

};
