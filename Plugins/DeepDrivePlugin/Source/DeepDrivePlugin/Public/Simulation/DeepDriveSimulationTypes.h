
#pragma once

#include "Engine.h"

struct SimulationConfiguration
{
	uint32			seed = 0;
	float			time_dilation = 1.0f;
	float			agent_start_location = -1.0f;
};

struct SimulationGraphicsSettings
{
	uint32			is_fullscreen = 0;
	uint32			vsync_enabled = 0;
	
	uint32			resolution_width = 1920;
	uint32			resolution_height = 1080;
	float			resolution_scale = 1.0f; 

	uint8			texture_quality = 3;
	uint8			shadow_quality = 3;
	uint8			effect_quality = 3;
	uint8			post_process_level = 3;
	uint8			motion_blur_quality = 3;
	uint8			view_distance = 3;
	uint8			ambient_occlusion = 3;
};

struct SunSimulationSettings
{
	uint16			year = 2011;
	uint16			month = 8;
	uint16			day = 1;
	uint16			hour = 11;
	uint16			minute = 30;
};

