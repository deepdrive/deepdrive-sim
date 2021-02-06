
#pragma once

#include "Server/Messages/DeepDriveServerMessageHeader.h"
#include "Simulation/DeepDriveSimulationTypes.h"

namespace deepdrive { namespace server {

struct GenericBooleanResponse : public MessageHeader
{
	GenericBooleanResponse(bool _result = false)
		: MessageHeader(MessageId::GenericBooleanResponse, sizeof(GenericBooleanResponse))
		, result(_result ? 1 : 0)
	{	}

	uint32		result;
};


struct ConfigureSimulationRequest : public MessageHeader
{
	ConfigureSimulationRequest()
		: MessageHeader(MessageId::ConfigureSimulationRequest, sizeof(ConfigureSimulationRequest))
	{	}

	SimulationConfiguration		configuration;

	SimulationGraphicsSettings	graphics_settings;
};

struct ConfigureSimulationResponse : public MessageHeader
{
	ConfigureSimulationResponse(bool _success = false)
		: MessageHeader(MessageId::ConfigureSimulationResponse, sizeof(ConfigureSimulationResponse))
		, success(_success ? 1 : 0)
	{	}

	uint32		success;
};

struct ResetSimulationRequest : public MessageHeader
{
	ResetSimulationRequest()
		: MessageHeader(MessageId::ResetSimulationRequest, sizeof(ResetSimulationRequest))
	{	}

	SimulationConfiguration		configuration;
};

struct ResetSimulationResponse : public MessageHeader
{
	ResetSimulationResponse(bool _result = false)
		: MessageHeader(MessageId::ResetSimulationResponse, sizeof(ResetSimulationResponse))
		, result(_result ? 1 : 0)
	{	}

	uint32		result;
};

struct SetGraphicsSettingsRequest : public MessageHeader
{
	SetGraphicsSettingsRequest()
		: MessageHeader(MessageId::SetGraphicsSettingsRequest, sizeof(SetGraphicsSettingsRequest))
	{	}

	SimulationGraphicsSettings	graphics_settings;
};

struct SetGraphicsSettingsResponse : public MessageHeader
{
	SetGraphicsSettingsResponse(bool _result = false)
		: MessageHeader(MessageId::SetGraphicsSettingsResponse, sizeof(SetGraphicsSettingsResponse))
		, result(_result ? 1 : 0)
	{	}

	uint32		result;
};


struct SetDateAndTimeRequest : public MessageHeader
{
	SetDateAndTimeRequest(uint32 _year, uint32 _month, uint32 _day, uint32 _hour, uint32 _minute)
		: MessageHeader(MessageId::SetDateAndTimeRequest, sizeof(SetDateAndTimeRequest))
		, year(_year)
		, month(_month)
		, day(_day)
		, hour(_hour)
		, minute(_minute)
	{
	}

	uint32			year;
	uint32			month;
	uint32			day;
	uint32			hour;
	uint32			minute;
};

struct SetDateAndTimeResponse : public MessageHeader
{
	SetDateAndTimeResponse(bool _result = false)
		: MessageHeader(MessageId::SetDateAndTimeResponse, sizeof(SetDateAndTimeResponse))
		, result(_result ? 1 : 0)
	{	}

	uint32		result;
};


struct SetSunSimulationSpeedRequest : public MessageHeader
{
	SetSunSimulationSpeedRequest(uint32 _speed)
		: MessageHeader(MessageId::SetSunSimulationSpeedRequest, sizeof(SetSunSimulationSpeedRequest))
		, speed(_speed)
	{
	}

	uint32			speed;
};


struct SetSunSimulationSpeedResponse : public MessageHeader
{
	SetSunSimulationSpeedResponse(bool _result = false)
		: MessageHeader(MessageId::SetSunSimulationSpeedResponse, sizeof(SetSunSimulationSpeedResponse))
		, result(_result ? 1 : 0)
	{	}

	uint32		result;
};

struct GetAgentsListRequest : public server::MessageHeader
{
	GetAgentsListRequest()
		: MessageHeader(MessageId::GetAgentsListRequest, sizeof(GetAgentsListRequest))
	{
	}
};

struct GetAgentsListResponse : public server::MessageHeader
{
	GetAgentsListResponse(uint32 agentCount = 0, const uint32 *agentIds = 0)
		: MessageHeader(MessageId::GetAgentsListResponse, sizeof(GetAgentsListResponse) + (agentCount > 0 ? (sizeof(uint32) * (agentCount - 1)) : 0) )
		, agent_count(agentCount)
	{
		if (agentIds)
		{
			for (uint32 i = 0; i < agentCount; ++i)
				agent_ids[i] = agentIds[i];
		}
	}

	uint32		agent_count;

	uint32		agent_ids[1];

	static size_t getMessageSize(uint32 agentCount)
	{
		return sizeof(GetAgentsListResponse) + (agentCount > 0 ? (sizeof(uint32) * (agentCount - 1)) : 0);
	}

};

struct RequestControlRequest : public MessageHeader
{
	RequestControlRequest(uint32 agentCount = 0, const uint32 *agentIds = 0)
		: MessageHeader(MessageId::RequestControlRequest, getMessageSize(agentCount))
		, agent_count(agentCount)
	{
		if (agentIds)
		{
			for (uint32 i = 0; i < agentCount; ++i)
				agent_ids[i] = agentIds[i];
		}
	}

	uint32		agent_count;

	uint32		agent_ids[1];

	static size_t getMessageSize(uint32 agentCount)
	{
		return sizeof(RequestControlRequest) + (agentCount > 0 ? (sizeof(uint32) * (agentCount - 1)) : 0);
	}
};

struct RequestControlResponse : public MessageHeader
{
	RequestControlResponse(bool _result = false)
		: MessageHeader(MessageId::RequestControlResponse, sizeof(RequestControlResponse))
		, result(_result ? 1 : 0)
	{
	}

	uint32		result;
};

struct ReleaseControlRequest : public MessageHeader
{
	ReleaseControlRequest(uint32 agentCount = 0, const uint32 *agentIds = 0)
		: MessageHeader(MessageId::ReleaseControlRequest, getMessageSize(agentCount))
		, agent_count(agentCount)
	{
		if (agentIds)
		{
			for (uint32 i = 0; i < agentCount; ++i)
				agent_ids[i] = agentIds[i];
		}
	}

	uint32 agent_count;

	uint32 agent_ids[1];

	static size_t getMessageSize(uint32 agentCount)
	{
		return sizeof(ReleaseControlRequest) + (agentCount > 0 ? (sizeof(uint32) * (agentCount - 1)) : 0);
	}
};

struct ReleaseControlResponse : public MessageHeader
{
	ReleaseControlResponse(bool _result = false)
		: MessageHeader(MessageId::ReleaseControlResponse, sizeof(ReleaseControlResponse)), result(_result ? 1 : 0)
	{
	}

	uint32 result;
};

struct SetControlValuesRequest : public MessageHeader
{
	struct ControlValueSet
	{
		ControlValueSet(uint32 id = 0, float s = 0.0f, float t = 0.0f, float b = 0.0f, uint32 hb = 0)
		:	agent_id(id)
		,	steering(s)
		,	throttle(t)
		,	brake(b)
		,	handbrake(hb)
		{

		}

		uint32		agent_id;
		float		steering;
		float		throttle;
		float		brake;
		uint32		handbrake;
	};

	SetControlValuesRequest(uint32 agentCount = 0, const ControlValueSet *controlValues = 0)
		: MessageHeader(MessageId::SetControlValuesRequest, getMessageSize(agentCount))
		, agent_count(agentCount)
	{
		if (controlValues)
		{
			for (uint32 i = 0; i < agentCount; ++i)
				control_values[i] = controlValues[i];
		}
	}

	uint32 				agent_count;

	ControlValueSet		control_values[1];

	static size_t getMessageSize(uint32 agentCount)
	{
		return sizeof(SetControlValuesRequest) + (agentCount > 0 ? (sizeof(ControlValueSet) * (agentCount - 1)) : 0);
	}
};

struct StepRequest : public MessageHeader
{
	StepRequest()
		: MessageHeader(MessageId::StepRequest, sizeof(StepRequest))
	{
	}

};

struct StepResponse : public MessageHeader
{
	struct SnapshotData
	{
		SnapshotData(uint32 id = 0)
		:	agent_id(id)
		{
		}

		uint32	agent_id;
		double	snapshot_timestamp;
		float	speed;

		float	steering;
		float	throttle;
		float	brake;
		uint32	handbrake;
		float	position[3];
		float	rotation[3];
		float	velocity[3];
		float	acceleration[3];
		float	angular_velocity[3];
		float	angular_acceleration[3];
		float	forward_vector[3];
		float	up_vector[3];
		float	right_vector[3];
		float	dimension[3];
		float	distance_along_route;
		float	route_length;
		float	distance_to_center_of_lane;
	};

	StepResponse(uint32 agentCount = 0, const SnapshotData *_snapshots = 0)
		: MessageHeader(MessageId::StepResponse, getMessageSize(agentCount))
		, agent_count(agentCount)
	{
		if (_snapshots)
		{
			for (uint32 i = 0; i < agentCount; ++i)
				snapshots[i] = _snapshots[i];
		}
	}

	uint32 				agent_count;
	SnapshotData 		snapshots[1];

	static size_t getMessageSize(uint32 agentCount)
	{
		return sizeof(StepResponse) + (agentCount > 0 ? (sizeof(SnapshotData) * (agentCount - 1)) : 0);
	}
};


} }	// deepdrive
