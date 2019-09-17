
#pragma once

#include "Simulation/Traffic/Maneuver/DeepDriveJunctionCalculatorBase.h"

DECLARE_LOG_CATEGORY_EXTERN(LogDeepDriveFourWayJunctionCalculator, Log, All);

class DeepDriveFourWayJunctionCalculator	:	public DeepDriveJunctionCalculatorBase
{
protected:

	struct SCrossTrafficRoadMask
	{
		SCrossTrafficRoadMask(uint16 r, uint16 s, uint16 l)
			:	turn_right(r)
			,	go_on_straight(s)
			,	turn_left(l)
		{
		}

		uint16		turn_right;
		uint16		go_on_straight;
		uint16		turn_left;
	};

	typedef TMap<uint32, FString>	TBehaviorTreeIdMap;					//	mapping of junction sub type to behavior tree id
	typedef TMap<uint32, SCrossTrafficRoadMask>	TCrossTrafficRoadMasks;	//	mapping of junction sub type to cross traffix road mask

public:

	DeepDriveFourWayJunctionCalculator(const SDeepDriveRoadNetwork &roadNetwork);

	virtual void calculate(SDeepDriveManeuver &maneuver) override;

protected:

	bool createBehaviorTree(SDeepDriveManeuver &maneuver);

	void extractCrossRoadTraffic(uint32 curEntryLinkId, SDeepDriveManeuver &maneuver, const SDeepDriveJunction &junction, const TCrossTrafficRoadMasks &masks);

	uint32 calcBehaviorTreeKey(const SDeepDriveManeuver &maneuver) const;
	uint32 calcBehaviorTreeKey(uint32 junctionSubType, uint32 rightOfWay) const;

	void loadConfiguration(const FString &configFile);

	TBehaviorTreeIdMap						m_BehaviorTreeIds;
	TCrossTrafficRoadMasks					m_CrossRoadRightMasks;
	TCrossTrafficRoadMasks					m_CrossRoadStraightMasks;
	TCrossTrafficRoadMasks					m_CrossRoadLeftMasks;

};
