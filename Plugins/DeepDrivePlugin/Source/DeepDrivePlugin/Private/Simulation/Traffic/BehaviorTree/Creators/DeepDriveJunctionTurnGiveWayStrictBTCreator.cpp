
#include "Simulation/Traffic/BehaviorTree/Creators/DeepDriveJunctionTurnGiveWayStrictBTCreator.h"

#include "Simulation/Traffic/Path/DeepDrivePathDefines.h"

#include "Simulation/Traffic/BehaviorTree/DeepDriveTrafficBehaviorTree.h"
#include "Simulation/Traffic/BehaviorTree/DeepDriveTrafficBehaviorTreeNode.h"
#include "Simulation/Traffic/BehaviorTree/DeepDriveTrafficBlackboard.h"

#include "Simulation/Traffic/BehaviorTree/Tasks/DeepDriveTBTStopAtLocationTask.h"
#include "Simulation/Traffic/BehaviorTree/Tasks/DeepDriveTBTWaitTask.h"
#include "Simulation/Traffic/BehaviorTree/Tasks/DeepDriveTBTStopTask.h"
#include "Simulation/Traffic/BehaviorTree/Tasks/DeepDriveTBTCheckIsJunctionClearTask.h"

#include "Simulation/Traffic/BehaviorTree/Decorators/DeepDriveTBTCheckFlagDecorator.h"

#include "Simulation/Traffic/BehaviorTree/DeepDriveBehaviorTreeFactory.h"

bool DeepDriveJunctionTurnGiveWayStrictBTCreator::s_isRegistered = DeepDriveJunctionTurnGiveWayStrictBTCreator::registerCreator();

bool DeepDriveJunctionTurnGiveWayStrictBTCreator::registerCreator()
{
	DeepDriveBehaviorTreeFactory &factory = DeepDriveBehaviorTreeFactory::GetInstance();
	factory.registerCreator("four_way_stop", &DeepDriveJunctionTurnGiveWayStrictBTCreator::createBehaviorTree);
	return true;
}

DeepDriveTrafficBehaviorTree* DeepDriveJunctionTurnGiveWayStrictBTCreator::createBehaviorTree()
{
	DeepDriveTrafficBehaviorTree *behaviorTree = new DeepDriveTrafficBehaviorTree("GiveWayStrict");
	if (behaviorTree)
	{
		DeepDriveTrafficBehaviorTreeNode *stopAtNode = behaviorTree->createNode(0);
		DeepDriveTrafficBehaviorTreeNode *waitNode = behaviorTree->createNode(0);
		DeepDriveTrafficBehaviorTreeNode *waitUntilClearNode = behaviorTree->createNode(0);

		behaviorTree->getRootNode()->addDecorator( new DeepDriveTBTCheckFlagDecorator("IsJunctionClear", false, false) );

		stopAtNode->addTask(new DeepDriveTBTStopAtLocationTask("StopLineLocation", 0.6f));

		waitNode->addTask(new DeepDriveTBTStopTask);
		waitNode->addTask(new DeepDriveTBTWaitTask(3.0f));
		
		waitUntilClearNode->addTask(new DeepDriveTBTStopTask);
		waitUntilClearNode->addTask(new DeepDriveTBTCheckIsJunctionClearTask("StopLineLocation", 1500.0f, true, "IsJunctionClear") );
	}

	return behaviorTree;
}
