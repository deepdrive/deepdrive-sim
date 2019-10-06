
#include "Simulation/Traffic/BehaviorTree/Creators/DeepDriveJunctionTurnGiveWayStrictBTCreator.h"

#include "Private/Simulation/Traffic/Path/DeepDrivePathDefines.h"

#include "Simulation/Traffic/BehaviorTree/DeepDriveTrafficBehaviorTree.h"
#include "Simulation/Traffic/BehaviorTree/DeepDriveTrafficBehaviorTreeNode.h"
#include "Simulation/Traffic/BehaviorTree/DeepDriveTrafficBlackboard.h"

#include "Simulation/Traffic/BehaviorTree/Tasks/DeepDriveTBTStopAtLocationTask.h"
#include "Simulation/Traffic/BehaviorTree/Tasks/DeepDriveTBTWaitTask.h"
#include "Simulation/Traffic/BehaviorTree/Tasks/DeepDriveTBTStopTask.h"

#include "Simulation/Traffic/BehaviorTree/Decorators/DeepDriveTBTIsJunctionClearDecorator.h"

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
	DeepDriveTrafficBehaviorTree *behaviorTree = new DeepDriveTrafficBehaviorTree();
	if (behaviorTree)
	{
		DeepDriveTrafficBehaviorTreeNode *stopAtNode = behaviorTree->createNode(0);
		DeepDriveTrafficBehaviorTreeNode *waitNode = behaviorTree->createNode(0);
		DeepDriveTrafficBehaviorTreeNode *checkJunctionClearNode = behaviorTree->createNode(0);

		stopAtNode->addTask(new DeepDriveTBTStopAtLocationTask("StopLineLocation", 0.6f, false));
		waitNode->addTask(new DeepDriveTBTWaitTask(3.0f));
		checkJunctionClearNode->addDecorator(new DeepDriveTBTIsJunctionClearDecorator(true));
		checkJunctionClearNode->addTask(new DeepDriveTBTStopTask);
	}

	return behaviorTree;
}
