
#include "Simulation/Traffic/BehaviorTree/Creators/DeepDriveJunctionTurnGiveWayStrictBTCreator.h"

#include "Private/Simulation/Traffic/Path/DeepDrivePathDefines.h"

#include "Simulation/Traffic/BehaviorTree/DeepDriveTrafficBehaviorTree.h"
#include "Simulation/Traffic/BehaviorTree/DeepDriveTrafficBehaviorTreeNode.h"
#include "Simulation/Traffic/BehaviorTree/DeepDriveTrafficBlackboard.h"

#include "Simulation/Traffic/BehaviorTree/Tasks/DeepDriveTBTStopAtLocationTask.h"
#include "Simulation/Traffic/BehaviorTree/Tasks/DeepDriveTBTWaitTask.h"
#include "Simulation/Traffic/BehaviorTree/Tasks/DeepDriveTBTWaitForClearance.h"

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
		DeepDriveTrafficBehaviorTreeNode *clearanceNode = behaviorTree->createNode(0);

		stopAtNode->addTask(new DeepDriveTBTStopAtLocationTask());
		waitNode->addTask(new DeepDriveTBTWaitTask(3.0f));
		clearanceNode->addTask(new DeepDriveTBTWaitForClearance());
	}

	return behaviorTree;
}
