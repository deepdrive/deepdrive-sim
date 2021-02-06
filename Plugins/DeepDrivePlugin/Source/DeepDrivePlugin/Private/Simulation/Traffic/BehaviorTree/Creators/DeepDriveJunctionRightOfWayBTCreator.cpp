
#include "Simulation/Traffic/BehaviorTree/Creators/DeepDriveJunctionRightOfWayBTCreator.h"

#include "Simulation/Traffic/Path/DeepDrivePathDefines.h"

#include "Simulation/Traffic/BehaviorTree/DeepDriveTrafficBehaviorTree.h"
#include "Simulation/Traffic/BehaviorTree/DeepDriveTrafficBehaviorTreeNode.h"

#include "Simulation/Traffic/BehaviorTree/Tasks/DeepDriveTBTStopAtLocationTask.h"
#include "Simulation/Traffic/BehaviorTree/Tasks/DeepDriveTBTWaitTask.h"
#include "Simulation/Traffic/BehaviorTree/Tasks/DeepDriveTBTStopTask.h"
#include "Simulation/Traffic/BehaviorTree/Tasks/DeepDriveTBTCheckIsJunctionClearTask.h"

#include "Simulation/Traffic/BehaviorTree/Decorators/DeepDriveTBTCheckFlagDecorator.h"

#include "Simulation/Traffic/BehaviorTree/DeepDriveBehaviorTreeFactory.h"

bool DeepDriveJunctionRightOfWayBTCreator::s_isRegistered = DeepDriveJunctionRightOfWayBTCreator::registerCreator();

bool DeepDriveJunctionRightOfWayBTCreator::registerCreator()
{
	DeepDriveBehaviorTreeFactory &factory = DeepDriveBehaviorTreeFactory::GetInstance();
	factory.registerCreator("four_way_row", &DeepDriveJunctionRightOfWayBTCreator::createBehaviorTree);
	return true;
}

DeepDriveTrafficBehaviorTree* DeepDriveJunctionRightOfWayBTCreator::createBehaviorTree()
{
	DeepDriveTrafficBehaviorTree *behaviorTree = new DeepDriveTrafficBehaviorTree("RightOfWay");
	if (behaviorTree)
	{
		DeepDriveTrafficBehaviorTreeNode *stopAtNode = behaviorTree->createNode(0);
		DeepDriveTrafficBehaviorTreeNode *waitUntilClearNode = behaviorTree->createNode(0);

		behaviorTree->getRootNode()->addDecorator(new DeepDriveTBTCheckFlagDecorator("IsJunctionClear", false, false));

		TSharedPtr<DeepDriveTBTTaskBase> checkJunctionTask(new DeepDriveTBTCheckIsJunctionClearTask("WaitingLocation", 500.0f, true, "IsJunctionClear"));

		stopAtNode->addTask(checkJunctionTask);
		stopAtNode->addTask(new DeepDriveTBTStopAtLocationTask("WaitingLocation", 0.6f));

		waitUntilClearNode->addTask(new DeepDriveTBTStopTask);
		waitUntilClearNode->addTask(checkJunctionTask);
	}

	return behaviorTree;
}
