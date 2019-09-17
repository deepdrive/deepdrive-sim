
#include "Simulation/Traffic/BehaviorTree/Creators/DeepDriveJunctionRightOfWayBTCreator.h"

#include "Private/Simulation/Traffic/Path/DeepDrivePathDefines.h"

#include "Simulation/Traffic/BehaviorTree/DeepDriveTrafficBehaviorTree.h"
#include "Simulation/Traffic/BehaviorTree/DeepDriveTrafficBehaviorTreeNode.h"

#include "Simulation/Traffic/BehaviorTree/Tasks/DeepDriveTBTWaitForOncomingTrafficTask.h"

#include "Simulation/Traffic/BehaviorTree/DeepDriveBehaviorTreeFactory.h"

bool DeepDriveJunctionRightOfWayBTCreator::s_isRegistered = DeepDriveJunctionRightOfWayBTCreator::registerCreator();

bool DeepDriveJunctionRightOfWayBTCreator::registerCreator()
{
	DeepDriveBehaviorTreeFactory &factory = DeepDriveBehaviorTreeFactory::GetInstance();
	factory.registerCreator("four_way_row", std::bind(&DeepDriveJunctionRightOfWayBTCreator::createBehaviorTree));
	return true;
}

DeepDriveTrafficBehaviorTree* DeepDriveJunctionRightOfWayBTCreator::createBehaviorTree()
{
	DeepDriveTrafficBehaviorTree *behaviorTree = new DeepDriveTrafficBehaviorTree();
	if (behaviorTree)
	{
		DeepDriveTrafficBehaviorTreeNode *node = behaviorTree->createNode(0);

		node->addTask(new DeepDriveTBTWaitForOncomingTrafficTask());
	}

	return behaviorTree;
}
