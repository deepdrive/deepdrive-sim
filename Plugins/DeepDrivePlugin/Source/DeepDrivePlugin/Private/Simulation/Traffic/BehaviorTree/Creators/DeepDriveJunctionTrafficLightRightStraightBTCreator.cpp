
#include "Simulation/Traffic/BehaviorTree/Creators/DeepDriveJunctionTrafficLightRightStraightBTCreator.h"

#include "Private/Simulation/Traffic/Path/DeepDrivePathDefines.h"

#include "Simulation/Traffic/BehaviorTree/DeepDriveTrafficBehaviorTree.h"
#include "Simulation/Traffic/BehaviorTree/DeepDriveTrafficBehaviorTreeNode.h"

#include "Simulation/Traffic/BehaviorTree/Tasks/DeepDriveTBTYieldTask.h"

#include "Simulation/Traffic/BehaviorTree/DeepDriveBehaviorTreeFactory.h"

bool DeepDriveJunctionTrafficLightRightStraightBTCreator::s_isRegistered = DeepDriveJunctionTrafficLightRightStraightBTCreator::registerCreator();

bool DeepDriveJunctionTrafficLightRightStraightBTCreator::registerCreator()
{
	DeepDriveBehaviorTreeFactory &factory = DeepDriveBehaviorTreeFactory::GetInstance();
	factory.registerCreator("four_way_tl_rs", std::bind(&DeepDriveJunctionTrafficLightRightStraightBTCreator::createBehaviorTree));
	return true;
}

DeepDriveTrafficBehaviorTree* DeepDriveJunctionTrafficLightRightStraightBTCreator::createBehaviorTree()
{
	DeepDriveTrafficBehaviorTree *behaviorTree = new DeepDriveTrafficBehaviorTree();
	if (behaviorTree)
	{
		DeepDriveTrafficBehaviorTreeNode *yieldNode = behaviorTree->createNode(0);

		yieldNode->addTask(new DeepDriveTBTYieldTask());
	}

	return behaviorTree;
}
