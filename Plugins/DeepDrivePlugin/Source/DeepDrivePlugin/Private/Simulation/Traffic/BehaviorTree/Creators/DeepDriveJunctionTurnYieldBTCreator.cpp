
#include "Simulation/Traffic/BehaviorTree/Creators/DeepDriveJunctionTurnYieldBTCreator.h"

#include "Private/Simulation/Traffic/Path/DeepDrivePathDefines.h"

#include "Simulation/Traffic/BehaviorTree/DeepDriveTrafficBehaviorTree.h"
#include "Simulation/Traffic/BehaviorTree/DeepDriveTrafficBehaviorTreeNode.h"

#include "Simulation/Traffic/BehaviorTree/Tasks/DeepDriveTBTYieldTask.h"

#include "Simulation/Traffic/BehaviorTree/DeepDriveBehaviorTreeFactory.h"

bool DeepDriveJunctionTurnYieldBTCreator::s_isRegistered = DeepDriveJunctionTurnYieldBTCreator::registerCreator();

bool DeepDriveJunctionTurnYieldBTCreator::registerCreator()
{
	DeepDriveBehaviorTreeFactory &factory = DeepDriveBehaviorTreeFactory::GetInstance();
	factory.registerCreator("four_way_yield", std::bind(&DeepDriveJunctionTurnYieldBTCreator::createBehaviorTree));
	return true;
}

DeepDriveTrafficBehaviorTree* DeepDriveJunctionTurnYieldBTCreator::createBehaviorTree()
{
	DeepDriveTrafficBehaviorTree *behaviorTree = new DeepDriveTrafficBehaviorTree();
	if (behaviorTree)
	{
		DeepDriveTrafficBehaviorTreeNode *yieldNode = behaviorTree->createNode(0);

		yieldNode->addTask(new DeepDriveTBTYieldTask());
	}

	return behaviorTree;
}
