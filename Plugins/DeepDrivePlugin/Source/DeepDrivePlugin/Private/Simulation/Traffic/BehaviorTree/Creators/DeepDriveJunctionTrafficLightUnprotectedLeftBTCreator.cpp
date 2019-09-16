
#include "Simulation/Traffic/BehaviorTree/Creators/DeepDriveJunctionTrafficLightUnprotectedLeftBTCreator.h"

#include "Private/Simulation/Traffic/Path/DeepDrivePathDefines.h"

#include "Simulation/Traffic/BehaviorTree/DeepDriveTrafficBehaviorTree.h"
#include "Simulation/Traffic/BehaviorTree/DeepDriveTrafficBehaviorTreeNode.h"

#include "Simulation/Traffic/BehaviorTree/Tasks/DeepDriveTBTYieldTask.h"

#include "Simulation/Traffic/BehaviorTree/DeepDriveBehaviorTreeFactory.h"

bool DeepDriveJunctionTrafficLightUnprotectedLeftBTCreator::s_isRegistered = DeepDriveJunctionTrafficLightUnprotectedLeftBTCreator::registerCreator();

bool DeepDriveJunctionTrafficLightUnprotectedLeftBTCreator::registerCreator()
{
	DeepDriveBehaviorTreeFactory &factory = DeepDriveBehaviorTreeFactory::GetInstance();
	factory.registerCreator("four_way_tl_upl", std::bind(&DeepDriveJunctionTrafficLightUnprotectedLeftBTCreator::createBehaviorTree));
	return true;
}

DeepDriveTrafficBehaviorTree* DeepDriveJunctionTrafficLightUnprotectedLeftBTCreator::createBehaviorTree()
{
	DeepDriveTrafficBehaviorTree *behaviorTree = new DeepDriveTrafficBehaviorTree();
	if (behaviorTree)
	{
		DeepDriveTrafficBehaviorTreeNode *yieldNode = behaviorTree->createNode(0);

		yieldNode->addTask(new DeepDriveTBTYieldTask());
	}

	return behaviorTree;
}
