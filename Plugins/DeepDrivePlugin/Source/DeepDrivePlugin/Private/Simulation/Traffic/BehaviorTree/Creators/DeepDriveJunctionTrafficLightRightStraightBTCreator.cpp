
#include "Simulation/Traffic/BehaviorTree/Creators/DeepDriveJunctionTrafficLightRightStraightBTCreator.h"

#include "Private/Simulation/Traffic/Path/DeepDrivePathDefines.h"

#include "Simulation/Traffic/BehaviorTree/DeepDriveTrafficBehaviorTree.h"
#include "Simulation/Traffic/BehaviorTree/DeepDriveTrafficBehaviorTreeNode.h"

#include "Simulation/Traffic/BehaviorTree/DeepDriveBehaviorTreeFactory.h"

#include "Simulation/Traffic/BehaviorTree/Tasks/DeepDriveTBTStopAtLocationTask.h"

#include "Simulation/Traffic/BehaviorTree/Decorators/DeepDriveTBTCheckTrafficLightDecorator.h"
#include "Simulation/Traffic/BehaviorTree/Decorators/DeepDriveTBTGreenToRedDecorator.h"
#include "Simulation/Traffic/BehaviorTree/Decorators/DeepDriveTBTCheckRedDecorator.h"

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
		DeepDriveTrafficBehaviorTreeNode *mainNode = behaviorTree->createNode(0);

		DeepDriveTrafficBehaviorTreeNode *redNode = behaviorTree->createNode(mainNode);
		// DeepDriveTrafficBehaviorTreeNode *greenNode = behaviorTree->createNode(mainNode);
		DeepDriveTrafficBehaviorTreeNode *greenToRedNode = behaviorTree->createNode(mainNode);
		// DeepDriveTrafficBehaviorTreeNode *redToGreenNode = behaviorTree->createNode(mainNode);

		redNode->addDecorator(new DeepDriveTBTCheckRedDecorator);
		redNode->addTask(new DeepDriveTBTStopAtLocationTask("StopLineLocation", 0.6f, true));

		greenToRedNode->addDecorator(new DeepDriveTBTGreenToRedDecorator);
		greenToRedNode->addTask(new DeepDriveTBTStopAtLocationTask("StopLineLocation", 0.6f, true));
	}

	return behaviorTree;
}
