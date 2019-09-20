
#include "Simulation/Traffic/BehaviorTree/Creators/DeepDriveJunctionTrafficLightUnprotectedLeftBTCreator.h"

#include "Private/Simulation/Traffic/Path/DeepDrivePathDefines.h"

#include "Simulation/Traffic/BehaviorTree/DeepDriveTrafficBehaviorTree.h"
#include "Simulation/Traffic/BehaviorTree/DeepDriveTrafficBehaviorTreeNode.h"

#include "Simulation/Traffic/BehaviorTree/DeepDriveBehaviorTreeFactory.h"

#include "Simulation/Traffic/BehaviorTree/Tasks/DeepDriveTBTStopAtLocationTask.h"

#include "Simulation/Traffic/BehaviorTree/Decorators/DeepDriveTBTCheckTrafficLightDecorator.h"
#include "Simulation/Traffic/BehaviorTree/Decorators/DeepDriveTBTGreenToRedDecorator.h"
#include "Simulation/Traffic/BehaviorTree/Decorators/DeepDriveTBTCheckRedDecorator.h"
#include "Simulation/Traffic/BehaviorTree/Decorators/DeepDriveTBTCheckOncomingTrafficDecorator.h"


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
		DeepDriveTrafficBehaviorTreeNode *mainNode = behaviorTree->createNode(0);

		DeepDriveTrafficBehaviorTreeNode *redNode = behaviorTree->createNode(mainNode);
		DeepDriveTrafficBehaviorTreeNode *greenToRedNode = behaviorTree->createNode(mainNode);
		DeepDriveTrafficBehaviorTreeNode *greenNode = behaviorTree->createNode(mainNode);
		DeepDriveTrafficBehaviorTreeNode *greenTrafficNode = behaviorTree->createNode(greenNode);

		redNode->addDecorator(new DeepDriveTBTCheckRedDecorator);
		redNode->addTask(new DeepDriveTBTStopAtLocationTask("StopLineLocation", 0.6f, true));

		greenToRedNode->addDecorator(new DeepDriveTBTGreenToRedDecorator);
		greenToRedNode->addTask(new DeepDriveTBTStopAtLocationTask("StopLineLocation", 0.6f, true));

		greenNode->addDecorator(new DeepDriveTBTCheckTrafficLightDecorator(EDeepDriveTrafficLightPhase::GREEN, EDeepDriveTrafficLightPhase::RED_TO_GREEN));

		greenTrafficNode->addDecorator( new DeepDriveTBTCheckOncomingTrafficDecorator("WaitingLocation") );
		greenTrafficNode->addTask(new DeepDriveTBTStopAtLocationTask("WaitingLocation", 0.25f, true));
	}

	return behaviorTree;
}
