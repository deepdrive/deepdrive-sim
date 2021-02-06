
#include "Simulation/Traffic/BehaviorTree/Creators/DeepDriveJunctionTrafficLightRightStraightBTCreator.h"

#include "Simulation/Traffic/Path/DeepDrivePathDefines.h"

#include "Simulation/Traffic/BehaviorTree/DeepDriveTrafficBehaviorTree.h"
#include "Simulation/Traffic/BehaviorTree/DeepDriveTrafficBehaviorTreeNode.h"

#include "Simulation/Traffic/BehaviorTree/DeepDriveBehaviorTreeFactory.h"

#include "Simulation/Traffic/BehaviorTree/Tasks/DeepDriveTBTStopAtLocationTask.h"
#include "Simulation/Traffic/BehaviorTree/Tasks/DeepDriveTBTHasPassedLocationTask.h"
#include "Simulation/Traffic/BehaviorTree/Tasks/DeepDriveTBTCheckGreenToRedTask.h"

#include "Simulation/Traffic/BehaviorTree/Decorators/DeepDriveTBTCheckTrafficLightDecorator.h"
#include "Simulation/Traffic/BehaviorTree/Decorators/DeepDriveTBTCheckFlagDecorator.h"
#include "Simulation/Traffic/BehaviorTree/Decorators/DeepDriveTBTCheckIntDecorator.h"

bool DeepDriveJunctionTrafficLightRightStraightBTCreator::s_isRegistered = DeepDriveJunctionTrafficLightRightStraightBTCreator::registerCreator();

bool DeepDriveJunctionTrafficLightRightStraightBTCreator::registerCreator()
{
	DeepDriveBehaviorTreeFactory &factory = DeepDriveBehaviorTreeFactory::GetInstance();
	factory.registerCreator("four_way_tl_rs", std::bind(&DeepDriveJunctionTrafficLightRightStraightBTCreator::createBehaviorTree));
	return true;
}

DeepDriveTrafficBehaviorTree* DeepDriveJunctionTrafficLightRightStraightBTCreator::createBehaviorTree()
{
	DeepDriveTrafficBehaviorTree *behaviorTree = new DeepDriveTrafficBehaviorTree("TL_StraightOrRight");
	if (behaviorTree)
	{
		DeepDriveTrafficBehaviorTreeNode *greenNode = behaviorTree->createNode(0);
		DeepDriveTrafficBehaviorTreeNode *redNode = behaviorTree->createNode(0);
		DeepDriveTrafficBehaviorTreeNode *greenToRedMainNode = behaviorTree->createNode(0);

		behaviorTree->getRootNode()->addDecorator( new DeepDriveTBTCheckFlagDecorator("HasEnteredJunction", false, false) );

		TSharedPtr<DeepDriveTBTTaskBase> hasPassedTask(new DeepDriveTBTHasPassedLocationTask("StopLineLocation", 200.0f, "HasEnteredJunction"));
		greenNode->addDecorator(new DeepDriveTBTCheckTrafficLightDecorator(EDeepDriveTrafficLightPhase::GREEN, EDeepDriveTrafficLightPhase::RED_TO_GREEN));
		greenNode->addTask(hasPassedTask);

		TSharedPtr<DeepDriveTBTTaskBase> stopTask(new DeepDriveTBTStopAtLocationTask("StopLineLocation", 0.6f));
		redNode->addDecorator(new DeepDriveTBTCheckTrafficLightDecorator(EDeepDriveTrafficLightPhase::RED));
 		redNode->addTask(stopTask);
		// redNode->addTask(new DeepDriveTBTSetFlagTask("GreenToRedStatus", -1));

		greenToRedMainNode->addDecorator(new DeepDriveTBTCheckTrafficLightDecorator(EDeepDriveTrafficLightPhase::GREEN_TO_RED));

		DeepDriveTrafficBehaviorTreeNode *greenToRedCheckNode = behaviorTree->createNode(greenToRedMainNode);
		DeepDriveTrafficBehaviorTreeNode *greenToRedStopNode = behaviorTree->createNode(greenToRedMainNode);
		DeepDriveTrafficBehaviorTreeNode *greenToRedGoNode = behaviorTree->createNode(greenToRedMainNode);

 		greenToRedCheckNode->addTask(new DeepDriveTBTCheckGreenToRedTask("StopLineLocation", "GreenToRedStatus"));

		greenToRedStopNode->addDecorator(new DeepDriveTBTCheckIntDecorator("GreenToRedStatus", 0, -1));
		greenToRedStopNode->addTask(stopTask);

		greenToRedGoNode->addDecorator(new DeepDriveTBTCheckIntDecorator("GreenToRedStatus", 1, -1));
		greenToRedGoNode->addTask(hasPassedTask);
	}

	return behaviorTree;
}
