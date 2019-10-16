
#include "Simulation/Traffic/BehaviorTree/Creators/DeepDriveJunctionTrafficLightUnprotectedLeftBTCreator.h"

#include "Private/Simulation/Traffic/Path/DeepDrivePathDefines.h"

#include "Simulation/Traffic/BehaviorTree/DeepDriveTrafficBehaviorTree.h"
#include "Simulation/Traffic/BehaviorTree/DeepDriveTrafficBehaviorTreeNode.h"

#include "Simulation/Traffic/BehaviorTree/DeepDriveBehaviorTreeFactory.h"

#include "Simulation/Traffic/BehaviorTree/Tasks/DeepDriveTBTStopAtLocationTask.h"
#include "Simulation/Traffic/BehaviorTree/Tasks/DeepDriveTBTHasPassedLocationTask.h"
#include "Simulation/Traffic/BehaviorTree/Tasks/DeepDriveTBTCheckGreenToRedTask.h"
#include "Simulation/Traffic/BehaviorTree/Tasks/DeepDriveTBTCheckIsJunctionClearTask.h"
#include "Simulation/Traffic/BehaviorTree/Tasks/DeepDriveTBTStopTask.h"

#include "Simulation/Traffic/BehaviorTree/Decorators/DeepDriveTBTCheckTrafficLightDecorator.h"
#include "Simulation/Traffic/BehaviorTree/Decorators/DeepDriveTBTCheckFlagDecorator.h"
#include "Simulation/Traffic/BehaviorTree/Decorators/DeepDriveTBTCheckIntDecorator.h"

bool DeepDriveJunctionTrafficLightUnprotectedLeftBTCreator::s_isRegistered = DeepDriveJunctionTrafficLightUnprotectedLeftBTCreator::registerCreator();

bool DeepDriveJunctionTrafficLightUnprotectedLeftBTCreator::registerCreator()
{
	DeepDriveBehaviorTreeFactory &factory = DeepDriveBehaviorTreeFactory::GetInstance();
	factory.registerCreator("four_way_tl_upl", std::bind(&DeepDriveJunctionTrafficLightUnprotectedLeftBTCreator::createBehaviorTree));
	return true;
}

DeepDriveTrafficBehaviorTree* DeepDriveJunctionTrafficLightUnprotectedLeftBTCreator::createBehaviorTree()
{
	DeepDriveTrafficBehaviorTree *behaviorTree = new DeepDriveTrafficBehaviorTree("TL_Left");
	if (behaviorTree)
	{
		// behaviorTree->activateDebugging();

		DeepDriveTrafficBehaviorTreeNode *notYetEnteredNode = behaviorTree->createNode(0, "NotYetEntered");
		{
			notYetEnteredNode->addDecorator(new DeepDriveTBTCheckFlagDecorator("HasEnteredJunction", false, false));

			DeepDriveTrafficBehaviorTreeNode *greenNode = behaviorTree->createNode(notYetEnteredNode, "GreenNode");
			DeepDriveTrafficBehaviorTreeNode *redNode = behaviorTree->createNode(notYetEnteredNode, "RedNode");
			DeepDriveTrafficBehaviorTreeNode *greenToRedMainNode = behaviorTree->createNode(notYetEnteredNode, "GreenToRed");

			TSharedPtr<DeepDriveTBTTaskBase> hasPassedTask(new DeepDriveTBTHasPassedLocationTask("StopLineLocation", 200.0f, "HasEnteredJunction"));
			greenNode->addDecorator(new DeepDriveTBTCheckTrafficLightDecorator(EDeepDriveTrafficLightPhase::GREEN, EDeepDriveTrafficLightPhase::RED_TO_GREEN));
			greenNode->addTask(hasPassedTask);

			TSharedPtr<DeepDriveTBTTaskBase> stopTask(new DeepDriveTBTStopAtLocationTask("StopLineLocation", 0.6f));
			redNode->addDecorator(new DeepDriveTBTCheckTrafficLightDecorator(EDeepDriveTrafficLightPhase::RED));
			redNode->addTask(stopTask);
			// redNode->addTask(new DeepDriveTBTSetFlagTask("GreenToRedStatus", -1));

			greenToRedMainNode->addDecorator(new DeepDriveTBTCheckTrafficLightDecorator(EDeepDriveTrafficLightPhase::GREEN_TO_RED));

			DeepDriveTrafficBehaviorTreeNode *greenToRedCheckNode = behaviorTree->createNode(greenToRedMainNode, "GreenToRedCheck");
			DeepDriveTrafficBehaviorTreeNode *greenToRedStopNode = behaviorTree->createNode(greenToRedMainNode, "GreenToRedStop");
			DeepDriveTrafficBehaviorTreeNode *greenToRedGoNode = behaviorTree->createNode(greenToRedMainNode, "GreenToRedGo");

			greenToRedCheckNode->addTask(new DeepDriveTBTCheckGreenToRedTask("StopLineLocation", "GreenToRedStatus"));

			greenToRedStopNode->addDecorator(new DeepDriveTBTCheckIntDecorator("GreenToRedStatus", 0, -1));
			greenToRedStopNode->addTask(stopTask);

			greenToRedGoNode->addDecorator(new DeepDriveTBTCheckIntDecorator("GreenToRedStatus", 1, -1));
			greenToRedGoNode->addTask(hasPassedTask);
		}

		DeepDriveTrafficBehaviorTreeNode *enteredNode = behaviorTree->createNode(0, "Entered");
		{
			enteredNode->addDecorator(new DeepDriveTBTCheckFlagDecorator("HasEnteredJunction", true, false));
			enteredNode->addDecorator(new DeepDriveTBTCheckFlagDecorator("IsJunctionClear", false, false));

			DeepDriveTrafficBehaviorTreeNode *stopAtNode = behaviorTree->createNode(enteredNode, "StopAt");
			DeepDriveTrafficBehaviorTreeNode *waitUntilClearNode = behaviorTree->createNode(enteredNode, "WaitUntilClear");

			TSharedPtr<DeepDriveTBTTaskBase> checkJunctionTask(new DeepDriveTBTCheckIsJunctionClearTask("WaitingLocation", 500.0f, false, "IsJunctionClear"));

			stopAtNode->addTask(checkJunctionTask);
			stopAtNode->addTask(new DeepDriveTBTStopAtLocationTask("WaitingLocation", 0.6f));

			waitUntilClearNode->addTask(new DeepDriveTBTStopTask);
			waitUntilClearNode->addTask(checkJunctionTask);
		}

	}

	return behaviorTree;
}
