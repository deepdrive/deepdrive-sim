
#include "Simulation/Traffic/BehaviorTree/DeepDriveBehaviorTreeFactory.h"
#include "Private/Simulation/Traffic/Path/DeepDrivePathDefines.h"

#include "Simulation/Traffic/BehaviorTree/DeepDriveTrafficBehaviorTree.h"
#include "Simulation/Traffic/BehaviorTree/DeepDriveTrafficBehaviorTreeNode.h"
#include "Simulation/Traffic/BehaviorTree/DeepDriveTrafficBlackboard.h"

DeepDriveBehaviorTreeFactory *DeepDriveBehaviorTreeFactory::theInstance = 0;

DeepDriveBehaviorTreeFactory::DeepDriveBehaviorTreeFactory()
{
	// std::bind(&DeepDriveServer::handleRegisterCamera, this, std::placeholders::_1);
}

void DeepDriveBehaviorTreeFactory::registerCreator(const FString &key, CreatorFuncPtr creator)
{
	m_Creators[key] = creator;
}

DeepDriveTrafficBehaviorTree* DeepDriveBehaviorTreeFactory::createBehaviorTree(const FString &key)
{
	DeepDriveTrafficBehaviorTree *behaviorTree = 0;

	Creators::iterator cIt = m_Creators.find(key);
	if(cIt != m_Creators.end())
	{
		behaviorTree = cIt->second();
	}

	return behaviorTree;
}

DeepDriveBehaviorTreeFactory& DeepDriveBehaviorTreeFactory::GetInstance()
{
	if (theInstance == 0)
	{
		theInstance = new DeepDriveBehaviorTreeFactory();
	}

	return *theInstance;
}

void DeepDriveBehaviorTreeFactory::Destroy()
{
	delete theInstance;
	theInstance = 0;
}
