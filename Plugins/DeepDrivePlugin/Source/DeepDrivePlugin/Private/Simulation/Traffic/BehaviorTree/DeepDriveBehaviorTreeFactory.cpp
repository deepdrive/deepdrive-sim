
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
	if(m_Creators.Contains(key))
		m_Creators[key] = creator;
	else
		m_Creators.Add(key, creator);
}

DeepDriveTrafficBehaviorTree* DeepDriveBehaviorTreeFactory::createBehaviorTree(const FString &key)
{
	DeepDriveTrafficBehaviorTree *behaviorTree = 0;
	if(m_Creators.Contains(key))
	{
		behaviorTree = m_Creators[key]();
	}

	return behaviorTree;
}

DeepDriveBehaviorTreeFactory& DeepDriveBehaviorTreeFactory::GetInstance()
{
	if (theInstance == 0)
	{
		theInstance = new DeepDriveBehaviorTreeFactory;
	}

	return *theInstance;
}

void DeepDriveBehaviorTreeFactory::Destroy()
{
	delete theInstance;
	theInstance = 0;
}
