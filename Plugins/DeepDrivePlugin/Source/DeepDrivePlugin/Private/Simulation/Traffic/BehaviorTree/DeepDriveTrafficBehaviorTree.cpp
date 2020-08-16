
#include "Simulation/Traffic/BehaviorTree/DeepDriveTrafficBehaviorTree.h"
#include "Simulation/Traffic/BehaviorTree/DeepDriveTrafficBehaviorTreeNode.h"
#include "Simulation/Traffic/BehaviorTree/DeepDriveTrafficBlackboard.h"

#include "Agent/DeepDriveAgent.h"

#include "ActorEventLogging/Public/ActorEventLoggingMacros.h"

DEFINE_LOG_CATEGORY(LogDeepDriveTrafficBehaviorTree);

DeepDriveTrafficBehaviorTree::DeepDriveTrafficBehaviorTree(const FString &name)
	:	m_Name(name)
{
	m_Blackboard = new DeepDriveTrafficBlackboard();

	m_RootNode = new DeepDriveTrafficBehaviorTreeNode(*m_Blackboard, "Root");
	m_Nodes.Add(m_RootNode);
}

DeepDriveTrafficBehaviorTree::~DeepDriveTrafficBehaviorTree()
{
	delete m_Blackboard;
}

void DeepDriveTrafficBehaviorTree::initialize(ADeepDriveAgent &agent)
{
	m_Blackboard->setAgent(agent);
}

void DeepDriveTrafficBehaviorTree::bind(DeepDrivePartialPath &path, SDeepDriveManeuver &maneuver)
{
	m_Blackboard->setPartialPath(path);
	m_Blackboard->setManeuver(maneuver);

	if(m_RootNode)
	{
		m_RootNode->bind(path);
	}
}

void DeepDriveTrafficBehaviorTree::execute(float deltaSeconds, float &speed, int32 pathPointIndex)
{
	if(m_isDebuggingActive)
		UE_LOG(LogDeepDriveTrafficBehaviorTree, Log, TEXT("DeepDriveTrafficBehaviorTree::execute %s started at index %d"), *m_Name, pathPointIndex);

	AEL_MESSAGE((*m_Blackboard->getAgent()), TEXT("DeepDriveTrafficBehaviorTree::execute %s at index %d"), *m_Name, pathPointIndex);

	if (m_RootNode)
	{
		m_RootNode->execute(deltaSeconds, speed, pathPointIndex);
	}

	if(m_isDebuggingActive)
		UE_LOG(LogDeepDriveTrafficBehaviorTree, Log, TEXT("DeepDriveTrafficBehaviorTree::execute %s finished"), *m_Name);
}


DeepDriveTrafficBehaviorTreeNode* DeepDriveTrafficBehaviorTree::createNode(DeepDriveTrafficBehaviorTreeNode *rootNode, const FString &name)
{
	DeepDriveTrafficBehaviorTreeNode *node = new DeepDriveTrafficBehaviorTreeNode(*m_Blackboard, name);
	m_Nodes.Add(node);

	if(rootNode)
		rootNode->addChild(node);
	else
		m_RootNode->addChild(node);

	return node;
}


void DeepDriveTrafficBehaviorTree::activateDebugging()
{
	if(m_Blackboard)
		m_Blackboard->setBooleanValue("DebuggingActive", true);
	m_isDebuggingActive = true;
}
