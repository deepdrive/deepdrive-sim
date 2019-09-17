
#include "Simulation/Traffic/BehaviorTree/DeepDriveTrafficBehaviorTree.h"
#include "Simulation/Traffic/BehaviorTree/DeepDriveTrafficBehaviorTreeNode.h"
#include "Simulation/Traffic/BehaviorTree/DeepDriveTrafficBlackboard.h"

DeepDriveTrafficBehaviorTree::DeepDriveTrafficBehaviorTree()
{
	m_Blackboard = new DeepDriveTrafficBlackboard();

	m_RootNode = new DeepDriveTrafficBehaviorTreeNode(*m_Blackboard);
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
	if(m_RootNode)
	{
		m_RootNode->execute(deltaSeconds, speed, pathPointIndex);
	}
}


DeepDriveTrafficBehaviorTreeNode* DeepDriveTrafficBehaviorTree::createNode(DeepDriveTrafficBehaviorTreeNode *rootNode)
{
	DeepDriveTrafficBehaviorTreeNode *node = new DeepDriveTrafficBehaviorTreeNode(*m_Blackboard);
	m_Nodes.Add(node);

	if(rootNode)
		rootNode->addChild(node);
	else
		m_RootNode->addChild(node);

	return node;
}
