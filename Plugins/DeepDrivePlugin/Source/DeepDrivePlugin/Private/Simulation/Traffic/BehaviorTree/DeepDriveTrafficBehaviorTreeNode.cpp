
#include "Simulation/Traffic/BehaviorTree/DeepDriveTrafficBehaviorTreeNode.h"
#include "Simulation/Traffic/BehaviorTree/DeepDriveTBTDecoratorBase.h"
#include "Simulation/Traffic/BehaviorTree/DeepDriveTBTTaskBase.h"

DeepDriveTrafficBehaviorTreeNode::DeepDriveTrafficBehaviorTreeNode(DeepDriveTrafficBlackboard &blackboard)
	:	m_Blackboard(blackboard)
{

}

void DeepDriveTrafficBehaviorTreeNode::addChild(DeepDriveTrafficBehaviorTreeNode *childNode)
{
	m_Children.Add(childNode);
}

void DeepDriveTrafficBehaviorTreeNode::addDecorator(DeepDriveTBTDecoratorBase *decorator)
{
	m_Decorators.Add(decorator);
}

void DeepDriveTrafficBehaviorTreeNode::addTask(DeepDriveTBTTaskBase *task)
{
	m_Tasks.Add(task);
}

void DeepDriveTrafficBehaviorTreeNode::bind(DeepDrivePartialPath &path)
{
	for (auto &task : m_Tasks)
		task->bind(m_Blackboard, path);

	for (auto &decorator : m_Decorators)
		decorator->bind(m_Blackboard, path);

	for(auto &childNode : m_Children)
		childNode->bind(path);
}

bool DeepDriveTrafficBehaviorTreeNode::execute(float deltaSeconds, float &speed, int32 pathPointIndex)
{
	bool isRunning = true;
	for (auto &decorator : m_Decorators)
	{
		isRunning = decorator->performCheck(m_Blackboard, pathPointIndex);
		if (isRunning == false)
			break;
	}

	if (isRunning)
	{
		for (auto &task : m_Tasks)
		{
			isRunning = task->execute(m_Blackboard, deltaSeconds, speed, pathPointIndex);
			if (isRunning == false)
				break;
		}

		if (isRunning)
		{
			for (auto &child : m_Children)
			{
				isRunning = child->execute(deltaSeconds, speed, pathPointIndex);
				if (isRunning)
					break;
			}
		}
	}

	return isRunning;
}
