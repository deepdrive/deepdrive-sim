
#include "Simulation/Traffic/BehaviorTree/DeepDriveTrafficBehaviorTreeNode.h"
#include "Simulation/Traffic/BehaviorTree/DeepDriveTBTDecoratorBase.h"
#include "Simulation/Traffic/BehaviorTree/DeepDriveTBTTaskBase.h"

#include "Simulation/Traffic/BehaviorTree/DeepDriveTrafficBlackboard.h"

DEFINE_LOG_CATEGORY(LogDeepDriveTrafficBehaviorTreeNode);

DeepDriveTrafficBehaviorTreeNode::DeepDriveTrafficBehaviorTreeNode(DeepDriveTrafficBlackboard &blackboard, const FString &name)
	:	m_Blackboard(blackboard)
	,	m_Name(name)
{

}

void DeepDriveTrafficBehaviorTreeNode::addChild(DeepDriveTrafficBehaviorTreeNode *childNode)
{
	m_Children.Add(childNode);
}

void DeepDriveTrafficBehaviorTreeNode::addDecorator(TSharedPtr<DeepDriveTBTDecoratorBase> decorator)
{
	m_Decorators.Add(decorator);
}

void DeepDriveTrafficBehaviorTreeNode::addDecorator(DeepDriveTBTDecoratorBase *decorator)
{
	addDecorator(TSharedPtr<DeepDriveTBTDecoratorBase>(decorator));
}

void DeepDriveTrafficBehaviorTreeNode::addTask(TSharedPtr<DeepDriveTBTTaskBase> task)
{
	m_Tasks.Add(task);
}

void DeepDriveTrafficBehaviorTreeNode::addTask(DeepDriveTBTTaskBase *task)
{
	addTask(TSharedPtr<DeepDriveTBTTaskBase>(task));
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
	const bool debuggingActive = m_Blackboard.getBooleanValue("DebuggingActive", false);
	if(debuggingActive)
		UE_LOG(LogDeepDriveTrafficBehaviorTreeNode, Log, TEXT("DeepDriveTrafficBehaviorTreeNode::execute %s started"), *m_Name);

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

	if(debuggingActive)
		UE_LOG(LogDeepDriveTrafficBehaviorTreeNode, Log, TEXT("DeepDriveTrafficBehaviorTreeNode::execute %s finished, isRunning %c"), *m_Name, isRunning ? 'T' : 'F');

	return isRunning;
}
