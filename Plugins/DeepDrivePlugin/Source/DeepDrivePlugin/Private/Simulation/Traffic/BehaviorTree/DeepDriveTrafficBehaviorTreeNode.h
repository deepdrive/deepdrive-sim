
#pragma once

#include "CoreMinimal.h"

class DeepDriveTrafficBlackboard;
class DeepDriveTBTDecoratorBase;
class DeepDriveTBTTaskBase;
class DeepDrivePartialPath;

class DeepDriveTrafficBehaviorTreeNode
{
public:

	DeepDriveTrafficBehaviorTreeNode(DeepDriveTrafficBlackboard &blackboard);

	void addChild(DeepDriveTrafficBehaviorTreeNode *childNode);

	void addDecorator(DeepDriveTBTDecoratorBase *decorator);

	void addTask(DeepDriveTBTTaskBase *task);

	void bind(DeepDrivePartialPath &path);

	bool execute(float deltaSeconds, float &speed, int32 pathPointIndex);

private:

	DeepDriveTrafficBlackboard						&m_Blackboard;
	
	TArray<DeepDriveTrafficBehaviorTreeNode*>		m_Children;


	TArray<DeepDriveTBTDecoratorBase*>				m_Decorators;
	TArray<DeepDriveTBTTaskBase*>					m_Tasks;

};
