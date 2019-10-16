
#pragma once

#include "CoreMinimal.h"

DECLARE_LOG_CATEGORY_EXTERN(LogDeepDriveTrafficBehaviorTreeNode, Log, All);

class DeepDriveTrafficBlackboard;
class DeepDriveTBTDecoratorBase;
class DeepDriveTBTTaskBase;
class DeepDrivePartialPath;

class DeepDriveTrafficBehaviorTreeNode
{
public:

	DeepDriveTrafficBehaviorTreeNode(DeepDriveTrafficBlackboard &blackboard, const FString &name = FString());

	void addChild(DeepDriveTrafficBehaviorTreeNode *childNode);

	void addDecorator(TSharedPtr<DeepDriveTBTDecoratorBase> decorator);
	void addDecorator(DeepDriveTBTDecoratorBase *decorator);

	void addTask(TSharedPtr<DeepDriveTBTTaskBase> task);
	void addTask(DeepDriveTBTTaskBase *task);

	void bind(DeepDrivePartialPath &path);

	bool execute(float deltaSeconds, float &speed, int32 pathPointIndex);

private:

	DeepDriveTrafficBlackboard						&m_Blackboard;
	FString											m_Name;

	TArray<DeepDriveTrafficBehaviorTreeNode*>		m_Children;

	TArray< TSharedPtr<DeepDriveTBTDecoratorBase> >	m_Decorators;
	TArray< TSharedPtr<DeepDriveTBTTaskBase> >		m_Tasks;

};
