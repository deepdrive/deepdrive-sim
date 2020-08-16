
#pragma once

#include "CoreMinimal.h"

DECLARE_LOG_CATEGORY_EXTERN(LogDeepDriveTrafficBehaviorTree, Log, All);

class DeepDriveTrafficBlackboard;
class DeepDrivePartialPath;
struct SDeepDriveManeuver;
class ADeepDriveAgent;
class DeepDriveTrafficBehaviorTreeNode;

class DeepDriveTrafficBehaviorTree
{
public:

	DeepDriveTrafficBehaviorTree(const FString &name = FString());

	~DeepDriveTrafficBehaviorTree();

	void initialize(ADeepDriveAgent &agent);

	void bind(DeepDrivePartialPath &path, SDeepDriveManeuver &maneuver);

	void execute(float deltaSeconds, float &speed, int32 pathPointIndex);

	DeepDriveTrafficBehaviorTreeNode* createNode(DeepDriveTrafficBehaviorTreeNode *rootNode, const FString &name = FString());

	DeepDriveTrafficBehaviorTreeNode* getRootNode();

	DeepDriveTrafficBlackboard& getBlackboard();

	void activateDebugging();

	const FString& getName() const;

private:

	FString										m_Name;

	DeepDriveTrafficBlackboard					*m_Blackboard = 0;

	DeepDrivePartialPath						*m_Path = 0;
	SDeepDriveManeuver							*m_Maneuver = 0;

	DeepDriveTrafficBehaviorTreeNode			*m_RootNode = 0;
	TArray<DeepDriveTrafficBehaviorTreeNode*>	m_Nodes;

	bool 										m_isDebuggingActive = false;
};

inline DeepDriveTrafficBehaviorTreeNode* DeepDriveTrafficBehaviorTree::getRootNode()
{
	return m_RootNode;
}

inline DeepDriveTrafficBlackboard &DeepDriveTrafficBehaviorTree::getBlackboard()
{
	return *m_Blackboard;
}

inline const FString& DeepDriveTrafficBehaviorTree::getName() const
{
	return m_Name;
}
