
#pragma once


class DeepDriveTrafficBlackboard;
class DeepDrivePartialPath;
struct SDeepDriveManeuver;
class ADeepDriveAgent;
class DeepDriveTrafficBehaviorTreeNode;

class DeepDriveTrafficBehaviorTree
{
public:
	DeepDriveTrafficBehaviorTree();

	~DeepDriveTrafficBehaviorTree();

	void initialize(ADeepDriveAgent &agent);

	void bind(DeepDrivePartialPath &path, SDeepDriveManeuver &maneuver);

	void execute(float deltaSeconds, float &speed, int32 pathPointIndex);

	DeepDriveTrafficBehaviorTreeNode* createNode(DeepDriveTrafficBehaviorTreeNode *rootNode);

	DeepDriveTrafficBlackboard& getBlackboard();

private :

	DeepDriveTrafficBlackboard					*m_Blackboard = 0;

	DeepDrivePartialPath						*m_Path = 0;
	SDeepDriveManeuver						*m_Maneuver = 0;

	DeepDriveTrafficBehaviorTreeNode			*m_RootNode = 0;
	TArray<DeepDriveTrafficBehaviorTreeNode*>	m_Nodes;

};

inline DeepDriveTrafficBlackboard &DeepDriveTrafficBehaviorTree::getBlackboard()
{
	return *m_Blackboard;
}
