
#pragma once

#include "Public/Simulation/RoadNetwork/DeepDriveRoadNetwork.h"
#include "Runtime/Core/Public/Containers/BinaryHeap.h"

DECLARE_LOG_CATEGORY_EXTERN(LogDeepDriveRouteCalc, Log, All);

class DeepDriveRouteCalculator
{
	struct Node
	{
		Node(uint32 junctionId, const Node *predecessor = 0, uint32 linkId = 0);

		bool operator < (const Node &rhs) const
		{
			return CostF < rhs.CostF;
		}

		uint32			JunctionId = 0;
		const Node		*Predecessor = 0;
		uint32			LinkId = 0;

		FVector			Position;

		float			CostF = TNumericLimits<float>::Max();
		float			CostG = TNumericLimits<float>::Max();
	};

	class OpenList
	{
	public:

		void add(Node *node);

		bool isEmpty() const;

		Node* pop();

		Node* get(uint32 id);

	private:

		FBinaryHeap<Node*, uint32>		m_PrioQueue;

		TArray<Node*>					m_Nodes;
		TMap<uint32, Node*>				m_NodeMap;
	};

public:

	DeepDriveRouteCalculator(const SDeepDriveRoadNetwork &roadNetwork);
	~DeepDriveRouteCalculator();

	SDeepDriveRouteData calculate(const FVector &start, const FVector &destination);


private:

	void expandNode(const Node &currentNode);

	Node* acquireNode(uint32 junctionId, const Node *predecessor, uint32 linkId, float costG);

	const SDeepDriveRoadNetwork         &m_RoadNetwork;

	FVector                             m_Destination;

	OpenList                            m_OpenList;
	TSet<uint32>                        m_ClosedList;

	TArray<Node*>						m_AllocatedNodes;

};
