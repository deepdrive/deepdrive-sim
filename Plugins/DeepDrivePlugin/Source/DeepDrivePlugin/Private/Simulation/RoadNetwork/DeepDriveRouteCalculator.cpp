
#include "DeepDrivePluginPrivatePCH.h"
#include "Private/Simulation/RoadNetwork/DeepDriveRouteCalculator.h"

DEFINE_LOG_CATEGORY(LogDeepDriveRouteCalc);

DeepDriveRouteCalculator::DeepDriveRouteCalculator(const SDeepDriveRoadNetwork &roadNetwork)
	:	m_RoadNetwork(roadNetwork)
{

}

DeepDriveRouteCalculator::~DeepDriveRouteCalculator()
{
	for(auto &n : m_AllocatedNodes)
		delete n;
}

SDeepDriveRouteData DeepDriveRouteCalculator::calculate(const FVector &start, const FVector &destination)
{
	SDeepDriveRouteData routeData;

	const uint32 startLinkId = m_RoadNetwork.findClosestLink(start);
	const uint32 destLinkId = m_RoadNetwork.findClosestLink(destination);

	if(startLinkId && destLinkId)
	{

		routeData.Start = start;
		routeData.Destination = destination;
		m_Destination = destination;

		const SDeepDriveRoadLink &startLink = m_RoadNetwork.Links[startLinkId];
		const SDeepDriveRoadLink &destLink = m_RoadNetwork.Links[destLinkId];

		UE_LOG(LogDeepDriveRouteCalc, Log, TEXT("Route calculation from %d to %d starting at %d"), startLinkId, destLinkId, destLink.FromJunctionId );

		if(startLinkId == destLinkId)
		{
			UE_LOG(LogDeepDriveRouteCalc, Log, TEXT("Start link equals destination link, no route for now") );
			// routeData.Links.Add(startLinkId);
		}
		else if(startLink.ToJunctionId == destLink.FromJunctionId)
		{
			routeData.Links.Add(startLinkId);
			routeData.Links.Add(destLinkId);
		}
		else if (startLink.ToJunctionId && destLink.FromJunctionId)
		{
			const uint32 destJunctionId = destLink.FromJunctionId;
			m_OpenList.add( acquireNode(startLink.ToJunctionId, 0, startLinkId, 0.0f) );
			bool success = false;
			const Node *currentNode = 0;

			do
			{
				// Knoten mit dem geringsten f-Wert aus der Open List entfernen
				currentNode = m_OpenList.pop();
				// Wurde das Ziel gefunden?
				if(currentNode->JunctionId == destJunctionId)
				{
					//	path found
					success = true;
					break;
				}
				// Der aktuelle Knoten soll durch nachfolgende Funktionen
				// nicht weiter untersucht werden, damit keine Zyklen entstehen
				m_ClosedList.Add(currentNode->JunctionId);

				// Wenn das Ziel noch nicht gefunden wurde: Nachfolgeknoten
				// des aktuellen Knotens auf die Open List setzen
				expandNode(*currentNode);

			} while(m_OpenList.isEmpty() == false);

			if(success)
			{
				TArray<uint32> routeLinks;

				const float totalCost = currentNode->CostF;

				if(currentNode->LinkId != destLinkId)
					routeLinks.Add(destLinkId);
				while(currentNode)
				{
					routeLinks.Add(currentNode->LinkId);
					currentNode = currentNode->Predecessor;
				}
				if(routeLinks[routeLinks.Num() - 1] != startLinkId)
					routeLinks.Add(startLinkId);

				for(int32 i = routeLinks.Num() - 1; i>= 0; --i)
				{
					routeData.Links.Add(routeLinks[i]);
					UE_LOG(LogDeepDriveRouteCalc, Log, TEXT("  %d"), routeLinks[i] );
				}

				UE_LOG(LogDeepDriveRouteCalc, Log, TEXT("Route successfully calculated %d links total cost %f"), routeData.Links.Num(), totalCost );

			}
			else
			{
				UE_LOG(LogDeepDriveRouteCalc, Log, TEXT("Route calculation failed") );
			}
		}
	}
	else
	{
		UE_LOG(LogDeepDriveRouteCalc, Log, TEXT("Route calculation failed No start or destination link %d %d"), startLinkId, destLinkId );
	}

	return routeData;
}

void DeepDriveRouteCalculator::expandNode(const Node &currentNode)
{
	UE_LOG(LogDeepDriveRouteCalc, Log, TEXT("Expanding junction node %d"), currentNode.JunctionId );
	const SDeepDriveJunction &junction = m_RoadNetwork.Junctions[currentNode.JunctionId];
	for(auto &outLinkId : junction.LinksOut)
	{
		if(junction.isTurningAllowed(currentNode.LinkId, outLinkId))
		{
			const SDeepDriveRoadLink &outLink = m_RoadNetwork.Links[outLinkId];

			if	(	outLink.ToJunctionId == 0
				||	m_ClosedList.Contains(outLink.ToJunctionId)
				)
				continue;

			const FVector junctionPos = m_RoadNetwork.Junctions[outLink.ToJunctionId].Center;
			const float curC = (currentNode.Position - junctionPos).Size();
			float tentativeG = currentNode.CostG + curC;

			Node *successorNode = m_OpenList.get(outLink.ToJunctionId);
			UE_LOG(LogDeepDriveRouteCalc, Log, TEXT("Successor node %d %p"), outLink.ToJunctionId, successorNode );
			if(successorNode && tentativeG >= successorNode->CostG)
				continue;

			if(successorNode == 0)
			{
				successorNode = acquireNode(outLink.ToJunctionId, &currentNode, outLinkId, tentativeG);
				m_OpenList.add(successorNode);
			}
			else
			{
				successorNode->CostF = tentativeG + (m_Destination - junctionPos).Size();
			}
		}
	}
}

DeepDriveRouteCalculator::Node* DeepDriveRouteCalculator::acquireNode(uint32 junctionId, const Node *predecessor, uint32 linkId, float costG)
{
	Node *node = new Node(junctionId, predecessor, linkId);

	if(node)
	{
		m_AllocatedNodes.Add(node);

		node->Position = m_RoadNetwork.Junctions[junctionId].Center;
		node->CostG = costG;
		node->CostF = costG + (m_Destination - node->Position).Size();
	}

	return node;
}

DeepDriveRouteCalculator::Node::Node(uint32 junctionId, const Node *predecessor, uint32 linkId)
	:	JunctionId(junctionId)
	,	Predecessor(predecessor)
	,	LinkId(linkId)
{

}


void DeepDriveRouteCalculator::OpenList::add(Node *node)
{
	m_Nodes.Add(node);
	m_NodeMap.Add(node->JunctionId, node);

	// m_PrioQueue.Add(node, node->JunctionId);
}

bool DeepDriveRouteCalculator::OpenList::isEmpty() const
{
	return m_Nodes.Num() == 0;
	// return m_PrioQueue.Num() == 0;
}

DeepDriveRouteCalculator::Node* DeepDriveRouteCalculator::OpenList::pop()
{
	float bestF = TNumericLimits<float>::Max();
	Node *node = 0;
	int32 index = 0;
	for(int32 i = 0; i < m_Nodes.Num(); ++i)
	{
		if(m_Nodes[i]->CostF < bestF)
		{
			node = m_Nodes[i];
			bestF = node->CostF;
			index = i;
		}
	}

	if(node)
	{
		m_Nodes.RemoveAt(index);
		m_NodeMap.Remove(node->JunctionId);
	}

	return node;

	// const uint32 index = m_PrioQueue.Top();
	// Node *node = m_PrioQueue.GetKey(index);
	// m_PrioQueue.Pop();
	// return node;
}

DeepDriveRouteCalculator::Node* DeepDriveRouteCalculator::OpenList::get(uint32 id)
{
	return m_NodeMap.Contains(id) ? m_NodeMap[id] : 0;
	// return m_PrioQueue.GetKey(id);
}
