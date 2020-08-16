
#include "DeepDrivePluginPrivatePCH.h"
#include "Private/Simulation/RoadNetwork/DeepDriveRouteCalculator.h"

DEFINE_LOG_CATEGORY(LogDeepDriveRouteCalc);

DeepDriveRouteCalculator::DeepDriveRouteCalculator(const SDeepDriveRoadNetwork &roadNetwork)
	:	m_RoadNetwork(roadNetwork)
	,	m_DestinationLinkId(0)
{

}

DeepDriveRouteCalculator::~DeepDriveRouteCalculator()
{
	for(auto &n : m_AllocatedLinks)
		delete n;
}

SDeepDriveRouteData DeepDriveRouteCalculator::calculate(const FVector &start, const FVector &destination)
{
	SDeepDriveRouteData routeData;
	routeData.Start = start;
	routeData.Destination = destination;
	calculate(start, destination, routeData.Links);
	return routeData;
}

bool DeepDriveRouteCalculator::calculate(const FVector &start, const FVector &destination, TArray<uint32> &linksOut)
{
	bool success = false;
	const uint32 startLinkId = m_RoadNetwork.findClosestLink(start);
	m_DestinationLinkId = m_RoadNetwork.findClosestLink(destination);

	if(startLinkId && m_DestinationLinkId)
	{
		m_Destination = destination;

		const SDeepDriveRoadLink &startLink = m_RoadNetwork.Links[startLinkId];
		// const SDeepDriveRoadLink &destLink = m_RoadNetwork.Links[m_DestinationLinkId];

		UE_LOG(LogDeepDriveRouteCalc, Log, TEXT("Route calculation from %d to %d"), startLinkId, m_DestinationLinkId);

		if(startLinkId == m_DestinationLinkId)
		{
			UE_LOG(LogDeepDriveRouteCalc, Log, TEXT("Start link equals destination link, no route for now") );
			// routeData.Links.Add(startLinkId);
		}
		else // if (startLink.ToJunctionId && destLink.FromJunctionId)
		{
			m_OpenList.add( acquireLink(startLinkId, 0, 0.0f) );
			success = false;
			const Link *currentLink = 0;

			do
			{
				currentLink = m_OpenList.pop();
				// add current link to close list to avoid expanding it again
				m_ClosedList.Add(currentLink->LinkId);

				if(expandLink(*currentLink))
				{
					//	path found
					success = true;
					break;
				}

			} while(m_OpenList.isEmpty() == false);

			if(success)// || currentLink) ??
			{
				const float totalCost = currentLink->CostF;

				TArray<uint32> tmpLinks;
				if(currentLink->LinkId != m_DestinationLinkId)
					tmpLinks.Add(m_DestinationLinkId);
				while(currentLink)
				{
					tmpLinks.Add(currentLink->LinkId);
					currentLink = currentLink->Predecessor;
				}
				if(tmpLinks[tmpLinks.Num() - 1] != startLinkId)
					tmpLinks.Add(startLinkId);

				for(int32 i = tmpLinks.Num() - 1; i>= 0; --i)
				{
					linksOut.Add(tmpLinks[i]);
				}

				success = true;
				UE_LOG(LogDeepDriveRouteCalc, Log, TEXT("Route successfully calculated %d links total cost %f"), linksOut.Num(), totalCost );
				for(auto &linkId : linksOut)
					UE_LOG(LogDeepDriveRouteCalc, Log, TEXT("  %s"), *(m_RoadNetwork.getDebugLinkName(linkId)) );

			}
			else
			{
				UE_LOG(LogDeepDriveRouteCalc, Log, TEXT("Route calculation failed") );
				while(currentLink)
				{
					UE_LOG(LogDeepDriveRouteCalc, Log, TEXT("  %s"), *(m_RoadNetwork.getDebugLinkName(currentLink->LinkId)) );
					currentLink = currentLink->Predecessor;
				}
			}
		}
	}
	else
	{
		UE_LOG(LogDeepDriveRouteCalc, Log, TEXT("Route calculation failed No start or destination link %d %d"), startLinkId, m_DestinationLinkId );
	}

	return success;
}


bool DeepDriveRouteCalculator::expandLink(const Link &currentLink)
{
	bool found = false;
	const SDeepDriveJunction &junction = m_RoadNetwork.Junctions[ m_RoadNetwork.Links[ currentLink.LinkId ].ToJunctionId ];
	UE_LOG(LogDeepDriveRouteCalc, Log, TEXT("Expanding link %s with junction Link %d"), *(m_RoadNetwork.getDebugLinkName(currentLink.LinkId)), junction.JunctionId);
	for (auto &outLinkId : junction.LinksOut)
	{
		// UE_LOG(LogDeepDriveRouteCalc, Log, TEXT("Checking link %d"), outLinkId );
		if (junction.isTurningAllowed(currentLink.LinkId, outLinkId))
		{
			// UE_LOG(LogDeepDriveRouteCalc, Log, TEXT("Turning is allowed") );
			if (outLinkId == m_DestinationLinkId)
			{
				// UE_LOG(LogDeepDriveRouteCalc, Log, TEXT("Destination link found") );
				found = true;
				break;
			}

			if(outLinkId == 0 || m_ClosedList.Contains(outLinkId))
				continue;

			const SDeepDriveRoadLink &outLink = m_RoadNetwork.Links[outLinkId];

			if (outLink.ToJunctionId == 0)
				continue;

			const FVector junctionPos = m_RoadNetwork.Junctions[outLink.ToJunctionId].Center;
			const float curC = (currentLink.Position - junctionPos).Size();
			float tentativeG = currentLink.CostG + curC;

			Link *successorLink = m_OpenList.get(outLinkId);
			UE_LOG(LogDeepDriveRouteCalc, Log, TEXT("   Successor Link %s %p with cost %f"), *(m_RoadNetwork.getDebugLinkName(outLinkId)), successorLink, tentativeG);
			if (successorLink && tentativeG >= successorLink->CostG)
				continue;

			if (successorLink == 0)
			{
				successorLink = acquireLink(outLinkId, &currentLink, tentativeG);
				m_OpenList.add(successorLink);
			}
			else
			{
				successorLink->CostF = tentativeG + (m_Destination - junctionPos).Size();
			}
		}
	}
	return found;
}

DeepDriveRouteCalculator::Link* DeepDriveRouteCalculator::acquireLink(uint32 linkId, const Link *predecessor, float costG)
{
	Link *link = new Link(linkId, predecessor);

	if(link)
	{
		m_AllocatedLinks.Add(link);

		link->Position = m_RoadNetwork.Junctions[ m_RoadNetwork.Links[linkId].ToJunctionId ].Center;
		link->CostG = costG;
		link->CostF = costG + (m_Destination - link->Position).Size();

	}

	return link;
}

DeepDriveRouteCalculator::Link::Link(uint32 linkId, const Link *predecessor)
	:	LinkId(linkId)
	,	Predecessor(predecessor)
{

}


void DeepDriveRouteCalculator::OpenList::add(Link *link)
{
	m_Links.Add(link);
	m_LinkMap.Add(link->LinkId, link);
}

bool DeepDriveRouteCalculator::OpenList::isEmpty() const
{
	return m_Links.Num() == 0;
}

DeepDriveRouteCalculator::Link* DeepDriveRouteCalculator::OpenList::pop()
{
	float bestF = TNumericLimits<float>::Max();
	Link *link = 0;
	int32 index = 0;
	for(int32 i = 0; i < m_Links.Num(); ++i)
	{
		if(m_Links[i]->CostF < bestF)
		{
			link = m_Links[i];
			bestF = link->CostF;
			index = i;
		}
	}

	if(link)
	{
		m_Links.RemoveAt(index);
		m_LinkMap.Remove(link->LinkId);
	}

	return link;
}

DeepDriveRouteCalculator::Link* DeepDriveRouteCalculator::OpenList::get(uint32 id)
{
	return m_LinkMap.Contains(id) ? m_LinkMap[id] : 0;
	// return m_PrioQueue.GetKey(id);
}
