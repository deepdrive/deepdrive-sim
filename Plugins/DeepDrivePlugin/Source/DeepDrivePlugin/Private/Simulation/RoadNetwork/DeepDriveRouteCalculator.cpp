
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

	const uint32 startLinkId = m_RoadNetwork.findClosestLink(start);
	m_DestinationLinkId = m_RoadNetwork.findClosestLink(destination);

	if(startLinkId && m_DestinationLinkId)
	{

		routeData.Start = start;
		routeData.Destination = destination;
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
			bool success = false;
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

			if(success || currentLink)
			{
				TArray<uint32> routeLinks;

				const float totalCost = currentLink->CostF;

				if(currentLink->LinkId != m_DestinationLinkId)
					routeLinks.Add(m_DestinationLinkId);
				while(currentLink)
				{
					routeLinks.Add(currentLink->LinkId);
					currentLink = currentLink->Predecessor;
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
		UE_LOG(LogDeepDriveRouteCalc, Log, TEXT("Route calculation failed No start or destination link %d %d"), startLinkId, m_DestinationLinkId );
	}

	return routeData;
}


bool DeepDriveRouteCalculator::expandLink(const Link &currentLink)
{
	bool found = false;
	const SDeepDriveJunction &junction = m_RoadNetwork.Junctions[ m_RoadNetwork.Links[ currentLink.LinkId ].ToJunctionId ];
	UE_LOG(LogDeepDriveRouteCalc, Log, TEXT("Expanding link %d with junction Link %d turningRestrictions %d"), currentLink.LinkId, junction.JunctionId, junction.TurningRestrictions.Num());
	for (auto &outLinkId : junction.LinksOut)
	{
		if (junction.isTurningAllowed(currentLink.LinkId, outLinkId))
		{

			if (outLinkId == m_DestinationLinkId)
			{
				found = true;
				break;
			}

			if(outLinkId == 0 || m_ClosedList.Contains(outLinkId))
				continue;

			const SDeepDriveRoadLink &outLink = m_RoadNetwork.Links[outLinkId];

			const FVector junctionPos = m_RoadNetwork.Junctions[outLink.ToJunctionId].Center;
			const float curC = (currentLink.Position - junctionPos).Size();
			float tentativeG = currentLink.CostG + curC;

			Link *successorLink = m_OpenList.get(outLinkId);
			UE_LOG(LogDeepDriveRouteCalc, Log, TEXT("Successor Link %d %p with cost %f"), outLinkId, successorLink, tentativeG);
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
