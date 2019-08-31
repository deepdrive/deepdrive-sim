
#pragma once

#include "Public/Simulation/RoadNetwork/DeepDriveRoadNetwork.h"
#include "Runtime/Core/Public/Containers/BinaryHeap.h"

DECLARE_LOG_CATEGORY_EXTERN(LogDeepDriveRouteCalc, Log, All);

class DeepDriveRouteCalculator
{
	struct Link
	{
		Link(uint32 linkId, const Link *predecessor = 0);

		bool operator < (const Link &rhs) const
		{
			return CostF < rhs.CostF;
		}

		uint32			LinkId = 0;
		const Link		*Predecessor = 0;

		FVector			Position;

		float			CostF = TNumericLimits<float>::Max();
		float			CostG = TNumericLimits<float>::Max();
	};

	class OpenList
	{
	public:

		void add(Link *link);

		bool isEmpty() const;

		Link* pop();

		Link* get(uint32 id);

	private:

		FBinaryHeap<Link*, uint32>		m_PrioQueue;

		TArray<Link*>					m_Links;
		TMap<uint32, Link*>				m_LinkMap;
	};

public:

	DeepDriveRouteCalculator(const SDeepDriveRoadNetwork &roadNetwork);
	~DeepDriveRouteCalculator();

	bool calculate(const FVector &start, const FVector &destination, TArray<uint32> &links);

	SDeepDriveRouteData calculate(const FVector &start, const FVector &destination);


private:

	bool expandLink(const Link &currentLink);

	Link* acquireLink(uint32 linkId, const Link *predecessor, float costG);

	const SDeepDriveRoadNetwork         &m_RoadNetwork;

	FVector                             m_Destination;
	uint32								m_DestinationLinkId = 0;

	OpenList                            m_OpenList;
	TSet<uint32>                        m_ClosedList;

	TArray<Link*>						m_AllocatedLinks;

};
