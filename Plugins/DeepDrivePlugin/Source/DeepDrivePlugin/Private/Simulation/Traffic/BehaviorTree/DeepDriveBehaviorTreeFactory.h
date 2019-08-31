
#pragma once

#include "CoreMinimal.h"

#include "Simulation/RoadNetwork/DeepDriveRoadNetwork.h"

#include <map>
#include <functional>

class ADeepDriveSimulation;
class ADeepDriveAgent;
class DeepDriveTrafficBehaviorTree;
struct SDeepDriveManeuver;


class DeepDriveBehaviorTreeFactory
{
public:

	typedef std::function<DeepDriveTrafficBehaviorTree *(void)> CreatorFuncPtr;

	static DeepDriveBehaviorTreeFactory &GetInstance();

	static void Destroy();

	DeepDriveBehaviorTreeFactory();

	void registerCreator(const FString &key, CreatorFuncPtr creator);

	DeepDriveTrafficBehaviorTree* createBehaviorTree(const FString &key);

private:

	typedef TMap<FString, CreatorFuncPtr> Creators;

	Creators								m_Creators;

	static DeepDriveBehaviorTreeFactory		*theInstance;

};
