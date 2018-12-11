
#pragma once

#if WITH_EDITOR
#include "CoreMinimal.h"
#include "UObject/WeakObjectPtr.h"
#include "Input/Reply.h"
#include "IDetailCustomization.h"

class ADeepDriveSplineTrack;
class IDetailLayoutBuilder;

class FDeepDriveSplineTrackDetails : public IDetailCustomization
{
public:
	/** Makes a new instance of this detail layout class for a specific detail view requesting it */
	static TSharedRef<IDetailCustomization> MakeInstance();
private:
	/** IDetailCustomization interface */
	virtual void CustomizeDetails( IDetailLayoutBuilder& DetailLayout ) override;

	FReply OnExportSpline();

private:
	/** The selected sky light */
	TWeakObjectPtr<ADeepDriveSplineTrack>   m_Track;
};
#endif