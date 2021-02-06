#if WITH_EDITOR


#include "Simulation/Misc/DeepDriveSplineTrackDetails.h"
#include "Simulation/Misc/DeepDriveSplineTrack.h"

#include "Components/SceneComponent.h"
#include "Widgets/DeclarativeSyntaxSupport.h"
#include "Widgets/Text/STextBlock.h"
#include "Widgets/Input/SButton.h"
#include "Engine/World.h"
#include "PropertyHandle.h"
#include "DetailLayoutBuilder.h"
#include "DetailWidgetRow.h"
#include "DetailCategoryBuilder.h"
#include "IDetailsView.h"

#define LOCTEXT_NAMESPACE "DeepDriveSplineTrackDetails"



TSharedRef<IDetailCustomization> FDeepDriveSplineTrackDetails::MakeInstance()
{
	return MakeShareable( new FDeepDriveSplineTrackDetails );
}

void FDeepDriveSplineTrackDetails::CustomizeDetails( IDetailLayoutBuilder& DetailLayout )
{
	const TArray< TWeakObjectPtr<UObject> >& SelectedObjects = DetailLayout.GetSelectedObjects();

	for( int32 ObjectIndex = 0; ObjectIndex < SelectedObjects.Num(); ++ObjectIndex )
	{
		const TWeakObjectPtr<UObject>& CurrentObject = SelectedObjects[ObjectIndex];
		if ( CurrentObject.IsValid() )
		{
			ADeepDriveSplineTrack* curObject = Cast<ADeepDriveSplineTrack>(CurrentObject.Get());
			if (curObject != NULL)
			{
				m_Track = curObject;
				break;
			}
		}
	}

	check(m_Track.IsValid());

	DetailLayout.EditCategory( "Export" )
	.AddCustomRow( NSLOCTEXT("ExportDetails", "Export", "Export") )
		.NameContent()
		[
			SNew( STextBlock )
			.Font( IDetailLayoutBuilder::GetDetailFont() )
			.Text( NSLOCTEXT("ExportDetails", "Export", "Export") )
		]
		.ValueContent()
		.MaxDesiredWidth(125.f)
		.MinDesiredWidth(125.f)
		[
			SNew(SButton)
			.ContentPadding(2)
			.VAlign(VAlign_Center)
			.HAlign(HAlign_Center)
			.OnClicked( this, &FDeepDriveSplineTrackDetails::OnExportSpline )
			[
				SNew( STextBlock )
				.Font( IDetailLayoutBuilder::GetDetailFont() )
				.Text( NSLOCTEXT("ExportDetails", "Export", "Export") )
			]
		];
}

FReply FDeepDriveSplineTrackDetails::OnExportSpline()
{
	if (m_Track.IsValid())
	{
		m_Track->exportTrack();
	}

	return FReply::Handled();
}

#undef LOCTEXT_NAMESPACE

#endif