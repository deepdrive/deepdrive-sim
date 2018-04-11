// 

using UnrealBuildTool;
using System.Collections.Generic;

public class DeepDriveTarget : TargetRules
{
	public DeepDriveTarget(TargetInfo Target) : base(Target)
	{
		Type = TargetType.Game;

		ExtraModuleNames.AddRange( new string[] { "DeepDrive" } );
	}
}
