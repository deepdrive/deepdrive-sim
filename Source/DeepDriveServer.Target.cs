using UnrealBuildTool;
using System.Collections.Generic;

[SupportedPlatforms(UnrealPlatformClass.Server)]
public class DeepDriveServerTarget : TargetRules   // Change this line as shown previously
{
       public DeepDriveServerTarget(TargetInfo Target) : base(Target)  // Change this line as shown previously
       {
        Type = TargetType.Server;
        ExtraModuleNames.Add("DeepDrive");    // Change this line as shown previously
       }
}
