
// Copyright 1998-2016 Epic Games, Inc. All Rights Reserved.

using UnrealBuildTool;
using System;
using System.Collections.Generic;
using System.Text;
using System.IO;
using System.Reflection;
using System.Diagnostics;
using System.Runtime.InteropServices;


public class DeepDrivePlugin : ModuleRules
{
	public DeepDrivePlugin(ReadOnlyTargetRules Target) : base(Target)
	{
		
		PublicIncludePaths.AddRange(
			new string[] {
				"DeepDrivePlugin/Public"
				// ... add public include paths required here ...
			}
			);
				
		
		PrivateIncludePaths.AddRange(
			new string[] {
				"DeepDrivePlugin/Private",
				// ... add other private include paths required here ...
			}
			);
			
		
		PublicDependencyModuleNames.AddRange(
			new string[]
			{
				"Core",
				"Sockets","Networking"
				// ... add other public dependencies that you statically link with here ...
			}
			);
			
		
		PrivateDependencyModuleNames.AddRange(
			new string[]
			{
				"CoreUObject",
				"Engine",
				"Slate",
				"SlateCore",
                "RenderCore",
                "RHI",
	            "PhysXVehicles",
				"Landscape"
            }
            );

		 if (Target.bBuildEditor)
		 {
			 PublicDependencyModuleNames.AddRange(
				 new string[] {
					"EditorStyle",
					"PropertyEditor",
				 }
			 );
		 }
		
		DynamicallyLoadedModuleNames.AddRange(
			new string[]
			{
				// ... add any modules that your module loads dynamically here ...
			}
			);


		if ((Target.Platform == UnrealTargetPlatform.Win64) || (Target.Platform == UnrealTargetPlatform.Win32))
			Definitions.Add("DEEPDRIVE_PLATFORM_WINDOWS=1");
		else if ((Target.Platform == UnrealTargetPlatform.Linux))
			Definitions.Add("DEEPDRIVE_PLATFORM_LINUX=1");

        Definitions.Add("DEEPDRIVE_WITH_UE4_LOGGING");

        string rootDir = System.IO.Directory.GetParent(
                                        System.IO.Directory.GetParent(
                                        System.IO.Directory.GetParent(
                                        System.IO.Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location))
                                        .FullName).FullName).FullName;

        // Build Python extension -------------------------------------------------------------------------------
        var envHome = (Target.Platform == UnrealTargetPlatform.Win64) || (Target.Platform == UnrealTargetPlatform.Win32) ?  "HOMEPATH" : "HOME";

        var userHome = Environment.GetEnvironmentVariable(envHome);
        Console.WriteLine("userHome " + userHome);
        var pythonBinConfig = Path.Combine(userHome, ".deepdrive", "python_bin");
        Console.WriteLine("pythonBinConfig " + pythonBinConfig);

        if( ! File.Exists(pythonBinConfig))
        {
            Console.WriteLine("Did not find python binary used to import deepdrive last, so not building the extension");
        } else
        {
            Console.WriteLine("Building extension ...");

            var pythonBin = System.IO.File.ReadAllText(pythonBinConfig);
            Console.WriteLine("pythonBin " + pythonBin);

            var buildExtensionArguments = "\"" + Path.Combine(
            		rootDir, "Plugins", "DeepDrivePlugin", "Source", "DeepDrivePython", "build", "build.py") +
            	"\" --type dev";
            Console.WriteLine("Build extension script " + buildExtensionArguments);
            System.Diagnostics.Process pProcess = new System.Diagnostics.Process();
            pProcess.StartInfo.FileName = pythonBin;
            pProcess.StartInfo.Arguments = buildExtensionArguments;
            pProcess.StartInfo.UseShellExecute = false;
            pProcess.StartInfo.RedirectStandardOutput = true;
            pProcess.StartInfo.RedirectStandardError = true;

            pProcess.Start();
            string strOutput = pProcess.StandardOutput.ReadToEnd();
            string strErr = pProcess.StandardError.ReadToEnd();

            Console.WriteLine(strOutput);
            Console.WriteLine(strErr);

            pProcess.WaitForExit();

            if(pProcess.ExitCode != 0)
            {
                throw new Exception("Could not build extension. Aborting build");
            }
        }
    }
}
