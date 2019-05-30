# Unreal Logging

Command line example

```
-LogCmds="global Verbose, LogPython Verbose, LogAnimMontage off, LogDeepDriveAgent VeryVerbose"
```

In DefaultEngine.ini or Engine.ini:

```
[Core.Log]
global=[default verbosity for things not listed later]
[cat]=[level]
foo=verbose break
```

#### Our log categories

    Defined with DEFINE_LOG_CATEGORY macro

```
LogSunSimulationComponentLogSunLightSimulator
LogDeepDriveAgentControllerBase
LogDeepDriveAgentSteeringController
LogDeepDriveAgent
LogDeepDriveSimulationServer
LogDeepDriveConnectionThread
LogDeepDriveAgentLocalAIController
LogDeepDriveAgentSpeedController
LogDeepDriveSimulationCaptureProxy
LogDeepDriveSimulationMessageHandler
LogDeepDriveSimulationServerProxy
LogDeepDriveSimulation
LogDeepDriveSplineTrack
LogSharedMemCaptureMessageBuilder
LogSharedMemCaptureSinkWorkerLogSharedMemCaptureSinkWorker
LogSharedMemCaptureSinkComponent
DeepDriveCaptureProxy
DeepDriveCaptureComponent
LogDeepDriveCapture
LogCaptureBufferPool
LogDeepDriveConnectionListener
LogDeepDriveClientConnection
LogDeepDriveServer
LogDeepDrivePlugin
LogSharedMemoryImpl_Linux
LogPython
```


#### Verbosities

Fatal

    Fatal level logs are always printed to console and log files and crashes even if logging is disabled.

Error

    Error level logs are printed to console and log files. These appear red by default.

Warning

    Warning level logs are printed to console and log files. These appear yellow by default.

Display

    Display level logs are printed to console and log files.

Log

    Log level logs are printed to log files but not to the in-game console. They can still be viewed in editor as they appear via the Output Log window.

Verbose

    Verbose level logs are printed to log files but not the in-game console. This is usually used for detailed logging and debugging.

VeryVerbose

    VeryVerbose level logs are printed to log files but not the in-game console. This is usually used for very detailed logging that would otherwise spam output.
