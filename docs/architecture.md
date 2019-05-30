# Architecture

deepdrive-sim consists of two main components.

1. https://github.com/deepdrive/deepdrive[DeepDrive] The 'game' or simulation, including all the assets, materials, blueprints, lighting, etc... a la a traditional Unreal game project
2. https://github.com/deepdrive/deepdrive-plugin[DeepDrivePlugin] The liaison between Unreal and your Python, delivering sensor data from the game and receiving control and configuration commands through a combined C++ Python extension / Unreal Plugin.

## DeepDrivePlugin technical details

### Abstract

The DeepDrivePlugin allows extracting images and depth from several cameras via shared memory along with controlling the car and configuring the simulation over TCP sockets.

### Overview

The functionality of the DeepDrivePlugin can divided into 2 main domains:

 - Capturing
 - Control

Capturing provides the means to extract an arbitrary number of live streams out of the simulated world including current state data of the agent.
Control provides the means to control capturing.

![Overview](/docs/images/DeepDrive_Overview.png)


### Capturing

Capturing is the process of regularly taking a snapshot of the simulated world and processing the data (i.e. transferring snapshot data to a client). Capturing is separated into these 2 steps:

1. Taking the snapshot
2. Processing the data

Currently a snapshot consists of the following data:

- state data of agent
- upto n camera streams

whereas a camera stream currently contains the following data:

- HDR image rendered by camera
- scene depth buffer from camera's point of view

#### Overview

The whole process of capturing is controlled by DeepDriveCapture which is also responsible of collecting state data from the agent. The actual capturing of the simulated world is done by a Capture Camera Component whereas the processing of a captured snapshot is handled by a Capture Sink.

![Overview](/docs/images/DeepDrive_Capturing_Overview.png)


#### Capture Camera Component

Capturing the simulated world is done by CaptureCameraComponents. A CaptureCameraComponent inherits from a UCameraComponent and thus represents a camera. CaptureCameraComponents can be arbitrarily placed into the scene and the number of CaptureCameraComponents is not limited. is used to render an HDR image of the simulated world as seen by the camera it represents. It also captures the depth buffer of the scene. A CaptureCameraComponent can simply be added to any Unreal Actor although it will most likely be added to the Actor representing the agent. A TextureRenderTarget2D must be set on a CaptureCameraComponent added to an Actor. The dimension of this render texture defines the dimension of the captured HDR image and depth buffer. The render target must also be defined as HDR:


![Overview](/docs/images/Screenshot_RenderTarget.png)


For each CaptureCameraComponent a camera type can be defined and an user defined id can be set. Additionally a CaptureCameraComponent can be activated or deactivated at runtime by calling _ActivateCapturing()_ or _DeactivateCapturing()_  from Blueprint or C++. When deactiavted a CaptureCameraComponent isn't contributing to a snapshot anymore.

#### Capture Sinks

Processing a snapshot is done by a so called CaptureSink. Once a snapshot of the simulated world is taken all snapshot data are forwarded to a CaptureSink. It is possible to have more than one CaptureSink. So there can be a CaptureSink simply saving out the HDR image to disk or a CaptureSink transferring the snapshot's data to a connected client via TCP/IP or any other means of communication.
	
#### Capture Proxy

Capturing is internally handled by DeepDriveCapture which is a singleton. DeepDriveCapture is not directly exposed and thus not accessible at least not from UE4's Blueprint system. For that purpose DeepDriveCaptureProxy exists which is an Unreal Engine actor. It exposes some properties such as capture rate to Blueprint. There must be one DeepDriveCaptureProxy actor placed in the scene for capturing to work and if more than one DeepDriveCaptureProxy is placed into the scene (which can not be prevented) the actor on which AActor::PreInitializeComponents() gets called first wins and will become the master DeepDriveCaptureProxy. 

- detailed description of capture proxy


#### BMP Disk capture sink

#### Shared memory capture sink


### Control

