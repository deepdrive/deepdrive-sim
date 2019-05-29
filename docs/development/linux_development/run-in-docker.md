# Running the Unreal Editor from a Docker container

## Contents

- [Prerequisites](#prerequisites)
- [General instructions](#general-instructions)
  - [Step 1: Obtaining the Docker image](#step-1-obtaining-the-docker-image)
  - [Step 2: Starting the container](#step-2-starting-the-container)
  - [Step 3: Building the project](#step-3-building-the-project)
  - [Step 4: Running the Editor](#step-4-running-the-editor)
  - [Step 5: (Optional) Running other commands inside the container](#step-5-optional-running-other-commands-inside-the-container)
  - [Step 6: Stopping the container](#step-6-stopping-the-container)
- [Deepdrive-specific instructions](#deepdrive-specific-instructions)
  - [Additional prerequisites](#additional-prerequisites)
  - [Obtaining the Deepdrive Docker image](#obtaining-the-deepdrive-docker-image)
  - [Additional arguments for Step 2](#additional-arguments-for-step-2)


## Prerequisites

In order to run the Unreal Editor from within a Docker container, you will need the following:

- A Linux distribution with X11 (and optionally PulseAudio if audio output is required)
- The official NVIDIA GPU drivers (open source drivers such as Nouveau will not work)
- [Docker](https://www.docker.com/)
- The [NVIDIA Docker runtime](https://github.com/NVIDIA/nvidia-docker)


## General instructions

### Step 1: Obtaining the Docker image

Before you can start working with an Unreal Editor container, you will first need to build or pull a suitable container image. Voyage employees will be working with the Deepdrive Docker image, which is documented in the section [Obtaining the Deepdrive Docker image](#obtaining-the-deepdrive-docker-image).

### Step 2: Starting the container

To start a container from which the Unreal Editor can be run, open a terminal window in the directory containing your Unreal project's source code and execute the following command (**$IMAGE_NAME** should be replaced with the name of the Docker image that you obtained in Step 1):

```bash
docker run --rm -ti --runtime=nvidia --name=unreal_editor "-v$HOME/.config/Epic:/home/ue4/.config/Epic" -v`pwd`:/projectdir -w /projectdir -v/tmp/.X11-unix:/tmp/.X11-unix:rw -e DISPLAY -v/run/user/$UID/pulse:/run/user/1000/pulse $IMAGE_NAME bash
```

The meaning of each of the arguments to the `docker run` command is as follows:

- `--rm` ensures the container will be deleted after it has finished running. This ensures you have a clean environment to work from each time you start the image. It is important to note that no data will be lost when the container is deleted, since the project source directory and the Unreal Editor's configuration directory are both bind-mounted from the host system's filesystem.

- `-ti` runs the container in interactive mode, which allows you to run commands from the bash shell inside the container in much the same manner as you would on the host system.

- `--runtime=nvidia` instructs Docker to use the NVIDIA Docker runtime instead of the default Docker runtime. This allows the container to access the GPU resources of the host system.

- `--name=unreal_editor` assigns the identifier "unreal_editor" to the container. This is useful if you need to run other Docker commands that manipulate or interact with the container while it is running.

- `-v$HOME/.config/Epic:/home/ue4/.config/Epic` bind-mounts the host filesystem directory `~/.config/Epic` into the container. This directory is where the Unreal Editor stores its configuration data and compiled shader cache. Storing this data on the host filesystem allows the Unreal Editor to remember information from previous runs and ensures it behaves in a manner more consistent with running the Editor directly on the host system.

- ``-v`pwd`:/projectdir`` bind-mounts the current working directory into the container. This allows the Unreal Editor to access your Unreal project's source code.

- `-w /projectdir` sets the default working directory of the container to the directory containing your Unreal project's source code.

- `-v/tmp/.X11-unix:/tmp/.X11-unix:rw` bind-mounts the host system's X11 socket into the container. This allows the Unreal Editor to create graphical windows that will be displayed on the host system.

- `-e DISPLAY` propagates the value of the `DISPLAY` environment variable from the host system to the container. This is needed for correct X11 display behaviour.

- `-v/run/user/$UID/pulse:/run/user/1000/pulse` bind-mounts the host system's PulseAudio socket into the container. This is only necessary if audio output is required.

- `$IMAGE_NAME` is a placeholder that should be replaced with the name of the Docker image that you obtained in Step 1.

- `bash` starts an interactive bash shell within the container.

### Step 3: Building the project

Unless you are working with a Blueprint-only Unreal project, you will need to compile your project's C++ source code before you can run the Unreal Editor. You can do this by running the following command inside the container's interactive bash shell:

```bash
ue4 build
```

### Step 4: Running the Editor

Once the source code for your Unreal project has been compiled, you can start the Unreal Editor by running the following command inside the container's interactive bash shell:

```bash
ue4 run
```

If everything is working correctly, you should see the Unreal Editor splash screen appear while the Editor is loading, followed by the window for the Editor itself.

### Step 5: (Optional) Running other commands inside the container

If you would like to run additional commands inside the container while the Unreal Editor is running, you can attach additional interactive bash shells to the container by running the following command:

```
docker exec -ti unreal_editor bash
```

This command can be run multiple times to attach multiple interactive shells. It is important to note that all additional interactive shells will be terminated when the container is stopped, which happens the moment that the original bash shell (the one created when the container was started) exits. Closing any of the additional attached shells will not stop the container.

### Step 6: Stopping the container

Once you have closed the Unreal Editor and returned to the container's interactive bash shell, you can stop the container by simply closing the bash shell:

```bash
exit
```

Note that this will also terminate any additional interactive shells that you may have attached to the container while it was running.


## Deepdrive-specific instructions

### Additional prerequisites

Because the Docker image for Deepdrive contains CUDA 10.0, the version number of the NVIDIA GPU driver matters. You will need to ensure you have a [version of the NVIDIA GPU driver compatible with CUDA 10](https://github.com/NVIDIA/nvidia-docker/wiki/CUDA#requirements).

### Obtaining the Deepdrive Docker image

Before you can pull the Docker image for Deepdrive, you will need to login to Docker Hub:

```bash
# This will prompt for your Docker ID and password
docker login
```

Once you have authenticated, the Docker image for Deepdrive can then be pulled by executing the following command:

```bash
docker pull deepdriveio/ue4-deepdrive-deps:latest
```

If you encounter permission issues when pulling the image then you may need to contact Craig Quiter to have your Docker Hub account added to the [deepdriveio organisation](https://hub.docker.com/u/deepdriveio/).

### Additional arguments for Step 2

If you intend to run the Deepdrive deep learning agents inside the container, you will need to bind-mount the directory containing the [Deepdrive source code](https://github.com/deepdrive/deepdrive) in addition to the source code for the Deepdrive simulator Unreal project:

```bash
# These are the original arguments specified in Step 2:
-v`pwd`:/projectdir -w /projectdir

# Replace them with these arguments:
-v`pwd`:/simdir -w /simdir -v/path/to/deepdrive:/deepdrivedir
```

Replace `/path/to/deepdrive` with the appropriate host filesystem path for the directory containing the Deepdrive source code.
