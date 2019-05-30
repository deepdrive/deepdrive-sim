# Setup

- [Associate your GitHub username with your Unreal account](https://www.unrealengine.com/en-US/ue4-on-github) to get access to Unreal sources on GitHub. 

## Set your python bin

This is where the Python extension is installed which enables communication between Unreal and Python.

### Option 1: Automatically set the python bin 

Setup the [deepdrive](https://github.com/deepdrive/deepdrive) project which creates your `~/.deepdrive/python_bin`

### Option 2: Manually set the python bin 

Create `~/.deepdrive/python_bin` on Unix or `%HOMEPATH%\.deepdrive\python_bin` on Windows and point it to the Python executable you want the deepdrive Python extension to be installed to, i.e. 

```
/home/[YOU]/.local/share/virtualenvs/my-env/bin/python
```



## OS specific setup

To finish your development setup, follow one of the following guides per your OS.

- [Linux](docs/development/linux/overview.md)
- [Windows](docs/development/windows/overview.md)
