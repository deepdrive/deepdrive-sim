# Deepdrive Sim Docs

## Setup

[Development setup instructions](/docs/setup)

## Requirements

- Linux
- Python 3.6+
- 20GB disk space
- 8GB RAM

NOTE: Windows support has been removed due to a mismatch of resources and interest. The main impediment to Windows working is the lack of an embedded python interpreter and issues installing the correct packages from uepy-requirements.txt. 


## Development Tips

### VERSION file

This is used for checking compatibility between sim and agents and will update automatically after building in Unreal. Please check it in. Also if you make a backwards incompatible change, i.e. to the shared memory interface, bump the minor version in the [MAJOR_MINOR_VERSION](Content/Data/MAJOR_MINOR_VERSION) file. 



### Clean builds

You'll often want to run `clean.sh` or `clean.bat` after pulling in changes, especially to the plugin as Unreal will spuriously cache old binaries.

### PyCharm

If you open an Unreal project in Pycharm, add Binaries, Build, Content, Intermediate, and Saved to your project’s “Excluded” directories in Project Structure or simply by right clicking and choosing “Mark Directory as” => “Excluded”. Keeping these large binary directories in the project will cause PyCharm to index them. Do the same with these directories (Binaries, Build, Content, Intermediate, and Saved) within any of the Plugins in the Plugins folder.


Windows - Package via the Unreal Editor

### Setting key binds

Unreal->Project Settings->Input->Action Mappings OR in Blueprints->Find (uncheck Find in Current Blueprint Only) and search for the input key, i.e. J.

