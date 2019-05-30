# Vulkan

With Unreal 4.21, Vulkan is automatically supported. I've tried it with NVIDIA 384 drivers and things crash, but newer drivers may work.

If you experience crashes and see mentions to Vulkan in the logs, you can ensure OpenGL is used by uinstalling these debian packages

```
sudo apt remove libvulkan1 mesa-vulkan-drivers vulkan-utils
```