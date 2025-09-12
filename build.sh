glslangValidator -V --target-env vulkan1.0 -o loiacono.comp.spv loiacono.comp

# Build executable
g++ -std=c++17 -O2 -Wall -Wextra -I. -I./Vulkan-Headers/include -g main.cpp audio.cpp vulkan_helpers.cpp -lvulkan -o audio_processor

# Build shared library (.so)
g++ -std=c++17 -O2 -Wall -Wextra -fPIC -I. -I./Vulkan-Headers/include -g -shared main.cpp audio.cpp vulkan_helpers.cpp -lvulkan -o libaudio_processor.so
