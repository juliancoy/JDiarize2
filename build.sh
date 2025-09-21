glslangValidator -V --target-env vulkan1.0 -o magnitude.comp.spv magnitude.comp
glslangValidator -V --target-env vulkan1.2 prefix_sum_real.comp -o prefix_sum_real.comp.spv
glslangValidator -V --target-env vulkan1.2 prefix_sum_imag.comp -o prefix_sum_imag.comp.spv
glslangValidator -V --target-env vulkan1.2 comb_filter.comp -o comb_filter.comp.spv
# Build executable
#g++ -std=c++17 -O2 -Wall -Wextra -I. -I./Vulkan-Headers/include -g main.cpp audio.cpp vulkan_helpers.cpp -lvulkan -o audio_processor

# Build shared library (.so)
echo "Building shared library..."
g++ -std=c++17 -O2 -Wall -Wextra -fPIC -I. -I./Vulkan-Headers/include -g -shared main.cpp audio.cpp vulkan_helpers.cpp -lvulkan -o libaudio_processor.so

echo "Done."
