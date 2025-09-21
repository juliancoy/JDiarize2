// main.cpp
// Single-file version combining main and audio processing utilities.
// Build example (adjust include/library paths as needed):
// g++ -std=c++17 main.cpp -I. -lvulkan -o vulkan_audio_processor
#include "main.h"

void cleanupVulkan(VkInstance instance, VkDevice device)
{
    if (device != VK_NULL_HANDLE)
    {
        vkDestroyDevice(device, nullptr);
    }
    if (instance != VK_NULL_HANDLE)
    {
        vkDestroyInstance(instance, nullptr);
    }
}

// C-compatible interface for Python FFI
typedef struct
{
    void *data;
    size_t size;
} FloatVector;

// C-compatible wrapper function
extern "C" int loiacono(FloatVector *audioData, FloatVector *magnitudeData, FloatVector *frequencies, float multiple)
{
    // Convert FloatVector to std::vector
    if (!audioData || !magnitudeData || !frequencies)
    {
        return 1; // Error: null pointers
    }

    std::vector<float> audioVec(static_cast<float *>(audioData->data),
                                static_cast<float *>(audioData->data) + audioData->size);
    std::vector<float> outputVec(static_cast<float *>(magnitudeData->data),
                                 static_cast<float *>(magnitudeData->data) + magnitudeData->size);
    std::vector<float> freqVec(static_cast<float *>(frequencies->data),
                               static_cast<float *>(frequencies->data) + frequencies->size);

    // Call the C++ implementation
    int result = loiacono(&audioVec,  &outputVec, &freqVec, multiple);

    // Copy results back to magnitudeData (if the C++ function modified the vectors)
    if (result == 0)
    {
        // Ensure output vectors have the same size as expected
        if (outputVec.size() == magnitudeData->size)
        {
            std::memcpy(magnitudeData->data, outputVec.data(), outputVec.size() * sizeof(float));
        }
        else
        {
            return 2; // Error: size mismatch
        }
    }

    return result;
}

// ---------- Main ----------

int main()
{
    std::cout << "Vulkan Audio Processor" << std::endl;
    std::cout << "======================" << std::endl;

    // Read audio data from WAV file
    std::vector<float> audioData;
    uint32_t sampleRate = 0;
    uint16_t numChannels = 0;
    float multiple = 10;
    if (!readAudioData("audio.wav", audioData, sampleRate, numChannels))
    {
        std::cerr << "Failed to read audio data" << std::endl;
        return 1;
    }
    std::cout << "Read " << audioData.size() << " audio samples" << std::endl;

    // Create output vector for processed results
    std::vector<float> magnitudeData(audioData.size());
    // Use default frequency of 440 Hz
    std::vector<float> frequencies = {440.0f};
    loiacono(&audioData, &magnitudeData, &frequencies, multiple);
}

int loiacono(std::vector<float> *audioData, std::vector<float> *outputData,
             std::vector<float> *frequencies, float multiple)
{
    // Initialize Vulkan (returns compute queue family and the queue handle)
    VkInstance instance = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    uint32_t computeQueueFamilyIndex = 0;
    VkQueue computeQueue = VK_NULL_HANDLE;

    // Create Vulkan instance
    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Audio Processor";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_0;

    // ---- Added: Enable validation layer & debug utils extension if available ----
    std::vector<const char *> enabledLayers;
    std::vector<const char *> enabledExtensions;

    const char *kValidationLayer = "VK_LAYER_KHRONOS_validation";
    if (isLayerAvailable(kValidationLayer))
    {
        enabledLayers.push_back(kValidationLayer);
        std::cout << "Enabling layer: " << kValidationLayer << std::endl;
    }
    else
    {
        std::cout << "Validation layer not available; continuing without it." << std::endl;
    }

    if (isExtensionAvailable(VK_EXT_DEBUG_UTILS_EXTENSION_NAME))
    {
        enabledExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        std::cout << "Enabling extension: " << VK_EXT_DEBUG_UTILS_EXTENSION_NAME << std::endl;
    }
    else
    {
        std::cout << "VK_EXT_debug_utils not available; continuing without it." << std::endl;
    }
    // ---------------------------------------------------------------------------

    VkInstanceCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    // ---- Added: hook up layers & extensions ----
    createInfo.enabledLayerCount = static_cast<uint32_t>(enabledLayers.size());
    createInfo.ppEnabledLayerNames = enabledLayers.empty() ? nullptr : enabledLayers.data();
    createInfo.enabledExtensionCount = static_cast<uint32_t>(enabledExtensions.size());
    createInfo.ppEnabledExtensionNames = enabledExtensions.empty() ? nullptr : enabledExtensions.data();

    // If debug utils is enabled, chain a messenger create info for early validation output
    VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
    if (!enabledExtensions.empty())
    {
        for (auto *ext : enabledExtensions)
        {
            if (std::strcmp(ext, VK_EXT_DEBUG_UTILS_EXTENSION_NAME) == 0)
            {
                debugCreateInfo = {};
                debugCreateInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
                debugCreateInfo.messageSeverity =
                    VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                    VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
                debugCreateInfo.messageType =
                    VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                    VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                    VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
                debugCreateInfo.pfnUserCallback = DebugCallback;
                createInfo.pNext = &debugCreateInfo; // minimal change: optional pNext
                break;
            }
        }
    }
    // -------------------------------------------

    VK_CHECK(vkCreateInstance(&createInfo, nullptr, &instance));

    // Create the debug messenger after instance creation (if extension enabled)
    if (!enabledExtensions.empty())
    {
        for (auto *ext : enabledExtensions)
        {
            if (std::strcmp(ext, VK_EXT_DEBUG_UTILS_EXTENSION_NAME) == 0)
            {
                VkDebugUtilsMessengerCreateInfoEXT ci = {};
                ci.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
                ci.messageSeverity =
                    VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                    VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
                ci.messageType =
                    VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                    VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                    VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
                ci.pfnUserCallback = DebugCallback;
                if (CreateDebugUtilsMessengerEXT(instance, &ci, nullptr, &g_debugMessenger) != VK_SUCCESS)
                {
                    std::cerr << "Failed to create debug messenger (continuing)" << std::endl;
                }
                break;
            }
        }
    }

    // Enumerate physical devices
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    if (deviceCount == 0)
    {
        std::cerr << "No Vulkan devices found" << std::endl;
        // Destroy debug messenger (if created) before destroying instance
        if (g_debugMessenger != VK_NULL_HANDLE)
        {
            DestroyDebugUtilsMessengerEXT(instance, g_debugMessenger, nullptr);
            g_debugMessenger = VK_NULL_HANDLE;
        }
        return VK_ERROR_DEVICE_LOST;
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

    // Choose first device with a compute queue
    physicalDevice = VK_NULL_HANDLE;
    for (auto pd : devices)
    {
        if (findComputeQueueFamily(pd, computeQueueFamilyIndex))
        {
            physicalDevice = pd;
            break;
        }
    }
    if (physicalDevice == VK_NULL_HANDLE)
    {
        std::cerr << "No compute-capable Vulkan device found" << std::endl;
        if (g_debugMessenger != VK_NULL_HANDLE)
        {
            DestroyDebugUtilsMessengerEXT(instance, g_debugMessenger, nullptr);
            g_debugMessenger = VK_NULL_HANDLE;
        }
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    // Create logical device with the compute-capable queue
    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueCreateInfo = {};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = computeQueueFamilyIndex;
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &queuePriority;

    VkDeviceCreateInfo deviceCreateInfo = {};
    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.queueCreateInfoCount = 1;
    deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
    // (No device extensions required for this compute-only sample)

    VK_CHECK(vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &device));

    // Retrieve the queue from that same family
    vkGetDeviceQueue(device, computeQueueFamilyIndex, 0, &computeQueue);
    if (computeQueue == VK_NULL_HANDLE)
    {
        std::cerr << "Failed to get compute device queue" << std::endl;
        if (g_debugMessenger != VK_NULL_HANDLE)
        {
            DestroyDebugUtilsMessengerEXT(instance, g_debugMessenger, nullptr);
            g_debugMessenger = VK_NULL_HANDLE;
        }
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    // Create compute pipelines
    VkPipeline magnitudePipeline = VK_NULL_HANDLE;
    VkPipeline prefixSumRealPipeline = VK_NULL_HANDLE;
    VkPipeline prefixSumImagPipeline = VK_NULL_HANDLE;
    VkPipeline combFilterPipeline = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkShaderModule magnitudeShaderModule = VK_NULL_HANDLE;
    VkShaderModule prefixSumRealShaderModule = VK_NULL_HANDLE;
    VkShaderModule prefixSumImagShaderModule = VK_NULL_HANDLE;
    VkShaderModule combFilterShaderModule = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;

    std::cout << "Device: " << device << std::endl;

    // Descriptor set layout (set = 0, bindings 0, 1, 2, 3, and 4)
    VkDescriptorSetLayoutBinding bindings[7] = {};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    bindings[2].binding = 2;
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    bindings[3].binding = 3;
    bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[3].descriptorCount = 1;
    bindings[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    bindings[4].binding = 4;
    bindings[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[4].descriptorCount = 1;
    bindings[4].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    bindings[5].binding = 5;
    bindings[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[5].descriptorCount = 1;
    bindings[5].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    bindings[6].binding = 6;
    bindings[6].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[6].descriptorCount = 1;
    bindings[6].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 7;
    layoutInfo.pBindings = bindings;

    VkResult result = vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout);
    std::cout << "vkCreateDescriptorSetLayout result: " << result << std::endl;
    VK_CHECK(result);

    // Pipeline layout + push constants
    VkPushConstantRange pcRange = {};
    pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcRange.offset = 0;
    pcRange.size = sizeof(MagnitudePushConstants);

    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pcRange;

    result = vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout);
    std::cout << "vkCreatePipelineLayout result: " << result << std::endl;
    VK_CHECK(result);

    // Load SPIR-V and create shader module
    std::vector<uint32_t> spirv;
    result = loadSpirvFile("magnitude.comp.spv", spirv);
    if (result != VK_SUCCESS)
    {
        std::cerr << "Failed to load SPIR-V for compute shader" << std::endl;
        if (g_debugMessenger != VK_NULL_HANDLE)
        {
            DestroyDebugUtilsMessengerEXT(instance, g_debugMessenger, nullptr);
            g_debugMessenger = VK_NULL_HANDLE;
        }
        return result;
    }

    // Load real prefix sum SPIR-V
    std::vector<uint32_t> prefixRealSpv;
    VK_CHECK(loadSpirvFile("prefix_sum_real.comp.spv", prefixRealSpv));

    VkShaderModuleCreateInfo prefix_real_smci{};
    prefix_real_smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    prefix_real_smci.codeSize = prefixRealSpv.size() * sizeof(uint32_t);
    prefix_real_smci.pCode = prefixRealSpv.data();
    VK_CHECK(vkCreateShaderModule(device, &prefix_real_smci, nullptr, &prefixSumRealShaderModule));

    VkPipelineShaderStageCreateInfo prefixRealStage{};
    prefixRealStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    prefixRealStage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    prefixRealStage.module = prefixSumRealShaderModule;
    prefixRealStage.pName = "main";

    VkComputePipelineCreateInfo prefixRealPipeInfo{};
    prefixRealPipeInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    prefixRealPipeInfo.stage = prefixRealStage;
    prefixRealPipeInfo.layout = pipelineLayout; // same layout if bindings match
    VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &prefixRealPipeInfo, nullptr, &prefixSumRealPipeline));

    // Load imaginary prefix sum SPIR-V
    std::vector<uint32_t> prefixImagSpv;
    VK_CHECK(loadSpirvFile("prefix_sum_imag.comp.spv", prefixImagSpv));

    VkShaderModuleCreateInfo prefix_imag_smci{};
    prefix_imag_smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    prefix_imag_smci.codeSize = prefixImagSpv.size() * sizeof(uint32_t);
    prefix_imag_smci.pCode = prefixImagSpv.data();
    VK_CHECK(vkCreateShaderModule(device, &prefix_imag_smci, nullptr, &prefixSumImagShaderModule));

    VkPipelineShaderStageCreateInfo prefixImagStage{};
    prefixImagStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    prefixImagStage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    prefixImagStage.module = prefixSumImagShaderModule;
    prefixImagStage.pName = "main";

    VkComputePipelineCreateInfo prefixImagPipeInfo{};
    prefixImagPipeInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    prefixImagPipeInfo.stage = prefixImagStage;
    prefixImagPipeInfo.layout = pipelineLayout; // same layout if bindings match
    VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &prefixImagPipeInfo, nullptr, &prefixSumImagPipeline));

    // Load comb filter SPIR-V
    std::vector<uint32_t> combSpv;
    VK_CHECK(loadSpirvFile("comb_filter.comp.spv", combSpv));

    VkShaderModuleCreateInfo comb_smci{};
    comb_smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    comb_smci.codeSize = combSpv.size() * sizeof(uint32_t);
    comb_smci.pCode = combSpv.data();
    VK_CHECK(vkCreateShaderModule(device, &comb_smci, nullptr, &combFilterShaderModule));

    VkPipelineShaderStageCreateInfo combStage{};
    combStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    combStage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    combStage.module = combFilterShaderModule;
    combStage.pName = "main";

    VkComputePipelineCreateInfo combPipeInfo{};
    combPipeInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    combPipeInfo.stage = combStage;
    combPipeInfo.layout = pipelineLayout; // same layout if bindings match
    VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &combPipeInfo, nullptr, &combFilterPipeline));

    VkShaderModuleCreateInfo magnitude_smci = {};
    magnitude_smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    magnitude_smci.codeSize = spirv.size() * sizeof(uint32_t);
    magnitude_smci.pCode = spirv.data();

    result = vkCreateShaderModule(device, &magnitude_smci, nullptr, &magnitudeShaderModule);
    VK_CHECK(result);

    VkPipelineShaderStageCreateInfo magnitudeShaderStageInfo = {};
    magnitudeShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    magnitudeShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    magnitudeShaderStageInfo.module = magnitudeShaderModule;
    magnitudeShaderStageInfo.pName = "main";

    VkComputePipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = magnitudeShaderStageInfo;
    pipelineInfo.layout = pipelineLayout;

    result = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &magnitudePipeline);
    if (result != VK_SUCCESS)
    {
        std::cerr << "Failed to create compute pipeline" << std::endl;
        if (g_debugMessenger != VK_NULL_HANDLE)
        {
            DestroyDebugUtilsMessengerEXT(instance, g_debugMessenger, nullptr);
            g_debugMessenger = VK_NULL_HANDLE;
        }
        return result;
    }

    // Descriptor pool
    VkDescriptorPoolSize poolSizes[1] = {};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[0].descriptorCount = 7; // five storage buffers total

    VkDescriptorPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = poolSizes;
    poolInfo.maxSets = 1;

    result = vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool);
    VK_CHECK(result);

    // Allocate descriptor set
    VkDescriptorSetAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &descriptorSetLayout;

    result = vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet);
    VK_CHECK(result);

    // Command pool & buffer — use the SAME compute queue family index
    VkCommandPoolCreateInfo cmdPoolInfo = {};
    cmdPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cmdPoolInfo.queueFamilyIndex = computeQueueFamilyIndex;

    result = vkCreateCommandPool(device, &cmdPoolInfo, nullptr, &commandPool);
    VK_CHECK(result);

    VkCommandBufferAllocateInfo cmdBufferInfo = {};
    cmdBufferInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdBufferInfo.commandPool = commandPool;
    cmdBufferInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdBufferInfo.commandBufferCount = 1;

    result = vkAllocateCommandBuffers(device, &cmdBufferInfo, &commandBuffer);

    if (result != VK_SUCCESS)
    {
        if (g_debugMessenger != VK_NULL_HANDLE)
        {
            DestroyDebugUtilsMessengerEXT(instance, g_debugMessenger, nullptr);
            g_debugMessenger = VK_NULL_HANDLE;
        }
        cleanupVulkan(instance, device);
        return 1;
    }

    // Create buffers for audio data
    VkBuffer signalBuffer = VK_NULL_HANDLE;
    VkBuffer realBuffer = VK_NULL_HANDLE;
    VkBuffer imagBuffer = VK_NULL_HANDLE;
    VkBuffer realPrefixSumBuffer = VK_NULL_HANDLE;
    VkBuffer imagPrefixSumBuffer = VK_NULL_HANDLE;
    VkBuffer combFilterBuffer = VK_NULL_HANDLE;
    VkDeviceMemory signalBufferMemory = VK_NULL_HANDLE;
    VkDeviceMemory realBufferMemory = VK_NULL_HANDLE;
    VkDeviceMemory imagBufferMemory = VK_NULL_HANDLE;
    VkDeviceMemory realPrefixSumBufferMemory = VK_NULL_HANDLE;
    VkDeviceMemory imagPrefixSumBufferMemory = VK_NULL_HANDLE;
    VkDeviceMemory combFilterBufferMemory = VK_NULL_HANDLE;

    // Create input buffer
    VkBufferCreateInfo signalBufferCreateInfo = {};
    signalBufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    signalBufferCreateInfo.size = audioData->size() * sizeof(float);
    signalBufferCreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    signalBufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    std::cout << "Input buffer size: " << signalBufferCreateInfo.size << " bytes" << std::endl;

    // Create output buffer - large enough for all frequencies
    VkBufferCreateInfo signalByFreqCountCreateInfo = {};
    signalByFreqCountCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;

    signalByFreqCountCreateInfo.size = audioData->size() * frequencies->size() * sizeof(float);
    const VkDeviceSize signalByFreqFloatSizeBytes = signalByFreqCountCreateInfo.size;
    std::cout << "Output buffer size: " << signalByFreqCountCreateInfo.size << " bytes" << std::endl;

    signalByFreqCountCreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    signalByFreqCountCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VK_CHECK(vkCreateBuffer(device, &signalBufferCreateInfo, nullptr, &signalBuffer));
    VK_CHECK(vkCreateBuffer(device, &signalByFreqCountCreateInfo, nullptr, &realBuffer));
    VK_CHECK(vkCreateBuffer(device, &signalByFreqCountCreateInfo, nullptr, &imagBuffer));

    // Allocate memory for input buffer
    VkMemoryRequirements signalBufferRequirements{};
    vkGetBufferMemoryRequirements(device, signalBuffer, &signalBufferRequirements);

    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

    auto findHostVisibleCoherent = [&](uint32_t typeBits) -> uint32_t
    {
        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
        {
            if ((typeBits & (1 << i)) &&
                (memProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
                (memProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))
            {
                return i;
            }
        }
        std::cerr << "Failed to find suitable memory type for input" << std::endl;

        if (g_debugMessenger != VK_NULL_HANDLE)
        {
            DestroyDebugUtilsMessengerEXT(instance, g_debugMessenger, nullptr);
            g_debugMessenger = VK_NULL_HANDLE;
        }
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    };

    uint32_t visibleCoherentMemoryIndex = findHostVisibleCoherent(signalBufferRequirements.memoryTypeBits);

    VkMemoryAllocateInfo signalBufferAllocInfo = {};
    signalBufferAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    signalBufferAllocInfo.allocationSize = signalBufferRequirements.size;
    signalBufferAllocInfo.memoryTypeIndex = visibleCoherentMemoryIndex;

    VK_CHECK(vkAllocateMemory(device, &signalBufferAllocInfo, nullptr, &signalBufferMemory));
    VK_CHECK(vkBindBufferMemory(device, signalBuffer, signalBufferMemory, 0));

    // Allocate memory for output buffer
    VkMemoryRequirements memRequirementsOut{};
    vkGetBufferMemoryRequirements(device, realBuffer, &memRequirementsOut);

    uint32_t memoryTypeIndexOut = findHostVisibleCoherent(memRequirementsOut.memoryTypeBits);

    VkMemoryAllocateInfo allocInfoOut = {};
    allocInfoOut.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfoOut.allocationSize = memRequirementsOut.size;
    allocInfoOut.memoryTypeIndex = memoryTypeIndexOut;

    VK_CHECK(vkAllocateMemory(device, &allocInfoOut, nullptr, &realBufferMemory));
    VK_CHECK(vkBindBufferMemory(device, realBuffer, realBufferMemory, 0));
    VK_CHECK(vkAllocateMemory(device, &allocInfoOut, nullptr, &imagBufferMemory));
    VK_CHECK(vkBindBufferMemory(device, imagBuffer, imagBufferMemory, 0));

    // Create frequency buffer (one frequency per workgroup)
    VkBuffer freqBuffer = VK_NULL_HANDLE;
    VkDeviceMemory freqBufferMemory = VK_NULL_HANDLE;

    const uint32_t frequencyCount = static_cast<uint32_t>(frequencies->size());

    VkBufferCreateInfo bufferInfoFreq = {};
    bufferInfoFreq.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfoFreq.size = frequencyCount * sizeof(float);
    bufferInfoFreq.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT; // readonly in shader, still STORAGE_BUFFER
    bufferInfoFreq.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VK_CHECK(vkCreateBuffer(device, &bufferInfoFreq, nullptr, &freqBuffer));

    VkMemoryRequirements memRequirements{};
    vkGetBufferMemoryRequirements(device, freqBuffer, &memRequirements);

    uint32_t memoryTypeIndex = UINT32_MAX;
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
    {
        if ((memRequirements.memoryTypeBits & (1 << i)) &&
            (memProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
            (memProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))
        {
            memoryTypeIndex = i;
            break;
        }
    }

    if (memoryTypeIndex == UINT32_MAX)
    {
        std::cerr << "Failed to find suitable memory type for frequency buffer" << std::endl;
        if (g_debugMessenger != VK_NULL_HANDLE)
        {
            DestroyDebugUtilsMessengerEXT(instance, g_debugMessenger, nullptr);
            g_debugMessenger = VK_NULL_HANDLE;
        }
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }

    VkMemoryAllocateInfo allocInfoFrequency = {};
    allocInfoFrequency.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfoFrequency.allocationSize = memRequirements.size;
    allocInfoFrequency.memoryTypeIndex = memoryTypeIndex;

    VK_CHECK(vkAllocateMemory(device, &allocInfoFrequency, nullptr, &freqBufferMemory));
    VK_CHECK(vkBindBufferMemory(device, freqBuffer, freqBufferMemory, 0));

    // Create prefix sum output buffer
    VkBufferCreateInfo prefixSumBufferCreateInfo = {};
    prefixSumBufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    prefixSumBufferCreateInfo.size = audioData->size() * frequencies->size() * sizeof(float);
    prefixSumBufferCreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    prefixSumBufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    const VkDeviceSize prefixSumBufferSizeBytes = prefixSumBufferCreateInfo.size;
    std::cout << "Prefix sum buffer size: " << prefixSumBufferSizeBytes << " bytes" << std::endl;

    VK_CHECK(vkCreateBuffer(device, &prefixSumBufferCreateInfo, nullptr, &realPrefixSumBuffer));
    VK_CHECK(vkCreateBuffer(device, &prefixSumBufferCreateInfo, nullptr, &imagPrefixSumBuffer));

    // Allocate memory for prefix sum buffer
    VkMemoryRequirements memRequirementsPrefix{};
    vkGetBufferMemoryRequirements(device, realPrefixSumBuffer, &memRequirementsPrefix);

    uint32_t memoryTypeIndexPrefix = findHostVisibleCoherent(memRequirementsPrefix.memoryTypeBits);

    VkMemoryAllocateInfo allocInfoPrefix = {};
    allocInfoPrefix.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfoPrefix.allocationSize = memRequirementsPrefix.size;
    allocInfoPrefix.memoryTypeIndex = memoryTypeIndexPrefix;

    VK_CHECK(vkAllocateMemory(device, &allocInfoPrefix, nullptr, &realPrefixSumBufferMemory));
    VK_CHECK(vkBindBufferMemory(device, realPrefixSumBuffer, realPrefixSumBufferMemory, 0));

    VK_CHECK(vkAllocateMemory(device, &allocInfoPrefix, nullptr, &imagPrefixSumBufferMemory));
    VK_CHECK(vkBindBufferMemory(device, imagPrefixSumBuffer, imagPrefixSumBufferMemory, 0));

    // Create comb filter output buffer
    VkBufferCreateInfo combFilterBufferCreateInfo = {};
    combFilterBufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    combFilterBufferCreateInfo.size = audioData->size() * frequencies->size() * sizeof(float);
    combFilterBufferCreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    combFilterBufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    const VkDeviceSize combFilterBufferSizeBytes = combFilterBufferCreateInfo.size;
    std::cout << "Comb filter buffer size: " << combFilterBufferSizeBytes << " bytes" << std::endl;

    VK_CHECK(vkCreateBuffer(device, &combFilterBufferCreateInfo, nullptr, &combFilterBuffer));

    // Allocate memory for comb filter buffer
    VkMemoryRequirements memRequirementsComb{};
    vkGetBufferMemoryRequirements(device, combFilterBuffer, &memRequirementsComb);

    uint32_t memoryTypeIndexComb = findHostVisibleCoherent(memRequirementsComb.memoryTypeBits);
    if (memoryTypeIndexComb == UINT32_MAX)
    {
        std::cerr << "Failed to find suitable memory type for comb filter buffer" << std::endl;
        if (g_debugMessenger != VK_NULL_HANDLE)
        {
            DestroyDebugUtilsMessengerEXT(instance, g_debugMessenger, nullptr);
            g_debugMessenger = VK_NULL_HANDLE;
        }
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }

    VkMemoryAllocateInfo allocInfoComb = {};
    allocInfoComb.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfoComb.allocationSize = memRequirementsComb.size;
    allocInfoComb.memoryTypeIndex = memoryTypeIndexComb;

    VK_CHECK(vkAllocateMemory(device, &allocInfoComb, nullptr, &combFilterBufferMemory));
    VK_CHECK(vkBindBufferMemory(device, combFilterBuffer, combFilterBufferMemory, 0));

    // Copy audio data to GPU (input)
    void *data;
    VK_CHECK(vkMapMemory(device, signalBufferMemory, 0, audioData->size() * sizeof(float), 0, &data));
    std::memcpy(data, audioData->data(), audioData->size() * sizeof(float));
    vkUnmapMemory(device, signalBufferMemory);

    // Copy frequency data to GPU
    VK_CHECK(vkMapMemory(device, freqBufferMemory, 0, frequencies->size() * sizeof(float), 0, &data));
    std::memcpy(data, frequencies->data(), frequencies->size() * sizeof(float));
    vkUnmapMemory(device, freqBufferMemory);

    // Update descriptor set
    VkDescriptorBufferInfo signalBufferInfo = {};
    signalBufferInfo.buffer = signalBuffer;
    signalBufferInfo.offset = 0;
    signalBufferInfo.range = VK_WHOLE_SIZE;

    VkDescriptorBufferInfo realBufferInfo = {};
    realBufferInfo.buffer = realBuffer;
    realBufferInfo.offset = 0;
    realBufferInfo.range = VK_WHOLE_SIZE;

    VkDescriptorBufferInfo imagBufferInfo = {};
    imagBufferInfo.buffer = imagBuffer;
    imagBufferInfo.offset = 0;
    imagBufferInfo.range = VK_WHOLE_SIZE;

    VkDescriptorBufferInfo freqBufferInfo = {};
    freqBufferInfo.buffer = freqBuffer;
    freqBufferInfo.offset = 0;
    freqBufferInfo.range = VK_WHOLE_SIZE;

    VkWriteDescriptorSet descriptorWrites[7] = {};
    descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[0].dstSet = descriptorSet;
    descriptorWrites[0].dstBinding = 0;
    descriptorWrites[0].dstArrayElement = 0;
    descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[0].descriptorCount = 1;
    descriptorWrites[0].pBufferInfo = &signalBufferInfo;

    descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[1].dstSet = descriptorSet;
    descriptorWrites[1].dstBinding = 1;
    descriptorWrites[1].dstArrayElement = 0;
    descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[1].descriptorCount = 1;
    descriptorWrites[1].pBufferInfo = &realBufferInfo;

    descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[2].dstSet = descriptorSet;
    descriptorWrites[2].dstBinding = 2;
    descriptorWrites[2].dstArrayElement = 0;
    descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[2].descriptorCount = 1;
    descriptorWrites[2].pBufferInfo = &freqBufferInfo;

    // Add descriptor for prefix sum output buffer (binding 3)
    VkDescriptorBufferInfo realPrefixSumBufferInfo = {};
    realPrefixSumBufferInfo.buffer = realPrefixSumBuffer;
    realPrefixSumBufferInfo.offset = 0;
    realPrefixSumBufferInfo.range = VK_WHOLE_SIZE;

    descriptorWrites[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[3].dstSet = descriptorSet;
    descriptorWrites[3].dstBinding = 3;
    descriptorWrites[3].dstArrayElement = 0;
    descriptorWrites[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[3].descriptorCount = 1;
    descriptorWrites[3].pBufferInfo = &realPrefixSumBufferInfo;

    // Add descriptor for comb filter output buffer (binding 4)
    VkDescriptorBufferInfo combFilterBufferInfo = {};
    combFilterBufferInfo.buffer = combFilterBuffer;
    combFilterBufferInfo.offset = 0;
    combFilterBufferInfo.range = VK_WHOLE_SIZE;

    descriptorWrites[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[4].dstSet = descriptorSet;
    descriptorWrites[4].dstBinding = 4;
    descriptorWrites[4].dstArrayElement = 0;
    descriptorWrites[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[4].descriptorCount = 1;
    descriptorWrites[4].pBufferInfo = &combFilterBufferInfo;

    descriptorWrites[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[5].dstSet = descriptorSet;
    descriptorWrites[5].dstBinding = 5;
    descriptorWrites[5].dstArrayElement = 0;
    descriptorWrites[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[5].descriptorCount = 1;
    descriptorWrites[5].pBufferInfo = &imagBufferInfo;

    // Add descriptor for prefix sum output buffer (binding 3)
    VkDescriptorBufferInfo imagPrefixSumBufferInfo = {};
    imagPrefixSumBufferInfo.buffer = imagPrefixSumBuffer;
    imagPrefixSumBufferInfo.offset = 0;
    imagPrefixSumBufferInfo.range = VK_WHOLE_SIZE;

    descriptorWrites[6].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[6].dstSet = descriptorSet;
    descriptorWrites[6].dstBinding = 6;
    descriptorWrites[6].dstArrayElement = 0;
    descriptorWrites[6].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[6].descriptorCount = 1;
    descriptorWrites[6].pBufferInfo = &imagPrefixSumBufferInfo;

    vkUpdateDescriptorSets(device, 6, descriptorWrites, 0, nullptr);

    std::cout << "Device: " << device << std::endl;
    std::cout << "Command Buffer: " << commandBuffer << std::endl;
    std::cout << "Pipeline: " << magnitudePipeline << std::endl;

    // Begin command buffer
    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    result = vkBeginCommandBuffer(commandBuffer, &beginInfo);
    VK_CHECK(result);

    // Pass 1: Magnitude
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, magnitudePipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);

    // Prepare push constants (Per-dispatch scalars go here)
    MagnitudePushConstants magnitudePC{};
    magnitudePC.startPos = 0u;
    magnitudePC.endPos = static_cast<uint32_t>(audioData->size());
    magnitudePC.multiple = multiple;

    vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(MagnitudePushConstants), &magnitudePC);
    vkCmdDispatch(commandBuffer, frequencyCount, 1, 1);

    // Barrier: make magnitude writes visible to prefix-sum reads
    VkMemoryBarrier toPrefix{};
    toPrefix.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    toPrefix.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    toPrefix.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(
        commandBuffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        1, &toPrefix,
        0, nullptr,
        0, nullptr);

    // Pass 2: Real and Imaginary prefix sum (run simultaneously)
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, prefixSumRealPipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);

    PrefixSumPushConstants prefixsumPC{};
    prefixsumPC.startPos = 0u;
    prefixsumPC.endPos = static_cast<uint32_t>(audioData->size());
    prefixsumPC.combSize = 800;

    vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(MagnitudePushConstants), &prefixsumPC);
    vkCmdDispatch(commandBuffer, frequencyCount, 1, 1);

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, prefixSumImagPipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(MagnitudePushConstants), &prefixsumPC);
    vkCmdDispatch(commandBuffer, frequencyCount, 1, 1);

    // Barrier: make prefix-sum writes visible to comb filter reads
    VkMemoryBarrier toComb{};
    toComb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    toComb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    toComb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(
        commandBuffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        1, &toComb,
        0, nullptr,
        0, nullptr);

    // Pass 3: Comb filter
    CombFilterPushConstants combPC{};
    combPC.startPos = 0u;
    combPC.endPos = static_cast<uint32_t>(audioData->size());
    combPC.multiple = multiple;

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, combFilterPipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(CombFilterPushConstants), &combPC);
    vkCmdDispatch(commandBuffer, frequencyCount, 1, 1);

    // Barrier: make comb filter writes visible to host
    VkMemoryBarrier toHost{};
    toHost.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    toHost.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    toHost.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(
        commandBuffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_HOST_BIT,
        0,
        1, &toHost,
        0, nullptr,
        0, nullptr);

    VK_CHECK(vkEndCommandBuffer(commandBuffer));

    // Submit to the provided compute queue
    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    result = vkQueueSubmit(computeQueue, 1, &submitInfo, VK_NULL_HANDLE);
    VK_CHECK(result);

    result = vkQueueWaitIdle(computeQueue);
    VK_CHECK(result);

    void* outputPtr = nullptr;
    if (outputData && 0) // always copy prefix sum for now
    {
        VkResult mapResult = vkMapMemory(device, realBufferMemory, 0, signalByFreqFloatSizeBytes, 0, &outputPtr);
        if (mapResult == VK_SUCCESS) {
            const size_t elementCount = signalByFreqFloatSizeBytes / sizeof(float);
            outputData->resize(elementCount);
            std::memcpy(outputData->data(), outputPtr, signalByFreqFloatSizeBytes);
            vkUnmapMemory(device, realBufferMemory);
            std::cout << "Copied " << outputData->size() << " comb filter samples to output vector" << std::endl;
        } else {
            std::cerr << "Failed to map comb filter buffer memory for copying to output vector" << std::endl;
        }
    }
    else if (outputData && 0) // always copy prefix sum for now
    {
        VkResult mapResult = vkMapMemory(device, realPrefixSumBufferMemory, 0, prefixSumBufferSizeBytes, 0, &outputPtr);
        if (mapResult == VK_SUCCESS) {
            const size_t elementCount = prefixSumBufferSizeBytes / sizeof(float);
            outputData->resize(elementCount);
            std::memcpy(outputData->data(), outputPtr, prefixSumBufferSizeBytes);
            vkUnmapMemory(device, realPrefixSumBufferMemory);
            std::cout << "Copied " << outputData->size() << " comb filter samples to output vector" << std::endl;
        } else {
            std::cerr << "Failed to map comb filter buffer memory for copying to output vector" << std::endl;
        }
    }
    // Copy entire comb filter output buffer to the output vector
    else if (outputData) {
        VkResult mapResult = vkMapMemory(device, combFilterBufferMemory, 0, combFilterBufferSizeBytes, 0, &outputPtr);
        if (mapResult == VK_SUCCESS) {
            const size_t elementCount = combFilterBufferSizeBytes / sizeof(float);
            outputData->resize(elementCount);
            std::memcpy(outputData->data(), outputPtr, combFilterBufferSizeBytes);
            vkUnmapMemory(device, combFilterBufferMemory);
            std::cout << "Copied " << outputData->size() << " comb filter samples to output vector" << std::endl;
        } else {
            std::cerr << "Failed to map comb filter buffer memory for copying to output vector" << std::endl;
        }
    }


    // Cleanup
    if (signalBuffer != VK_NULL_HANDLE)
        vkDestroyBuffer(device, signalBuffer, nullptr);
    if (signalBufferMemory != VK_NULL_HANDLE)
        vkFreeMemory(device, signalBufferMemory, nullptr);
    if (realBuffer != VK_NULL_HANDLE)
        vkDestroyBuffer(device, realBuffer, nullptr);
    if (realBufferMemory != VK_NULL_HANDLE)
        vkFreeMemory(device, realBufferMemory, nullptr);
    if (imagBuffer != VK_NULL_HANDLE)
        vkDestroyBuffer(device, imagBuffer, nullptr);
    if (imagBufferMemory != VK_NULL_HANDLE)
        vkFreeMemory(device, imagBufferMemory, nullptr);
    if (freqBuffer != VK_NULL_HANDLE)
        vkDestroyBuffer(device, freqBuffer, nullptr);
    if (freqBufferMemory != VK_NULL_HANDLE)
        vkFreeMemory(device, freqBufferMemory, nullptr);
    if (imagPrefixSumBuffer != VK_NULL_HANDLE)
        vkDestroyBuffer(device, imagPrefixSumBuffer, nullptr);
    if (imagPrefixSumBufferMemory != VK_NULL_HANDLE)
        vkFreeMemory(device, imagPrefixSumBufferMemory, nullptr);
    if (realPrefixSumBufferMemory != VK_NULL_HANDLE)
        vkFreeMemory(device, realPrefixSumBufferMemory, nullptr);
    if (commandPool != VK_NULL_HANDLE)
        vkDestroyCommandPool(device, commandPool, nullptr);
    if (descriptorPool != VK_NULL_HANDLE)
        vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    if (descriptorSetLayout != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
    if (magnitudeShaderModule != VK_NULL_HANDLE)
        vkDestroyShaderModule(device, magnitudeShaderModule, nullptr);
    if (pipelineLayout != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    if (magnitudePipeline != VK_NULL_HANDLE)
        vkDestroyPipeline(device, magnitudePipeline, nullptr);
    if (prefixSumRealPipeline != VK_NULL_HANDLE)
        vkDestroyPipeline(device, prefixSumRealPipeline, nullptr);
    if (prefixSumImagPipeline != VK_NULL_HANDLE)
        vkDestroyPipeline(device, prefixSumImagPipeline, nullptr);
    // Cleanup comb filter resources
    if (combFilterPipeline != VK_NULL_HANDLE)
        vkDestroyPipeline(device, combFilterPipeline, nullptr);
    if (combFilterShaderModule != VK_NULL_HANDLE)
        vkDestroyShaderModule(device, combFilterShaderModule, nullptr);
    if (combFilterBuffer != VK_NULL_HANDLE)
        vkDestroyBuffer(device, combFilterBuffer, nullptr);
    if (combFilterBufferMemory != VK_NULL_HANDLE)
        vkFreeMemory(device, combFilterBufferMemory, nullptr);

    // Destroy debug messenger before instance destruction
    if (g_debugMessenger != VK_NULL_HANDLE)
    {
        DestroyDebugUtilsMessengerEXT(instance, g_debugMessenger, nullptr);
        g_debugMessenger = VK_NULL_HANDLE;
    }

    cleanupVulkan(instance, device);

    std::cout << "Processing completed successfully!" << std::endl;
    return 0;
}
