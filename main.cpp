// main.cpp
// Single-file version combining main and audio processing utilities.
// Build example (adjust include/library paths as needed):
// g++ -std=c++17 main.cpp -I. -lvulkan -o vulkan_audio_processor
#include "main.h"


void cleanupVulkan(VkInstance instance, VkDevice device) {
    if (device != VK_NULL_HANDLE) {
        vkDestroyDevice(device, nullptr);
    }
    if (instance != VK_NULL_HANDLE) {
        vkDestroyInstance(instance, nullptr);
    }
}

VkResult readResultFromGPU(VkDevice device, VkDeviceMemory bufferMemory, float& result) {
    void* data;
    VkResult res = vkMapMemory(device, bufferMemory, 0, sizeof(float), 0, &data);
    if (res != VK_SUCCESS) {
        std::cerr << "Failed to map memory for result" << std::endl;
        return res;
    }
    result = *reinterpret_cast<float*>(data);
    vkUnmapMemory(device, bufferMemory);
    return VK_SUCCESS;
}

void cleanupComputeResources(VkDevice device, VkPipeline pipeline, VkPipelineLayout pipelineLayout,
                            VkShaderModule shaderModule, VkDescriptorSetLayout descriptorSetLayout,
                            VkDescriptorPool descriptorPool, VkCommandPool commandPool,
                            VkBuffer inputBuffer, VkDeviceMemory inputBufferMemory,
                            VkBuffer outputBuffer, VkDeviceMemory outputBufferMemory,
                            VkBuffer freqBuffer, VkDeviceMemory freqBufferMemory) {
    if (inputBuffer != VK_NULL_HANDLE) vkDestroyBuffer(device, inputBuffer, nullptr);
    if (inputBufferMemory != VK_NULL_HANDLE) vkFreeMemory(device, inputBufferMemory, nullptr);
    if (outputBuffer != VK_NULL_HANDLE) vkDestroyBuffer(device, outputBuffer, nullptr);
    if (outputBufferMemory != VK_NULL_HANDLE) vkFreeMemory(device, outputBufferMemory, nullptr);
    if (freqBuffer != VK_NULL_HANDLE) vkDestroyBuffer(device, freqBuffer, nullptr);
    if (freqBufferMemory != VK_NULL_HANDLE) vkFreeMemory(device, freqBufferMemory, nullptr);
    if (commandPool != VK_NULL_HANDLE) vkDestroyCommandPool(device, commandPool, nullptr);
    if (descriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    if (descriptorSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
    if (shaderModule != VK_NULL_HANDLE) vkDestroyShaderModule(device, shaderModule, nullptr);
    if (pipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    if (pipeline != VK_NULL_HANDLE) vkDestroyPipeline(device, pipeline, nullptr);
}

// ---------- Main ----------

int main() {
    std::cout << "Vulkan Audio Processor" << std::endl;
    std::cout << "======================" << std::endl;

    // Read audio data from WAV file
    std::vector<float> audioData;
    uint32_t sampleRate = 0;
    uint16_t numChannels = 0;
    if (!readAudioData("audio.wav", audioData, sampleRate, numChannels)) {
        std::cerr << "Failed to read audio data" << std::endl;
        return 1;
    }
    std::cout << "Read " << audioData.size() << " audio samples" << std::endl;
    
    // Create output vector for processed results
    std::vector<float> outputData(audioData.size());
    // Use default frequency of 440 Hz
    std::vector<float> frequencies = { 440.0f };
    loiacono(&audioData, sampleRate, &outputData, &frequencies);
}

int loiacono(std::vector<float>* audioData, float sampleRate, std::vector<float>* outputData, 
    std::vector<float> * frequencies){
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
    std::vector<const char*> enabledLayers;
    std::vector<const char*> enabledExtensions;

    const char* kValidationLayer = "VK_LAYER_KHRONOS_validation";
    if (isLayerAvailable(kValidationLayer)) {
        enabledLayers.push_back(kValidationLayer);
        std::cout << "Enabling layer: " << kValidationLayer << std::endl;
    } else {
        std::cout << "Validation layer not available; continuing without it." << std::endl;
    }

    if (isExtensionAvailable(VK_EXT_DEBUG_UTILS_EXTENSION_NAME)) {
        enabledExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        std::cout << "Enabling extension: " << VK_EXT_DEBUG_UTILS_EXTENSION_NAME << std::endl;
    } else {
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
    if (!enabledExtensions.empty()) {
        for (auto* ext : enabledExtensions) {
            if (std::strcmp(ext, VK_EXT_DEBUG_UTILS_EXTENSION_NAME) == 0) {
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
    if (!enabledExtensions.empty()) {
        for (auto* ext : enabledExtensions) {
            if (std::strcmp(ext, VK_EXT_DEBUG_UTILS_EXTENSION_NAME) == 0) {
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
                if (CreateDebugUtilsMessengerEXT(instance, &ci, nullptr, &g_debugMessenger) != VK_SUCCESS) {
                    std::cerr << "Failed to create debug messenger (continuing)" << std::endl;
                }
                break;
            }
        }
    }

    // Enumerate physical devices
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    if (deviceCount == 0) {
        std::cerr << "No Vulkan devices found" << std::endl;
        // Destroy debug messenger (if created) before destroying instance
        if (g_debugMessenger != VK_NULL_HANDLE) {
            DestroyDebugUtilsMessengerEXT(instance, g_debugMessenger, nullptr);
            g_debugMessenger = VK_NULL_HANDLE;
        }
        return VK_ERROR_DEVICE_LOST;
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

    // Choose first device with a compute queue
    physicalDevice = VK_NULL_HANDLE;
    for (auto pd : devices) {
        if (findComputeQueueFamily(pd, computeQueueFamilyIndex)) {
            physicalDevice = pd;
            break;
        }
    }
    if (physicalDevice == VK_NULL_HANDLE) {
        std::cerr << "No compute-capable Vulkan device found" << std::endl;
        if (g_debugMessenger != VK_NULL_HANDLE) {
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
    if (computeQueue == VK_NULL_HANDLE) {
        std::cerr << "Failed to get compute device queue" << std::endl;
        if (g_debugMessenger != VK_NULL_HANDLE) {
            DestroyDebugUtilsMessengerEXT(instance, g_debugMessenger, nullptr);
            g_debugMessenger = VK_NULL_HANDLE;
        }
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    // Create compute pipeline
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkShaderModule shaderModule = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;

    std::cout << "Device: " << device << std::endl;

    // Descriptor set layout (set = 0, bindings 0, 1, and 2)
    VkDescriptorSetLayoutBinding bindings[3] = {};
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

    VkDescriptorSetLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 3;
    layoutInfo.pBindings = bindings;

    VkResult result = vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout);
    std::cout << "vkCreateDescriptorSetLayout result: " << result << std::endl;
    VK_CHECK(result);

    // Pipeline layout + push constants
    VkPushConstantRange pcRange = {};
    pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcRange.offset     = 0;
    pcRange.size       = sizeof(PushConstants);

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
    result = loadSpirvFile("loiacono.comp.spv", spirv);
    if (result != VK_SUCCESS) {
        std::cerr << "Failed to load SPIR-V for compute shader" << std::endl;
        if (g_debugMessenger != VK_NULL_HANDLE) {
            DestroyDebugUtilsMessengerEXT(instance, g_debugMessenger, nullptr);
            g_debugMessenger = VK_NULL_HANDLE;
        }
        return result;
    }

    VkShaderModuleCreateInfo smci = {};
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = spirv.size() * sizeof(uint32_t);
    smci.pCode    = spirv.data();

    result = vkCreateShaderModule(device, &smci, nullptr, &shaderModule);
    std::cout << "vkCreateShaderModule result: " << result << std::endl;
    VK_CHECK(result);

    VkPipelineShaderStageCreateInfo shaderStageInfo = {};
    shaderStageInfo.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStageInfo.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    shaderStageInfo.module = shaderModule;
    shaderStageInfo.pName  = "main";

    VkComputePipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage  = shaderStageInfo;
    pipelineInfo.layout = pipelineLayout;

    result = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline);
    std::cout << "vkCreateComputePipelines result: " << result << std::endl;
    if (result != VK_SUCCESS) {
        std::cerr << "Failed to create compute pipeline" << std::endl;
        if (g_debugMessenger != VK_NULL_HANDLE) {
            DestroyDebugUtilsMessengerEXT(instance, g_debugMessenger, nullptr);
            g_debugMessenger = VK_NULL_HANDLE;
        }
        return result;
    }

    // Descriptor pool
    VkDescriptorPoolSize poolSizes[1] = {};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[0].descriptorCount = 3; // three storage buffers total

    VkDescriptorPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = poolSizes;
    poolInfo.maxSets = 1;

    result = vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool);
    std::cout << "vkCreateDescriptorPool result: " << result << std::endl;
    VK_CHECK(result);

    // Allocate descriptor set
    VkDescriptorSetAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &descriptorSetLayout;

    result = vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet);
    std::cout << "vkAllocateDescriptorSets result: " << result << std::endl;
    VK_CHECK(result);

    // Command pool & buffer — use the SAME compute queue family index
    VkCommandPoolCreateInfo cmdPoolInfo = {};
    cmdPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cmdPoolInfo.queueFamilyIndex = computeQueueFamilyIndex;

    result = vkCreateCommandPool(device, &cmdPoolInfo, nullptr, &commandPool);
    std::cout << "vkCreateCommandPool result: " << result << std::endl;
    VK_CHECK(result);

    VkCommandBufferAllocateInfo cmdBufferInfo = {};
    cmdBufferInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdBufferInfo.commandPool = commandPool;
    cmdBufferInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdBufferInfo.commandBufferCount = 1;

    result = vkAllocateCommandBuffers(device, &cmdBufferInfo, &commandBuffer);
    std::cout << "vkAllocateCommandBuffers result: " << result << std::endl;

    if (result != VK_SUCCESS) {
        if (g_debugMessenger != VK_NULL_HANDLE) {
            DestroyDebugUtilsMessengerEXT(instance, g_debugMessenger, nullptr);
            g_debugMessenger = VK_NULL_HANDLE;
        }
        cleanupVulkan(instance, device);
        return 1;
    }

    // Create buffers for audio data
    VkBuffer inputBuffer = VK_NULL_HANDLE;
    VkBuffer outputBuffer = VK_NULL_HANDLE;
    VkDeviceMemory inputBufferMemory = VK_NULL_HANDLE;
    VkDeviceMemory outputBufferMemory = VK_NULL_HANDLE;

    // Create input & output buffers
    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = audioData->size() * sizeof(float);
    bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VK_CHECK(vkCreateBuffer(device, &bufferInfo, nullptr, &inputBuffer));
    VK_CHECK(vkCreateBuffer(device, &bufferInfo, nullptr, &outputBuffer));

    // Allocate memory for input buffer
    VkMemoryRequirements memRequirementsIn{};
    vkGetBufferMemoryRequirements(device, inputBuffer, &memRequirementsIn);

    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

    auto findHostVisibleCoherent = [&](uint32_t typeBits)->uint32_t {
        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeBits & (1 << i)) &&
                (memProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
                (memProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
                return i;
            }
        }
        return UINT32_MAX;
    };

    uint32_t memoryTypeIndexIn = findHostVisibleCoherent(memRequirementsIn.memoryTypeBits);
    if (memoryTypeIndexIn == UINT32_MAX) {
        std::cerr << "Failed to find suitable memory type for input" << std::endl;
        if (g_debugMessenger != VK_NULL_HANDLE) {
            DestroyDebugUtilsMessengerEXT(instance, g_debugMessenger, nullptr);
            g_debugMessenger = VK_NULL_HANDLE;
        }
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }

    VkMemoryAllocateInfo allocInfoIn = {};
    allocInfoIn.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfoIn.allocationSize = memRequirementsIn.size;
    allocInfoIn.memoryTypeIndex = memoryTypeIndexIn;

    VK_CHECK(vkAllocateMemory(device, &allocInfoIn, nullptr, &inputBufferMemory));
    VK_CHECK(vkBindBufferMemory(device, inputBuffer, inputBufferMemory, 0));

    // Allocate memory for output buffer
    VkMemoryRequirements memRequirementsOut{};
    vkGetBufferMemoryRequirements(device, outputBuffer, &memRequirementsOut);

    uint32_t memoryTypeIndexOut = findHostVisibleCoherent(memRequirementsOut.memoryTypeBits);
    if (memoryTypeIndexOut == UINT32_MAX) {
        std::cerr << "Failed to find suitable memory type for output" << std::endl;
        if (g_debugMessenger != VK_NULL_HANDLE) {
            DestroyDebugUtilsMessengerEXT(instance, g_debugMessenger, nullptr);
            g_debugMessenger = VK_NULL_HANDLE;
        }
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }

    VkMemoryAllocateInfo allocInfoOut = {};
    allocInfoOut.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfoOut.allocationSize = memRequirementsOut.size;
    allocInfoOut.memoryTypeIndex = memoryTypeIndexOut;

    VK_CHECK(vkAllocateMemory(device, &allocInfoOut, nullptr, &outputBufferMemory));
    VK_CHECK(vkBindBufferMemory(device, outputBuffer, outputBufferMemory, 0));

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
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((memRequirements.memoryTypeBits & (1 << i)) &&
            (memProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
            (memProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            memoryTypeIndex = i;
            break;
        }
    }

    if (memoryTypeIndex == UINT32_MAX) {
        std::cerr << "Failed to find suitable memory type for frequency buffer" << std::endl;
        if (g_debugMessenger != VK_NULL_HANDLE) {
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

    // Copy audio data to GPU (input)
    void* data;
    VK_CHECK(vkMapMemory(device, inputBufferMemory, 0, audioData->size() * sizeof(float), 0, &data));
    std::memcpy(data, audioData->data(), audioData->size() * sizeof(float));
    vkUnmapMemory(device, inputBufferMemory);

    // Copy frequency data to GPU
    VK_CHECK(vkMapMemory(device, freqBufferMemory, 0, frequencies->size() * sizeof(float), 0, &data));
    std::memcpy(data, frequencies->data(), frequencies->size() * sizeof(float));
    vkUnmapMemory(device, freqBufferMemory);

    // IMPORTANT: Initialize output buffer to 0.0f if shader accumulates into it
    {
        void* outPtr = nullptr;
        VkResult mapRes = vkMapMemory(device, outputBufferMemory, 0, audioData->size() * sizeof(float), 0, &outPtr);
        if (mapRes != VK_SUCCESS) {
            std::cerr << "Failed to map output buffer memory for initialization" << std::endl;
            cleanupComputeResources(device, pipeline, pipelineLayout, shaderModule,
                                    descriptorSetLayout, descriptorPool, commandPool,
                                    inputBuffer, inputBufferMemory,
                                    outputBuffer, outputBufferMemory,
                                    freqBuffer, freqBufferMemory);
            if (g_debugMessenger != VK_NULL_HANDLE) {
                DestroyDebugUtilsMessengerEXT(instance, g_debugMessenger, nullptr);
                g_debugMessenger = VK_NULL_HANDLE;
            }
            cleanupVulkan(instance, device);
            return 1;
        }
        std::memset(outPtr, 0, audioData->size() * sizeof(float));
        vkUnmapMemory(device, outputBufferMemory);
    }

    // Update descriptor set
    VkDescriptorBufferInfo inputBufferInfo = {};
    inputBufferInfo.buffer = inputBuffer;
    inputBufferInfo.offset = 0;
    inputBufferInfo.range = VK_WHOLE_SIZE;

    VkDescriptorBufferInfo outputBufferInfo = {};
    outputBufferInfo.buffer = outputBuffer;
    outputBufferInfo.offset = 0;
    outputBufferInfo.range = VK_WHOLE_SIZE;

    VkDescriptorBufferInfo freqBufferInfo = {};
    freqBufferInfo.buffer = freqBuffer;
    freqBufferInfo.offset = 0;
    freqBufferInfo.range = VK_WHOLE_SIZE;

    VkWriteDescriptorSet descriptorWrites[3] = {};
    descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[0].dstSet = descriptorSet;
    descriptorWrites[0].dstBinding = 0;
    descriptorWrites[0].dstArrayElement = 0;
    descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[0].descriptorCount = 1;
    descriptorWrites[0].pBufferInfo = &inputBufferInfo;

    descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[1].dstSet = descriptorSet;
    descriptorWrites[1].dstBinding = 1;
    descriptorWrites[1].dstArrayElement = 0;
    descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[1].descriptorCount = 1;
    descriptorWrites[1].pBufferInfo = &outputBufferInfo;

    descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[2].dstSet = descriptorSet;
    descriptorWrites[2].dstBinding = 2;
    descriptorWrites[2].dstArrayElement = 0;
    descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[2].descriptorCount = 1;
    descriptorWrites[2].pBufferInfo = &freqBufferInfo;

    vkUpdateDescriptorSets(device, 3, descriptorWrites, 0, nullptr);

    // Prepare push constants (Per-dispatch scalars go here)
    PushConstants pc{};
    pc.startPos         = 0u;
    pc.endPos           = static_cast<uint32_t>(audioData->size());
    pc.sampleFrequency  = static_cast<float>(sampleRate);
    pc.multiple         = 4096.0f; // tweak as desired; ensure ringBufferSize <= RING_BUFFER_MAX_SIZE

    std::cout << "Device: " << device << std::endl;
    std::cout << "Command Buffer: " << commandBuffer << std::endl;
    std::cout << "Pipeline: " << pipeline << std::endl;

    // Begin command buffer
    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    result = vkBeginCommandBuffer(commandBuffer, &beginInfo);
    std::cout << "vkBeginCommandBuffer result: " << result << std::endl;
    VK_CHECK(result);

    // Bind pipeline and descriptor set
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);

    // Push constants (Per-dispatch scalars go here)
    vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &pc);

    // Dispatch compute
    // EACH WORKGROUP PROCESSES A SINGLE FREQUENCY!
    // So we launch one workgroup per frequency in X dimension.
    vkCmdDispatch(commandBuffer, frequencyCount, 1, 1);

    VkMemoryBarrier memBarrier{};
    memBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;

    vkCmdPipelineBarrier(
        commandBuffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_HOST_BIT,
        0,
        1, &memBarrier,
        0, nullptr,
        0, nullptr
    );

    result = vkEndCommandBuffer(commandBuffer);
    std::cout << "vkEndCommandBuffer result: " << result << std::endl;
    VK_CHECK(result);

    // Submit to the provided compute queue
    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    result = vkQueueSubmit(computeQueue, 1, &submitInfo, VK_NULL_HANDLE);
    std::cout << "vkQueueSubmit result: " << result << std::endl;
    VK_CHECK(result);

    result = vkQueueWaitIdle(computeQueue);
    std::cout << "vkQueueWaitIdle result: " << result << std::endl;
    VK_CHECK(result);

    // Read back a sample result (for example, the first output sample)
    float firstSample = 0.0f;
    vkMapMemory(device, outputBufferMemory, 0, sizeof(float), 0, &data);
    firstSample = *reinterpret_cast<float*>(data);
    vkUnmapMemory(device, outputBufferMemory);

    std::cout << "First output sample (for inspection): " << firstSample << std::endl;

    // Copy entire output buffer to output vector
    if (outputData != nullptr) {
        void* outputPtr;
        VkResult mapResult = vkMapMemory(device, outputBufferMemory, 0, audioData->size() * sizeof(float), 0, &outputPtr);
        if (mapResult == VK_SUCCESS) {
            outputData->resize(audioData->size());
            std::memcpy(outputData->data(), outputPtr, audioData->size() * sizeof(float));
            vkUnmapMemory(device, outputBufferMemory);
            std::cout << "Copied " << outputData->size() << " processed samples to output vector" << std::endl;
        } else {
            std::cerr << "Failed to map output buffer memory for copying to output vector" << std::endl;
        }
    }

    // Cleanup
    cleanupComputeResources(device, pipeline, pipelineLayout, shaderModule,
                            descriptorSetLayout, descriptorPool, commandPool,
                            inputBuffer, inputBufferMemory,
                            outputBuffer, outputBufferMemory,
                            freqBuffer, freqBufferMemory);

    // Destroy debug messenger before instance destruction
    if (g_debugMessenger != VK_NULL_HANDLE) {
        DestroyDebugUtilsMessengerEXT(instance, g_debugMessenger, nullptr);
        g_debugMessenger = VK_NULL_HANDLE;
    }

    cleanupVulkan(instance, device);

    std::cout << "Processing completed successfully!" << std::endl;
    return 0;
}
