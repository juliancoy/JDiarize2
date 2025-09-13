#pragma once

#include <iostream>
#include <vector>
#include <fstream>
#include <cstdint>
#include <cstring>
#include <string>
#include <limits>
#include <cmath>

// Keep your local header path as provided by the user
#include "vulkan/vulkan.h"
#include "vulkan_helpers.h"
#include "audio.h"

// Vulkan lifecycle
VkResult initializeVulkan(VkInstance& instance, VkDevice& device, VkPhysicalDevice& physicalDevice,
                          uint32_t& computeQueueFamilyIndex, VkQueue& computeQueue);
void cleanupVulkan(VkInstance instance, VkDevice device);

// Compute pipeline & buffers
VkResult createComputePipeline(VkDevice device, VkPhysicalDevice /*physicalDevice*/,
                              uint32_t computeQueueFamilyIndex,
                              VkPipeline& pipeline, VkPipelineLayout& pipelineLayout,
                              VkShaderModule& shaderModule, VkDescriptorSetLayout& descriptorSetLayout,
                              VkDescriptorPool& descriptorPool, VkDescriptorSet& descriptorSet,
                              VkCommandPool& commandPool, VkCommandBuffer& commandBuffer);

VkResult createAudioBuffers(VkDevice device, VkPhysicalDevice physicalDevice, size_t dataSize,
                           VkBuffer& inputBuffer, VkDeviceMemory& inputBufferMemory,
                           VkBuffer& outputBuffer, VkDeviceMemory& outputBufferMemory);

VkResult createFrequencyBuffer(VkDevice device, VkPhysicalDevice physicalDevice, size_t freqCount,
                              VkBuffer& freqBuffer, VkDeviceMemory& freqBufferMemory);

VkResult copyAudioDataToGPU(VkDevice device, VkDeviceMemory bufferMemory, const std::vector<float>& audioData);
VkResult copyFrequenciesToGPU(VkDevice device, VkDeviceMemory bufferMemory, const std::vector<float>& freqs);
VkResult updateDescriptorSet(VkDevice device, VkDescriptorSet descriptorSet, VkBuffer inputBuffer, VkBuffer outputBuffer, VkBuffer freqBuffer);

// Execute & readback
struct PushConstants {
    uint32_t startPos;
    uint32_t endPos;
    float    sampleFrequency;
    float    multiple;          // if truly per-dispatch; otherwise move to spec const
};

VkResult executeComputeShader(VkDevice device, VkQueue computeQueue, VkCommandBuffer commandBuffer, VkPipeline pipeline,
                             VkPipelineLayout pipelineLayout, VkDescriptorSet descriptorSet,
                             const PushConstants& pc, uint32_t frequencyCount);

VkResult readResultFromGPU(VkDevice device, VkDeviceMemory bufferMemory, float& result);

// Loiacono function
int loiacono(std::vector<float>* audioData, float sampleRate, std::vector<float>* outputData, std::vector<float>* frequencies, float multiple);

// Cleanup helpers
void cleanupComputeResources(VkDevice device, VkPipeline pipeline, VkPipelineLayout pipelineLayout,
                            VkShaderModule shaderModule, VkDescriptorSetLayout descriptorSetLayout,
                            VkDescriptorPool descriptorPool, VkCommandPool commandPool,
                            VkBuffer inputBuffer, VkDeviceMemory inputBufferMemory,
                            VkBuffer outputBuffer, VkDeviceMemory outputBufferMemory,
                            VkBuffer freqBuffer, VkDeviceMemory freqBufferMemory);
