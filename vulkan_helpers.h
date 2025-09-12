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

// Error checking macro for Vulkan calls (only use in VkResult-returning functions)
#define VK_CHECK(result) \
    if ((result) != VK_SUCCESS) { \
        std::cerr << "Vulkan error: " << (result) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        return (result); \
    }

// ---------- Helpers & Declarations ----------

// Load SPIR-V from disk (must be multiple of 4 bytes)
VkResult loadSpirvFile(const char* path, std::vector<uint32_t>& outCode);

// Find a queue family that supports compute
bool findComputeQueueFamily(VkPhysicalDevice phys, uint32_t& outIndex);


// ---- Added: minimal helpers for validation layers & debug utils ----
extern VkDebugUtilsMessengerEXT g_debugMessenger;

VKAPI_ATTR VkBool32 VKAPI_CALL DebugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT /*messageSeverity*/,
    VkDebugUtilsMessageTypeFlagsEXT        /*messageType*/,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* /*pUserData*/);

bool isLayerAvailable(const char* name);

bool isExtensionAvailable(const char* name);

VkResult CreateDebugUtilsMessengerEXT(VkInstance instance,
    const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
    const VkAllocationCallbacks* pAllocator,
    VkDebugUtilsMessengerEXT* pMessenger);

void DestroyDebugUtilsMessengerEXT(VkInstance instance,
    VkDebugUtilsMessengerEXT messenger,
    const VkAllocationCallbacks* pAllocator);
// --------------------------------------------------------------------
