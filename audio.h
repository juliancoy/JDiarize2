#pragma once

#include <string>
#include <vector>

// Simple WAV file reader (for 16-bit PCM WAV files)
bool readAudioData(const std::string& filename, std::vector<float>& audioData, uint32_t& outSampleRate, uint16_t& outNumChannels);
