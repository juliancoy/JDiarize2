#include "audio.h"
#include <iostream>
#include <fstream>
#include <cstring>


// Simple WAV file reader (for 16-bit PCM WAV files)
bool readAudioData(const std::string& filename, std::vector<float>& audioData, uint32_t& outSampleRate, uint16_t& outNumChannels) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open audio file: " << filename << std::endl;
        return false;
    }

    // Read WAV header
    char riff[4];
    file.read(riff, 4);
    if (std::string(riff, 4) != "RIFF") {
        std::cerr << "Not a valid WAV file" << std::endl;
        return false;
    }

    // Skip file size
    file.seekg(4, std::ios::cur);

    char wave[4];
    file.read(wave, 4);
    if (std::string(wave, 4) != "WAVE") {
        std::cerr << "Not a valid WAV file" << std::endl;
        return false;
    }

    // Find fmt chunk
    char chunkId[4];
    uint32_t chunkSize;
    while (file) {
        file.read(chunkId, 4);
        file.read(reinterpret_cast<char*>(&chunkSize), 4);

        if (std::string(chunkId, 4) == "fmt ") {
            break;
        }
        file.seekg(chunkSize, std::ios::cur);
    }

    // Read fmt chunk
    uint16_t audioFormat, numChannels;
    uint32_t sampleRate, byteRate;
    uint16_t blockAlign, bitsPerSample;

    file.read(reinterpret_cast<char*>(&audioFormat), 2);
    file.read(reinterpret_cast<char*>(&numChannels), 2);
    file.read(reinterpret_cast<char*>(&sampleRate), 4);
    file.read(reinterpret_cast<char*>(&byteRate), 4);
    file.read(reinterpret_cast<char*>(&blockAlign), 2);
    file.read(reinterpret_cast<char*>(&bitsPerSample), 2);

    if (audioFormat != 1) { // PCM format
        std::cerr << "Only PCM WAV files are supported" << std::endl;
        return false;
    }

    if (bitsPerSample != 16) {
        std::cerr << "Only 16-bit WAV files are supported" << std::endl;
        return false;
    }

    // Skip any extra bytes in fmt chunk
    file.seekg(static_cast<std::streamoff>(chunkSize) - 16, std::ios::cur);

    // Find data chunk
    while (file) {
        file.read(chunkId, 4);
        file.read(reinterpret_cast<char*>(&chunkSize), 4);

        if (std::string(chunkId, 4) == "data") {
            break;
        }
        file.seekg(chunkSize, std::ios::cur);
    }

    // Read audio data
    uint32_t numSamples = chunkSize / (numChannels * (bitsPerSample / 8));
    std::vector<int16_t> rawData(numSamples * numChannels);
    file.read(reinterpret_cast<char*>(rawData.data()), chunkSize);

    // Convert to float and handle stereo->mono conversion
    audioData.resize(numSamples);
    for (uint32_t i = 0; i < numSamples; ++i) {
        if (numChannels == 2) {
            float left  = rawData[i * 2] / 32768.0f;
            float right = rawData[i * 2 + 1] / 32768.0f;
            audioData[i] = (left + right) / 2.0f;
        } else {
            audioData[i] = rawData[i] / 32768.0f;
        }
    }

    std::cout << "Read " << audioData.size() << " audio samples from " << filename << std::endl;
    std::cout << "Sample rate: " << sampleRate << " Hz, Channels: " << numChannels << std::endl;

    outSampleRate  = sampleRate;
    outNumChannels = numChannels;
    return true;
}
