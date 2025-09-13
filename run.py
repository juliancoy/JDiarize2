#!/usr/bin/env python3

import numpy as np
import cv2
from cffi import FFI
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import sys
import os

def create_sine_wave(frequency=440.0, duration=1.0, sample_rate=44100, amplitude=0.5):
    """Create a sine wave with the specified frequency and duration"""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    sine_wave = amplitude * np.sin(2 * np.pi * frequency * t)
    return sine_wave.astype(np.float32), sample_rate

def generate_logspace_frequencies(start_freq=20.0, end_freq=20000.0, num_freqs=128):
    """Generate a logarithmic space of frequencies"""
    return np.logspace(np.log10(start_freq), np.log10(end_freq), num_freqs, dtype=np.float32)

def create_heatmap(data, frequencies, sample_rate, output_path="heatmap.png"):
    """Create and save a heatmap visualization"""
    # Normalize the data for better visualization
    data_normalized = (data - data.min()) / (data.max() - data.min() + 1e-8)
    
    # Create time axis (in seconds)
    time_axis = np.linspace(0, len(data) / sample_rate, len(data))
    
    # Create frequency axis (log scale)
    freq_axis = frequencies
    
    # Create meshgrid for plotting
    T, F = np.meshgrid(time_axis, freq_axis)
    
    # Create the heatmap
    plt.figure(figsize=(12, 8))
    plt.pcolormesh(T, F, data_normalized.T, shading='auto', cmap='viridis', norm=LogNorm())
    plt.colorbar(label='Normalized Amplitude')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Audio Processing Heatmap (Loiacono Algorithm)')
    plt.yscale('log')
    plt.ylim(frequencies[0], frequencies[-1])
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Heatmap saved to {output_path}")

def main():
    # Initialize FFI
    ffi = FFI()
    
    # Define the C interface
    ffi.cdef("""
        typedef struct {
            void* data;
            size_t size;
        } FloatVector;
        
        int loiacono(FloatVector* audioData, float sampleRate, FloatVector* outputData, FloatVector* frequencies);
    """)
    
    # Load the shared library
    try:
        lib = ffi.dlopen("./libaudio_processor.so")
        print("Successfully loaded libaudio_processor.so")
    except Exception as e:
        print(f"Failed to load shared library: {e}")
        print("Make sure to run build.sh first to compile the library")
        return 1
    
    # Create a 440Hz sine wave
    print("Generating 440Hz sine wave...")
    audio_data, sample_rate = create_sine_wave(frequency=440.0, duration=2.0)
    print(f"Generated {len(audio_data)} samples at {sample_rate}Hz sample rate")
    
    # Generate logspace of 128 frequencies
    print("Generating logspace of 128 frequencies...")
    frequencies = generate_logspace_frequencies(num_freqs=128)
    print(f"Generated frequencies from {frequencies[0]:.1f}Hz to {frequencies[-1]:.1f}Hz")
    
    # Prepare output data buffer - we need space for all frequencies x time samples
    # Each frequency will produce output for the entire audio duration
    total_output_size = len(audio_data) * len(frequencies)
    output_data = np.zeros(total_output_size, dtype=np.float32)
    
    # Create FFI-compatible structures
    audio_vec = ffi.new("FloatVector*")
    audio_vec.data = ffi.cast("float*", ffi.from_buffer(audio_data))
    audio_vec.size = len(audio_data)
    
    output_vec = ffi.new("FloatVector*")
    output_vec.data = ffi.cast("float*", ffi.from_buffer(output_data))
    output_vec.size = len(output_data)
    
    freq_vec = ffi.new("FloatVector*")
    freq_vec.data = ffi.cast("float*", ffi.from_buffer(frequencies))
    freq_vec.size = len(frequencies)
    
    # Call the loiacono function
    print("Calling loiacono function...")
    result = lib.loiacono(audio_vec, sample_rate, output_vec, freq_vec)
    
    if result != 0:
        print(f"loiacono function returned error code: {result}")
        return 1
    
    print("Processing completed successfully!")
    print(f"Output data shape: {output_data.shape}")
    
    # Reshape output for heatmap (time x frequencies)
    # The loiacono function processes each frequency and stores results
    heatmap_data = output_data.reshape(-1, len(frequencies))
    
    # Create and save heatmap
    print("Creating heatmap visualization...")
    create_heatmap(heatmap_data, frequencies, sample_rate, "loiacono_heatmap.png")
    
    # Also save with OpenCV for comparison
    heatmap_normalized = cv2.normalize(heatmap_data, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_uint8 = heatmap_normalized.astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_VIRIDIS)
    
    # Resize for better visualization
    height, width = heatmap_colored.shape[:2]
    new_height = 800
    new_width = int(width * (new_height / height))
    heatmap_resized = cv2.resize(heatmap_colored, (new_width, new_height))
    
    cv2.imwrite("loiacono_heatmap_cv2.png", heatmap_resized)
    print("OpenCV heatmap saved to loiacono_heatmap_cv2.png")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
