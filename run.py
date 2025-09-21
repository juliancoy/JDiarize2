#!/usr/bin/env python3

import numpy as np
import cv2
from cffi import FFI
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import sys
import os


def create_heatmap(data, frequencies, sample_rate, output_path="heatmap.png"):
    """Create and save a heatmap visualization"""
    # Handle NaN and infinite values
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Check if data has valid range for normalization
    if np.all(data == 0) or (data.max() - data.min()) == 0:
        # All zeros or constant data, use zeros for normalized data
        data_normalized = np.zeros_like(data)
    else:
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

def opencv_heatmap(heatmap_data, output_path="heatmap_opencv.png"):
    heatmap_normalized = cv2.normalize(heatmap_data, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_uint8 = heatmap_normalized.astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_VIRIDIS)
    
    # Resize for better visualization (only if dimensions are valid)
    height, width = heatmap_colored.shape[:2]
    print(f"Original heatmap size: {width}x{height}")
    if height > 0 and width > 0:
        new_height = height
        new_width = 1000
        if new_width > 0 and new_height > 0:
            heatmap_resized = cv2.resize(heatmap_colored, (new_width, new_height))
            cv2.imwrite(output_path, heatmap_resized)
            print("OpenCV heatmap saved to loiacono_heatmap_cv2.png")
        else:
            print("Skipping OpenCV heatmap: invalid resize dimensions")
    else:
        print("Skipping OpenCV heatmap: invalid image dimensions")

def main():
    # Initialize FFI
    ffi = FFI()
    
    # Define the C interface
    ffi.cdef("""
        typedef struct {
            void* data;
            size_t size;
        } FloatVector;
        
        int loiacono(FloatVector* audioData, FloatVector* outputData, FloatVector* frequencies, float multiple);
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

    """Create a sine wave with the specified frequency and duration"""
    frequency=440.0
    duration=1.0
    sample_rate=44100
    amplitude=0.5
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio_data = amplitude * np.sin(2 * np.pi * frequency * t)
    audio_data = audio_data.astype(np.float32, copy=False)

    print(f"Generated {len(audio_data)} samples at {sample_rate}Hz sample rate")
    
    # Generate logspace of 128 frequencies
    start_freq=20.0
    end_freq=20000.0
    num_freqs=32
    print(f"Generating logspace of {num_freqs} frequencies...")
    """Generate a logarithmic space of frequencies"""
    frequencies = np.logspace(np.log10(start_freq), np.log10(end_freq), num_freqs, dtype=np.float32)/sample_rate
    print(f"Generated frequencies from {frequencies[0]:.1f}Hz to {frequencies[-1]:.1f}Hz")
    
    frequencies = np.array([220, 440.0, 660, 880], dtype=np.float32) 
    frequencies = np.array([440.0], dtype=np.float32) 
    # Prepare output data buffer - we need space for all frequencies x time samples
    # Each frequency will produce output for the entire audio duration
    print(f"audio data {audio_data}")
    print(f"length {len(audio_data)}")
    print(f"frequencies {frequencies}")
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
    multiple = 1
    
    # Call the loiacono function
    print("Calling loiacono function...")
    result = lib.loiacono(audio_vec, output_vec, freq_vec, multiple)
    
    if result != 0:
        print(f"loiacono function returned error code: {result}")
        return 1
    
    print("Processing completed successfully!")
    print(f"Output data shape: {output_data.shape}")
    
    # Reshape output for heatmap (time x frequencies)
    # The loiacono function processes each frequency and stores results
    heatmap_data = output_data.reshape(len(frequencies), len(audio_data))
    print(heatmap_data)
    heatmap_data[:,-10:] = 0.0  # Zero out the last column to avoid artifacts
    print(f"sum: {np.sum(heatmap_data)}")
    
    # Plot each frequency as a different color
    plt.figure(figsize=(12, 8))
    
    # Create a colormap with enough colors for all frequencies
    colors = plt.cm.viridis(np.linspace(0, 1, len(frequencies)))
    
    # Plot each frequency's time series
    for i, freq in enumerate(frequencies):
        time_series = heatmap_data[i, :]
        plt.plot(time_series, color=colors[i], label=f'{freq:.1f} Hz', alpha=0.7, linewidth=1)
    
    plt.title("Comb Filter Output for Each Frequency")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
        
    # Create and save heatmap (optional)
    print("Creating heatmap visualization...")
    create_heatmap(heatmap_data, frequencies, sample_rate, "loiacono_heatmap.png")
    
    # Also save with OpenCV for comparison (only if data is valid)
    try:
        opencv_heatmap(heatmap_data, "loiacono_heatmap_cv2.png")
    except Exception as e:
        print(f"Error creating OpenCV heatmap: {e}")
        print("Continuing without OpenCV heatmap")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
