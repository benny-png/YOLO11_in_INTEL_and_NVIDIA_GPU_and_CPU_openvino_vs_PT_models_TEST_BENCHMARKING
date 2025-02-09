# YOLO Multi-Device Benchmark Tool

![Benchmarking GIF](https://github.com/benny-png/YOLO11_in_INTEL_and_NVIDIA_GPU_and_CPU_openvino_vs_PT_models_TEST_BENCHMARKING/blob/main/ezgif-3b89b35cbf3cc0.gif)

A real-time benchmarking tool that compares YOLO model performance across different devices and model formats simultaneously. This tool provides a visual comparison of object detection performance on:
- NVIDIA GPU with PyTorch
- Intel GPU with OpenVINO
- Intel CPU with OpenVINO
- CPU with PyTorch

## Features

- Real-time concurrent inference on multiple devices
- Live performance metrics (FPS, inference time, total processing time)
- Split-screen visualization of all models running simultaneously
- Video recording of benchmark sessions
- Support for both PyTorch and OpenVINO formats
- Threaded processing for optimal performance
- Video speed adjustment utilities

## Requirements

- NVIDIA GPU with CUDA support
- Intel GPU (for OpenVINO acceleration)
- Python 3.8 or higher
- OpenVINO toolkit
- CUDA toolkit

## Installation

1. Clone the repository:
```bash
git clone https://github.com/benny-png/YOLO11_in_INTEL_and_NVIDIA_GPU_and_CPU_openvino_vs_PT_models_TEST_BENCHMARKING
cd YOLO11_in_INTEL_and_NVIDIA_GPU_and_CPU_openvino_vs_PT_models_TEST_BENCHMARKING
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install CUDA toolkit (if not already installed)
5. Install OpenVINO toolkit (if not already installed)

## Usage

### Running the Benchmark

1. Start the benchmark:
```bash
python benchmark_intel_vs_nvidia.py
```

2. The application will show a split-screen view with:
   - Top-left: NVIDIA GPU inference
   - Top-right: Intel GPU inference
   - Bottom-left: Intel CPU inference
   - Bottom-right: Regular CPU inference

3. Press 'q' to stop the benchmark and save the video.

### Adjusting Video Speed

To modify the speed of a recorded benchmark video:

```bash
python increase_speed.py
```

## Files Description

- `benchmark.py`: Main benchmarking script with concurrent model inference
- `speed_up_video.py`: Utility to adjust the speed of recorded benchmark videos
- `requirements.txt`: Python package dependencies
- `README.md`: Project documentation

## Performance Metrics

The tool displays real-time metrics for each device:
- Frames Per Second (FPS)
- Inference Time (ms)
- Total Processing Time (ms)

## Troubleshooting

### Common Issues

1. CUDA not found:
   - Ensure CUDA toolkit is properly installed
   - Check NVIDIA drivers are up to date

2. OpenVINO errors:
   - Verify OpenVINO toolkit installation
   - Check Intel GPU drivers are installed

3. Video lag:
   - Reduce input resolution
   - Close other GPU-intensive applications
   - Ensure adequate system cooling

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.