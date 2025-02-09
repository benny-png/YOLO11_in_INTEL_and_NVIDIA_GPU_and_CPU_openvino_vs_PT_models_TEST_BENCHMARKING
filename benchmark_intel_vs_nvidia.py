from ultralytics import YOLO
import cv2
import numpy as np
from statistics import mean
import time
from datetime import datetime
import threading
from queue import Queue, Empty
import torch

class ModelThread(threading.Thread):
    def __init__(self, frame_queue, result_queue, config, model):
        super().__init__()
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.config = config
        self.model = model
        self.running = True

    def run(self):
        while self.running:
            try:
                # Non-blocking get with timeout
                try:
                    frame = self.frame_queue.get_nowait()
                except Empty:
                    time.sleep(0.001)  # Small sleep to prevent CPU spinning
                    continue
                    
                if frame is None:
                    break
                
                # Run inference
                start_time = time.time()
                results = self.model(frame, device=self.config["device"], verbose=True)
                inference_time = (time.time() - start_time) * 1000
                
                # Get processed frame
                result_frame = results[0].plot()
                
                # Send results (non-blocking)
                try:
                    self.result_queue.put_nowait({
                        'frame': result_frame,
                        'inference_time': inference_time,
                        'speed': results[0].speed,
                        'config_name': self.config["name"]
                    })
                except Full:
                    pass  # Skip frame if result queue is full
                    
            except Exception as e:
                print(f"Error in {self.config['name']}: {str(e)}")
                continue

class BenchmarkVisualizer:
    def __init__(self, width=1920, height=1080):
        self.width = width
        self.height = height
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.configs = [
            {"name": "NVIDIA GPU (PT)", "device": 0, "model_type": "pt"},
            {"name": "Intel GPU (OpenVINO)", "device": 1, "model_type": "openvino"},
            {"name": "Intel CPU (OpenVINO)", "device": "cpu", "model_type": "openvino"},
            {"name": "CPU (PT)", "device": "cpu", "model_type": "pt"}
        ]

    def create_output_layout(self, frame):
        h, w = frame.shape[:2]
        grid_h = h * 2
        grid_w = w * 2
        layout = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
        return layout

    def draw_stats(self, img, stats, pos, name):
        x, y = pos
        cv2.rectangle(img, (x, y), (x + 400, y + 100), (0, 0, 0), -1)
        
        cv2.putText(img, name, (x + 10, y + 30), self.font, 0.7, (255, 255, 255), 2)
        cv2.putText(img, f"FPS: {stats['fps']:.1f}", (x + 10, y + 50), self.font, 0.6, (0, 255, 0), 2)
        cv2.putText(img, f"Inference: {stats['mean_inference']:.1f}ms", (x + 10, y + 70), self.font, 0.6, (255, 165, 0), 2)
        cv2.putText(img, f"Total: {stats['total_time']:.1f}ms", (x + 10, y + 90), self.font, 0.6, (0, 165, 255), 2)
        
        return img

    def run_benchmark(self):
        # Initialize CUDA first
        if torch.cuda.is_available():
            torch.cuda.init()
            print(f"CUDA initialized: {torch.cuda.get_device_name(0)}")

        # Load models
        model_pt = YOLO("yolo11x.pt")
        model_pt.export(format="openvino", half=True)
        model_openvino = YOLO("yolo11x_openvino_model/")
        
        # Initialize queues and threads
        frame_queues = {config["name"]: Queue(maxsize=1) for config in self.configs}  # Limit queue size to 1
        result_queue = Queue()
        
        # Create and start threads
        threads = []
        for config in self.configs:
            model = model_openvino if config["model_type"] == "openvino" else model_pt
            thread = ModelThread(frame_queues[config["name"]], result_queue, config, model)
            thread.daemon = True  # Make threads daemon so they exit when main program exits
            threads.append(thread)
            thread.start()
        
        # Initialize video capture and writer
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FPS, 30)  # Set camera FPS
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 30.0  # Set output video FPS
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = cv2.VideoWriter(f'threaded_benchmark_{timestamp}.mp4', fourcc, fps, (self.width, self.height))
        
        # Initialize frame rate control
        frame_time = 1/fps  # Time per frame in seconds
        
        # Initialize statistics
        running_stats = {config["name"]: {
            'fps': 0,
            'mean_inference': 0,
            'total_time': 0,
            'frames': 0
        } for config in self.configs}
        
        # Initialize frame buffer
        frame_buffer = {}
        
        try:
            # Start timing for first frame
            frame_start_time = time.time()
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Send frame to all threads (non-blocking)
                for name, queue in frame_queues.items():
                    try:
                        queue.put_nowait(frame)
                    except Full:
                        # Skip frame for this model if queue is full
                        pass
                
                # Collect results (non-blocking, with timeout)
                results_collected = False
                collection_start = time.time()
                while time.time() - collection_start < frame_time:  # Try collecting for one frame period
                    try:
                        result = result_queue.get_nowait()
                        frame_buffer[result['config_name']] = result
                        results_collected = True
                        
                        # Update statistics
                        stats = running_stats[result['config_name']]
                        stats['frames'] += 1
                        stats['total_time'] = result['inference_time']
                        stats['mean_inference'] = (stats['mean_inference'] * (stats['frames'] - 1) + 
                                                 result['speed']['inference']) / stats['frames']
                        stats['fps'] = 1000 / stats['total_time']
                    except Empty:
                        if results_collected:  # If we got at least one result, that's good enough
                            break
                        time.sleep(0.001)  # Small sleep to prevent CPU spinning
                
                # Create output layout
                layout = self.create_output_layout(frame)
                
                # Update display for each configuration
                for idx, config in enumerate(self.configs):
                    if config["name"] in frame_buffer:
                        result_frame = frame_buffer[config["name"]]['frame']
                        grid_x = (idx % 2) * frame.shape[1]
                        grid_y = (idx // 2) * frame.shape[0]
                        
                        layout[grid_y:grid_y + frame.shape[0], 
                              grid_x:grid_x + frame.shape[1]] = result_frame
                        
                        self.draw_stats(layout, running_stats[config["name"]], 
                                      (grid_x + 10, grid_y + 10), config["name"])
                
                # Resize and add timestamp
                output_frame = cv2.resize(layout, (self.width, self.height))
                cv2.putText(output_frame, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                          (10, self.height - 20), self.font, 0.7, (255, 255, 255), 2)
                
                out.write(output_frame)
                cv2.imshow('Threaded Benchmark Comparison', output_frame)
                
                # Control frame rate
                frame_end_time = time.time()
                time_diff = frame_end_time - frame_start_time
                if time_diff < frame_time:
                    time.sleep(frame_time - time_diff)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                # Start timing for next frame
                frame_start_time = time.time()
                
        finally:
            # Clean up
            for thread in threads:
                thread.running = False
            for queue in frame_queues.values():
                queue.put(None)
            for thread in threads:
                thread.join()
                
            cap.release()
            out.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    visualizer = BenchmarkVisualizer()
    visualizer.run_benchmark()