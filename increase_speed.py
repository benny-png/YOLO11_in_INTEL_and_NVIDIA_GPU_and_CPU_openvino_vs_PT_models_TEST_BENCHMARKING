import cv2

def speed_up_video(input_path, output_path, speed_factor=2.0):
    """
    Speed up a video by the specified factor.
    Args:
        input_path: Path to input video
        output_path: Path to save the output video
        speed_factor: How much to speed up the video (e.g., 2.0 means 2x speed)
    """
    # Read input video
    cap = cv2.VideoCapture(input_path)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Original FPS: {fps}")
    print(f"Total frames: {total_frames}")
    print(f"New FPS will be: {fps * speed_factor}")
    
    # Create video writer with increased FPS
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps * speed_factor, (width, height))
    
    frame_number = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Write every frame (for smooth fast motion)
        out.write(frame)
        
        frame_number += 1
        if frame_number % 100 == 0:
            print(f"Processed {frame_number}/{total_frames} frames")
    
    cap.release()
    out.release()
    print("Video processing complete!")

if __name__ == "__main__":
    input_video = "threaded_benchmark_20250209_175253.mp4"
    output_video = "final_threaded_benchmark_speedup.mp4"
    speed_up_video(input_video, output_video, speed_factor=2.0)  # 2x speed