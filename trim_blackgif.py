import imageio
import numpy as np

def is_black_frame(frame, threshold=10):
    """Check if frame is predominantly black."""
    mean_value = np.mean(frame)
    return mean_value < threshold

def trim_black_frames_gif(input_path, output_path):
    # Read the GIF
    reader = imageio.get_reader(input_path)
    
    # Find first non-black frame
    frames_to_keep = []
    found_first_frame = False
    
    for frame in reader:
        if not found_first_frame:
            if not is_black_frame(frame):
                found_first_frame = True
                frames_to_keep.append(frame)
        else:
            frames_to_keep.append(frame)
    
    # Write the new GIF
    imageio.mimsave(output_path, frames_to_keep, duration=1/30)  # Assuming 30fps
    print(f"Processed GIF saved to {output_path}")

if __name__ == "__main__":
    input_gif = "ezgif-3b89b35cbf3cc0.gif"
    output_gif = "trimmed_output.gif"
    trim_black_frames_gif(input_gif, output_gif)