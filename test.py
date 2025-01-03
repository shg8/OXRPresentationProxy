import torch
import numpy as np
import OXRPresentationPython as oxr
import time

def create_test_pattern(height, width, is_left_eye=True, t=0.0):
    """Create a simple test pattern with red and blue gradients for left/right eyes"""
    # Create coordinate grids on CUDA
    y = torch.linspace(0, 1, height, device='cuda').view(-1, 1).expand(height, width)
    x = torch.linspace(0, 1, width, device='cuda').view(1, -1).expand(height, width)
    
    # Convert time to tensor and move to CUDA
    t = torch.tensor(t, device='cuda')
    
    # Create a circular gradient with moving center
    center_x = 0.5 + 0.2 * torch.cos(t)
    center_y = 0.5 + 0.2 * torch.sin(t)
    distance = torch.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # Create animated rings pattern
    rings = torch.sin(distance * 20 - t * 2) * 0.5 + 0.5
    gradient = 1.0 - distance
    
    # Create RGBA tensor (first as float32 for calculations)
    tensor = torch.zeros((height, width, 4), dtype=torch.float32, device='cuda')
    
    # Fill color channels with rotating hue
    if is_left_eye:
        # Left eye: red gradient with slight green
        tensor[..., 0] = gradient * rings  # Red channel
        tensor[..., 1] = gradient * rings * 0.3 * (torch.sin(t) * 0.5 + 0.5)  # Green channel
    else:
        # Right eye: blue gradient with slight green
        tensor[..., 2] = gradient * rings  # Blue channel
        tensor[..., 1] = gradient * rings * 0.3 * (torch.cos(t) * 0.5 + 0.5)  # Green channel
    
    # Set alpha channel to fully opaque
    tensor[..., 3] = 1.0
    
    # Convert to uint8 (multiply by 255 and clamp)
    tensor = (tensor * 255).clamp(0, 255).to(torch.uint8)
    
    # Ensure the tensor is contiguous and in the correct format
    tensor = tensor.contiguous()
    
    return tensor

def main():
    try:
        # Initialize the VR system
        print("Initializing VR system...")
        dims = oxr.initialize()
        width, height = dims["width"], dims["height"]
        print(f"Swapchain dimensions: {width}x{height}")

        # Main render loop
        print("Starting render loop...")
        start_time = time.time()
        try:
            while True:
                # Get current time for animation
                current_time = time.time() - start_time
                
                # Start new frame and get eye matrices
                frame_info = oxr.startFrame()
                
                # If frame_info is empty, skip this frame
                if not frame_info:
                    continue
                
                # Create animated test patterns for both eyes
                left_eye = create_test_pattern(height, width, is_left_eye=True, t=current_time)
                right_eye = create_test_pattern(height, width, is_left_eye=False, t=current_time)
                
                # Submit the frame
                oxr.submitFrame(left_eye, right_eye)
                
                # Small sleep to prevent maxing out CPU
                time.sleep(0.001)

        except KeyboardInterrupt:
            print("\nStopping render loop...")

    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        # Clean up
        print("Cleaning up...")
        oxr.cleanup()

if __name__ == "__main__":
    main() 