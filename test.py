import torch
import numpy as np
import OXRPresentationPython as oxr
import time

def create_test_pattern(height, width, is_left_eye=True):
    """Create a simple test pattern with red and blue gradients for left/right eyes"""
    # Create a tensor with shape (H, W, 4) for RGBA
    tensor = torch.zeros((height, width, 4), dtype=torch.float32, device='cuda')
    
    # Create coordinate grids on CUDA
    y = torch.linspace(0, 1, height, device='cuda').view(-1, 1).expand(height, width)
    x = torch.linspace(0, 1, width, device='cuda').view(1, -1).expand(height, width)
    
    # Create a circular gradient
    center_x = 0.5
    center_y = 0.5
    distance = torch.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # Create color pattern
    if is_left_eye:
        # Left eye: red gradient
        tensor[..., 0] = 1.0 - distance  # Red channel
    else:
        # Right eye: blue gradient
        tensor[..., 2] = 1.0 - distance  # Blue channel
    
    # Add some rings
    rings = torch.sin(distance * 20) * 0.5 + 0.5
    tensor[..., 0] = tensor[..., 0] * rings if is_left_eye else tensor[..., 0]
    tensor[..., 2] = tensor[..., 2] * rings if not is_left_eye else tensor[..., 2]
    
    # Set alpha channel to fully opaque
    tensor[..., 3] = 1.0
    
    return tensor

def main():
    try:
        # Initialize the VR system
        print("Initializing VR system...")
        dims = oxr.initialize()
        width, height = dims["width"], dims["height"]
        print(f"Swapchain dimensions: {width}x{height}")

        # Main render loop
        try:
            while True:
                # Start new frame and get eye matrices
                frame_info = oxr.startFrame()
                
                # If frame_info is empty, skip this frame
                if not frame_info:
                    continue

                # Create test patterns for each eye
                left_eye = create_test_pattern(height, width, is_left_eye=True)
                right_eye = create_test_pattern(height, width, is_left_eye=False)
                
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