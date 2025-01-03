from typing import Dict, Union, TypedDict
import torch
import numpy.typing as npt
import numpy as np

class SwapchainDimensions(TypedDict):
    width: int
    height: int

class FrameInfo(TypedDict):
    left_view_matrix: npt.NDArray[np.float32]
    left_projection_matrix: npt.NDArray[np.float32]
    right_view_matrix: npt.NDArray[np.float32]
    right_projection_matrix: npt.NDArray[np.float32]

def initialize() -> SwapchainDimensions: ...
def cleanup() -> None: ...
def startFrame() -> FrameInfo: ...
def submitFrame(leftEyeTensor: torch.Tensor, rightEyeTensor: torch.Tensor) -> None: ... 