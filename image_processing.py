import numpy
from numpy.typing import NDArray 

def find_best_match(input_mask: NDArray[numpy.bool_], 
                    reference_masks: NDArray[numpy.bool_]
                    ) -> tuple[int, int]:
    """
    Compares an input boolean mask (H, W) to a batch of reference masks (N, H, W),
    and returns the index of the best match based on pixel mismatch.

    Args:
        input_mask: A 2D boolean array of shape (H, W).
        reference_masks: A 3D boolean array of shape (N, H, W).

    Returns:
        A tuple (best_index, mismatch_count) where:
            - best_index_int is the index of the reference mask with the fewest mismatches.
            - mismatch_count_int is the number of differing pixels.
    """
    mismatches = input_mask != reference_masks  # Shape: (N, H, W)
    mismatch_counts = mismatches.sum(axis=(1, 2))  # Shape: (N,)
    best_index = numpy.argmin(mismatch_counts)
    best_score = mismatch_counts[best_index]
    best_index_int = int(best_index)
    best_score_int = int(best_score) 
    return best_index_int, best_score_int

def to_ink_mask(arr: NDArray[numpy.float64], 
                threshold: int = 0
                ) -> NDArray[numpy.bool_]:
    """
    Converts an image array (grayscale or RGB) into a binary ink mask.

    Pixels greater than the given threshold are marked as ink (True), others as background (False).
    RGB inputs are converted to grayscale via channel summation.

    Args:
        arr (NDArray): Input image array of shape (H, W) or (H, W, 3).
        threshold (int): Pixel intensity threshold. Defaults to 0.

    Returns:
        NDArray[numpy.bool_]: A 2D boolean mask where True indicates ink regions.
    """
    if arr.ndim == 3 and arr.shape[2] == 3:
        arr = arr.sum(axis=2) # collapsed 3D into 2D array
        
    return arr > threshold

def apply_diagonal_mask_top_left(arr: NDArray[numpy.float64],
                                 strength: float = 0.3
                                ) -> NDArray[numpy.float64]:
    """
    Applies a diagonal top-left corner mask to a 2D or 3D image array.

    A triangular region in the top-left is masked out (multiplied by zero) using
    a diagonal mask scaled by the `strength` parameter. This is useful for
    ignoring irrelevant parts changing accross cards while they should be ignored
    for the sake of template matching.

    Args:
        arr (NDArray[numpy.float64]): Input image array of shape (H, W) or (H, W, 3).
        strength (float): Proportion (0 < strength < 1) of the image to mask diagonally.

    Returns:
        NDArray[numpy.float64]: The masked array, same shape and dtype as input.
    """
    h, w = arr.shape[:2]
    max_x = int(w * strength)
    max_y = int(h * strength)

    mask = numpy.ones((h, w), dtype=numpy.uint8)

    for y in range(max_y):
        x_limit = max_x - int((max_x / max_y) * y)
        mask[y, :x_limit] = 0

    # Apply mask
    if arr.ndim == 3:  # RGB image
        expanded_mask = mask[:, :, None]  # Broadcast over channels
        arr_masked: NDArray[numpy.float64] = arr * expanded_mask
    else: # grayscale image
        arr_masked = arr * mask
    return arr_masked