import numpy as np
import torch

def heatmap_object(img: np.ndarray, bounding_box: dict, heatmap: np.ndarray) -> np.ndarray:
    """
    This function generates the heatmaps over the objects given the input image:

    Args:
        - img (np.ndarray): input image
        - bounding_box (dict): labels with one object bounding box
        - heatmap (np.ndarray): heatmap of the current input img

    Returns:
        - heatmap (np.ndarray): output heatmap with the current object heatmap added
    """

    return heatmap

def sizemap_object(img: np.ndarray, bounding_box: dict, sizemap: np.ndarray) -> np.ndarray:
    """
    This function generates the sizemaps over the objects given the input image:

    Args:
        - img (np.ndarray): input image
        - bounding_box (dict): labels with one object bounding box
        - sizemap (np.ndarray): sizemap of the current input img

    Returns:
        - sizemap (np.ndarray): output sizemap with the current object sizemap added
    """

    return sizemap