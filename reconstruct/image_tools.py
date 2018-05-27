import cv2
import numpy as np
from numpy import ndarray
from datetime import datetime
from time import time

def line_filter(image: ndarray, line_search_mask: ndarray, orientation_is_vertical: bool):
    blue_component = image * np.array([0, 0, 1])
    height, width, channels = image.shape

    start_time = time()
    search_pixels = [(x, y) for y in range(height) for x in range(width) if line_search_mask[y][x] == 255]
    print(len(search_pixels))
    print((time() - start_time) * 1000)