import cv2
import numpy as np
from numpy import ndarray
from datetime import datetime
from time import time


def line_filter(image: ndarray, line_search_mask: ndarray, grass_mask: ndarray, line_width: int):
    start_time = time()

    # line_search_space_only = cv2.bitwise_and(image, image, mask=line_search_mask)
    blue_component = np.array(image * np.array([0, 0, 1], dtype=np.uint8), dtype=np.uint8)

    half_linewidth_shifted_down = np.roll(blue_component, round(line_width / 2), 0)
    half_linewidth_shifted_up = np.roll(blue_component, -round(line_width / 2), 0)
    diff = np.minimum(
        blue_component - half_linewidth_shifted_down,
        blue_component - half_linewidth_shifted_up
    )

    target = cv2.bitwise_and(diff, diff, mask=cv2.bitwise_not(grass_mask))
    # turn into a mask
    target_mask = cv2.inRange(target, np.array([0, 0, 1]), np.array([0, 0, 255]))

    print(f'line_filter: {(time() - start_time) * 1000} ms')
    return target_mask
