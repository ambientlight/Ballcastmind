import cv2
import numpy as np
from numpy import ndarray
from datetime import datetime
from time import time


def line_filter(image: ndarray, line_search_mask: ndarray, grass_mask: ndarray, line_width: int):
    start_time = time()

    # line_search_space = image
    line_search_space = cv2.bitwise_and(image, image, mask=line_search_mask)
    blue_component = cv2.extractChannel(line_search_space, 2)

    half_linewidth_shifted_down = np.roll(blue_component, round(line_width / 2), 0)
    half_linewidth_shifted_up = np.roll(blue_component, -round(line_width / 2), 0)
    diff = np.minimum(
        blue_component - half_linewidth_shifted_down,
        blue_component - half_linewidth_shifted_up
    )

    target = cv2.bitwise_and(diff, diff, mask=cv2.bitwise_not(grass_mask))
    # turn into a mask
    target_mask = cv2.inRange(target, np.array([1]), np.array([255]))

    print(f'line_filter: {(time() - start_time) * 1000} ms')
    return target_mask
