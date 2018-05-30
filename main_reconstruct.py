from types import SimpleNamespace as Namespace
from typing import List, Dict, Any, Tuple
from os.path import isfile
import json
import cv2
from pprint import pformat
from matplotlib import pyplot as plt
from xml.etree import ElementTree
import re
import numpy as np
from numpy import ndarray
from math import sin, cos
from math import radians
from time import time
from numpy.linalg import lstsq

from reconstruct.core import project, linear_parameters, cut_off_line, buffer_lines, line_orientation_is_vertical, \
    point_infinite_line_distance
from reconstruct.perspective_camera import PerspectiveCamera
from reconstruct.image_tools import line_filter

SEARCH_WINDOW_CORNER_CUTOFF = 20
METER_TO_YARD_MULT = 1.093613298337708
SEARCH_WINDOW_RADIUS = 40
LOWER_GRASS_GREEN = np.array([40, 40, 40])
UPPER_GRASS_GREEN = np.array([80, 200, 180])


data_directory_path = './data'
sample_name = '10_min_sample'
sample_id = 'cebd5588-926b-486b-8add-bbe1e74a1226_v2'
sample_dir = f'{data_directory_path}/input/{sample_name}_{sample_id}'


def read_regular_frame_data():
    frame_data_path = f'{sample_dir}/frame_data.json'
    if not isfile(frame_data_path):
        print("frame_data.json hasn't been found in sample directory")
        exit(1)

    with open(frame_data_path, encoding='utf-8') as frame_data_file:
        frame_data = json.load(frame_data_file, object_hook=lambda d: Namespace(**d))

    return [frame_sample for frame_sample in frame_data if frame_sample.type == 'regular']


def dist(unprojected_line: ndarray, target_point: ndarray, camera: PerspectiveCamera):
    projected_line = np.array([
        project(unprojected_line[0], camera, last_frame_data.canvasSize.width, last_frame_data.canvasSize.height),
        project(unprojected_line[1], camera, last_frame_data.canvasSize.width, last_frame_data.canvasSize.height)
    ], dtype=np.float64)
    return point_infinite_line_distance(target_point, projected_line)


def ddist_dtheta_x(unprojected_line: ndarray, target_point: ndarray, camera: PerspectiveCamera):
    h = 1e-6
    cam1 = camera.copy()
    cam2 = camera.copy()
    cam1.rotation = np.array([cam1.rotation[0] + h, cam1.rotation[1], cam1.rotation[2]])
    cam2.rotation = np.array([cam1.rotation[0] - h, cam1.rotation[1], cam1.rotation[2]])
    return (dist(unprojected_line, target_point, cam1) - dist(unprojected_line, target_point, cam2)) / (2 * h)


def ddist_dtheta_y(unprojected_line: ndarray, target_point: ndarray, camera: PerspectiveCamera):
    h = 1e-6
    cam1 = camera.copy()
    cam2 = camera.copy()
    cam1.rotation = np.array([cam1.rotation[0], cam1.rotation[1] + h, cam1.rotation[2]])
    cam2.rotation = np.array([cam1.rotation[0], cam1.rotation[1] - h, cam1.rotation[2]])
    return (dist(unprojected_line, target_point, cam1) - dist(unprojected_line, target_point, cam2)) / (2 * h)


def ddist_dtheta_z(unprojected_line: ndarray, target_point: ndarray, camera: PerspectiveCamera):
    h = 1e-6
    cam1 = camera.copy()
    cam2 = camera.copy()
    cam1.rotation = np.array([cam1.rotation[0], cam1.rotation[1], cam1.rotation[2] + h])
    cam2.rotation = np.array([cam1.rotation[0], cam1.rotation[1], cam1.rotation[2] - h])
    return (dist(unprojected_line, target_point, cam1) - dist(unprojected_line, target_point, cam2)) / (2 * h)


def ddist_dfov(unprojected_line: ndarray, target_point: ndarray, camera: PerspectiveCamera):
    h = 1e-6
    cam1 = camera.with_new_fov(camera.fov + h)
    cam2 = camera.with_new_fov(camera.fov - h)
    return (dist(unprojected_line, target_point, cam1) - dist(unprojected_line, target_point, cam2)) / (2 * h)


last_frame_data = read_regular_frame_data()[-1]
frame_image_path = f'{sample_dir}/{last_frame_data.imagePath}'
frame_image = cv2.imread(frame_image_path)
frame_image = cv2.cvtColor(frame_image, cv2.COLOR_BGR2RGB)

# no camera roll is assumed, instead let pitch to have incline and minimize for it aswell later
pitch_tilt = last_frame_data.camera.rotation.z
# print(camera_pose)

line_ids = [
    'L-16y-Bottom',
    'L-16y-Right',
    'L-16y-Top',
    'L-6y-Right',
    'L-6y-Bottom',
    'L-6y-Top',
    'R-16y-Bottom',
    'R-16y-Right',
    'R-16y-Top',
    'R-6y-Right',
    'R-6y-Top',
    'R-GL',
    'L-GL',
    'SL-Top',
    'SL-Bottom',
    'CL'
]

parse_start_time = time()
pitch_name = 'anfield_liverpool'
pitch_model_path = f'{data_directory_path}/pitch_models/{pitch_name}.svg'
svg_root = ElementTree.parse(pitch_model_path).getroot()


line_elements = [svg_root.find(f".//*[@id='{line_id}']") for line_id in line_ids]
line_d_attribs = [line_element.attrib['d'] for line_element in line_elements]
coord_lines_tuples = [re.search('M(\d+),(\d+)\s+L(\d+),(\d+)', line_d_attrib).groups()
                      for line_d_attrib in line_d_attribs]


coords = np.array([np.array([[coord_tuple[0], coord_tuple[1]], [coord_tuple[2], coord_tuple[3]]])
                   for coord_tuple in coord_lines_tuples], dtype=np.float32)
print(f'parse: {(time() - parse_start_time) * 1000} ms')

proj_start_time = time()
# using center line dimentions readjust the coordinate system to start from field center
cl_coords = coords[len(coords)-1]
center = [cl_coords[0][0], cl_coords[1][1] / 2]
# print(f'Center: {center}')
coords = coords - center

# divide by svg-model yard scaling factor
coords = coords / 10.0
# convert to meters
coords = coords / METER_TO_YARD_MULT

line_dict = {}
for index, line_name in enumerate(line_ids):
    line_dict[line_name] = coords[index]

camera = PerspectiveCamera(last_frame_data.camera.fov, last_frame_data.canvasSize.width / last_frame_data.canvasSize.height, 0.1, 1000)
camera.position = np.array([
    last_frame_data.camera.position.x,
    last_frame_data.camera.position.y,
    last_frame_data.camera.position.z])
camera.rotation = np.array([
    radians(last_frame_data.camera.rotation.x),
    radians(last_frame_data.camera.rotation.y),
    radians(last_frame_data.camera.rotation.z)])

overlay = np.empty(frame_image.shape, np.uint8)
line_search_mask = np.empty((frame_image.shape[0], frame_image.shape[1]), np.uint8)
line_dicts: List[Dict[str, Any]] = []
for line_id in line_ids:
    p1 = np.array([line_dict[line_id][0][0], line_dict[line_id][0][1], 0.], dtype=np.float32)
    p2 = np.array([line_dict[line_id][1][0], line_dict[line_id][1][1], 0.], dtype=np.float32)

    p1_proj = project(p1, camera, last_frame_data.canvasSize.width, last_frame_data.canvasSize.height)
    p2_proj = project(p2, camera, last_frame_data.canvasSize.width, last_frame_data.canvasSize.height)

    projected = np.array([p1_proj, p2_proj], dtype=np.int32)
    line = cut_off_line(projected, SEARCH_WINDOW_CORNER_CUTOFF)
    lines = buffer_lines(line, SEARCH_WINDOW_RADIUS).astype(np.int32)
    reshaped_lines = lines.reshape((-1, 1, 2))
    cv2.fillPoly(overlay, [reshaped_lines], (0, 0, 255))
    cv2.fillPoly(line_search_mask, [reshaped_lines], (255, 255, 255))

    line_dicts.append({
        'original': np.array([p1, p2]),
        'projected': projected,
        'reduced_projected': line,
        'buffer_contour': lines.reshape((-1, 1, 2)),
        'id': line_id,
        'orientation': line_orientation_is_vertical(projected)
    })

frame_image_line_space = cv2.addWeighted(frame_image, 0.9, overlay, 0.1, 0)
# frame_image = cv2.bitwise_and(frame_image, frame_image, mask=line_search_mask)

hsv_frame = cv2.cvtColor(frame_image.copy(), cv2.COLOR_BGR2HSV)
grass_mask = cv2.inRange(hsv_frame, LOWER_GRASS_GREEN, UPPER_GRASS_GREEN)
grass_only_frame_image = cv2.bitwise_and(frame_image, frame_image, mask=grass_mask)

print(f'proj_fill: {(time() - proj_start_time) * 1000} ms')

filtered_lines_mask = line_filter(
    grass_only_frame_image,
    line_search_mask,
    grass_mask,
    15)

hough_start_time = time()
lines: List[List[Tuple[int, int, int, int]]] = cv2.HoughLinesP(
    filtered_lines_mask,
    rho=1, theta=np.pi/180, threshold=150, minLineLength=150, maxLineGap=20
)
print(f'hough_lines_p: {(time() - hough_start_time) * 1000} ms')

# print(f'Line count: {len(lines)}')
for line in lines:
    for x1, y1, x2, y2 in line:
        cv2.line(frame_image_line_space, (x1, y1), (x2, y2), (255, 0, 0), 2)

for line_dict in line_dicts:
    projected = line_dict['projected']
    cv2.line(frame_image_line_space,
             (projected[0][0], projected[0][1]),
             (projected[1][0], projected[1][1]), (0, 255, 0), 2)

group_by_buffer_start_time = time()
for line_dict in line_dicts:
    reshaped_lines = line_dict['buffer_contour']
    match_idxs = []
    for line_index, line in enumerate(lines):
        x1, y1, x2, y2 = line[0]
        p1_res = cv2.pointPolygonTest(reshaped_lines, (x1, y1), True)
        p2_res = cv2.pointPolygonTest(reshaped_lines, (x2, y2), True)

        # print(f'{line_dict["id"]}: {p1_res}, {p2_res}')
        if (p1_res > -30) and (p2_res > -30):
            match_idxs.append(line_index)

    line_dict['observed'] = []
    if len(match_idxs) > 0:
        line_dict['observed'] = np.array([
            [[lines[idx][0][0], lines[idx][0][1]],
             [lines[idx][0][2], lines[idx][0][3]]] for idx in match_idxs], dtype=np.int32)
        # remove extracted lines from the list
        lines = [line for line_index, line in enumerate(lines) if line_index not in match_idxs]

    # print(f'{line_dict["id"]}: {len(line_dict["extracted"])}')

print(f'group_by_buffer: {(time() - group_by_buffer_start_time) * 1000} ms')

A_rows = []
bs = []
for line_dict in [line_dict for line_dict in line_dicts if len(line_dict['observed']) > 0]:
    # print(line_dict['id'])
    # for observed_line in line_dict['observed']:
    #     print(f'{point_infinite_line_distance(observed_line[0], line_dict["projected"])}, '
    #           f'{point_infinite_line_distance(observed_line[1], line_dict["projected"])}')

    dtheta_x_0 = ddist_dtheta_x(line_dict['original'], line_dict['observed'][0][0], camera)
    dtheta_y_0 = ddist_dtheta_y(line_dict['original'], line_dict['observed'][0][0], camera)
    dtheta_z_0 = ddist_dtheta_z(line_dict['original'], line_dict['observed'][0][0], camera)
    dfov_0 = ddist_dfov(line_dict['original'], line_dict['observed'][0][0], camera)

    dtheta_x_1 = ddist_dtheta_x(line_dict['original'], line_dict['observed'][0][1], camera)
    dtheta_y_1 = ddist_dtheta_y(line_dict['original'], line_dict['observed'][0][1], camera)
    dtheta_z_1 = ddist_dtheta_z(line_dict['original'], line_dict['observed'][0][1], camera)
    dfov_1 = ddist_dfov(line_dict['original'], line_dict['observed'][0][1], camera)

    A_rows.append(np.array([dtheta_x_0, dtheta_y_0, dtheta_z_0, dfov_0], dtype=np.float32))
    A_rows.append(np.array([dtheta_x_1, dtheta_y_1, dtheta_z_1, dfov_1], dtype=np.float32))
    bs.append(-point_infinite_line_distance(line_dict['observed'][0][0], line_dict['projected']))
    bs.append(-point_infinite_line_distance(line_dict['observed'][0][1], line_dict['projected']))

A = np.array(A_rows, dtype=np.float32)
b = np.array(bs, dtype=np.float32)

# print(A)
# print(b)

x, residuals, rank, s = lstsq(A, b, rcond=None)
# print(x)

camera.rotation = np.array([camera.rotation[0] + x[0], camera.rotation[1] + x[1], camera.rotation[2] + x[2]])
camera = camera.with_new_fov(camera.fov + x[3])

for line_dict in line_dicts:
    p1 = line_dict['original'][0]
    p2 = line_dict['original'][1]
    p1_proj = project(p1, camera, last_frame_data.canvasSize.width, last_frame_data.canvasSize.height)
    p2_proj = project(p2, camera, last_frame_data.canvasSize.width, last_frame_data.canvasSize.height)
    line = np.array([p1_proj, p2_proj], dtype=np.int32)
    cv2.line(frame_image_line_space,
             (line[0][0], line[0][1]),
             (line[1][0], line[1][1]), (0, 0, 255), 2)

# plot frame_image
plt.imshow(frame_image_line_space)
plt.title('Frame Image')
plt.xticks([])
plt.yticks([])
plt.show()

# ax1 = fig.add_subplot(2,1,1)
# ax1.imshow(target)
# ax2 = fig.add_subplot(2,1,2)
# ax2.imshow(frame_image)
# plt.show()