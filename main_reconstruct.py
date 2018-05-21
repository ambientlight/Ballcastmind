from types import SimpleNamespace as Namespace
from os.path import isfile
import json
import cv2
from pprint import pformat
from matplotlib import pyplot as plt
from xml.etree import ElementTree
import re
import numpy as np
from math import sin, cos
from math import radians

from reconstruct.core import project
from reconstruct.perspective_camera import PerspectiveCamera
from reconstruct.quaternion import Quaternion
from reconstruct.matrix4 import Matrix4


METER_TO_YARD_MULT = 1.093613298337708

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
    'CL'
]

pitch_name = 'anfield_liverpool'
pitch_model_path = f'{data_directory_path}/pitch_models/{pitch_name}.svg'
svg_root = ElementTree.parse(pitch_model_path).getroot()


line_elements = [svg_root.find(f".//*[@id='{line_id}']") for line_id in line_ids]
line_d_attribs = [line_element.attrib['d'] for line_element in line_elements]
coord_lines_tuples = [re.search('M(\d+),(\d+)\s+L(\d+),(\d+)', line_d_attrib).groups()
                      for line_d_attrib in line_d_attribs]


coords = np.array([np.array([[coord_tuple[0], coord_tuple[1]], [coord_tuple[2], coord_tuple[3]]])
                   for coord_tuple in coord_lines_tuples], dtype=np.float32)

# using center line dimentions readjust the coordinate system to start from field center
cl_coords = coords[len(coords)-1]
center = [cl_coords[0][0], cl_coords[1][1] / 2]
print(f'Center: {center}')
coords = coords - center

# divide by svg-model yard scaling factor
coords = coords / 10.0
# convert to meters
coords = coords / METER_TO_YARD_MULT

# print(coords)
# print(coords.shape)

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

point = np.array([
    line_dict['L-16y-Right'][0][0],
    line_dict['L-16y-Right'][0][1],
    0.
], dtype=np.float32)

print(project(
    point,
    camera,
    last_frame_data.canvasSize.width,
    last_frame_data.canvasSize.height))


# plot frame_image
# plt.imshow(frame_image)
# plt.title('Frame Image')
# plt.xticks([])
# plt.yticks([])
# plt.show()
