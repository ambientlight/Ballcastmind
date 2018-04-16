from typing import Sequence, Any, Generator, Optional, Dict, List
from os.path import isfile
from types import SimpleNamespace as Namespace
import numpy
import json

from keras.models import Model
from keras.preprocessing.image import load_img, img_to_array
from keras import Input, layers
from keras.applications.inception_v3 import conv2d_bn
from keras.backend import image_data_format

from random import shuffle


from src.model_descriptor import ModelDescriptor
from src.threadsafe_generator import threadsafe_generator


data_directory_path = './data'
sample_name = '10_min_sample'
sample_id = 'cebd5588-926b-486b-8add-bbe1e74a1226_v2'
sample_dir = f'{data_directory_path}/input/{sample_name}_{sample_id}'


def inception_test_module(x, filters: int):
    if image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    branch_a = layers.Conv2D(filters, 1,
                             activation='relu', strides=2)(x)

    branch_b = layers.Conv2D(filters, 1, activation='relu', padding='same')(x)
    branch_b = layers.Conv2D(filters, 3, activation='relu', strides=2, padding='same')(branch_b)

    branch_c = layers.AveragePooling2D(3, strides=2, padding='same')(x)
    branch_c = layers.Conv2D(filters, 3, activation='relu', padding='same')(branch_c)

    branch_d = layers.Conv2D(filters, 1, activation='relu', padding='same')(x)
    branch_d = layers.Conv2D(filters, 3, activation='relu', padding='same')(branch_d)
    branch_d = layers.Conv2D(filters, 3, activation='relu', strides=2, padding='same')(branch_d)

    return layers.concatenate(
        [branch_a, branch_b, branch_c, branch_d], axis=channel_axis)


def inception_v3_mixed_module(x):
    if image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    branch1x1 = conv2d_bn(x, 64, 1, 1)
    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)
    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch_pool = layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 32, 1, 1)
    return layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis)


class InceptionTest(ModelDescriptor):
    _version = 0

    def __init__(self, name: str, model_dir_path: str):
        super().__init__(name, model_dir_path)

    def create_model(self, space: Optional[Dict[str, Any]] = None) -> Model:
        if space:
            print('Using hyperopt space:')
            print(space)

        img_input = Input(shape=(640, 400, 3), dtype='float32')
        x = layers.SeparableConv2D(32, 5, activation='relu')(img_input)
        x = layers.MaxPooling2D(2)(x)
        # x = inception_test_module(img_input, 32)

        x = layers.SeparableConv2D(64, 5, activation='relu')(x)
        x = layers.MaxPooling2D(2)(x)
        # x = inception_test_module(x, 64)

        x = layers.SeparableConv2D(128, 5, activation='relu')(x)
        x = layers.MaxPooling2D(2)(x)
        # x = inception_test_module(x, 128)

        # x = layers.SeparableConv2D(128, 5, activation='relu')(x)
        # x = layers.MaxPooling2D(2)(x)
        x = inception_test_module(x, 128)

        # x = layers.SeparableConv2D(128, 5, activation='relu')(x)
        # x = layers.MaxPooling2D(2)(x)
        x = inception_test_module(x, 128)

        x = layers.Flatten()(x)
        x = layers.Dense(512, activation='relu')(x)
        rot_x_pred = layers.Dense(1, name='rot_x')(x)

        model = Model(img_input, rot_x_pred)
        model.compile(optimizer='rmsprop',
                      loss='mae')
        return model

    def hyperopt_space(self):
        return {}

    def data_locators(self) -> List[Any]:
        frame_data_path = f'{sample_dir}/frame_data.json'
        if not isfile(frame_data_path):
            print("frame_data.json hasn't been found in sample directory")
            exit(1)

        with open(frame_data_path, encoding='utf-8') as frame_data_file:
            frame_data = json.load(frame_data_file, object_hook=lambda d: Namespace(**d))

        return [frame_sample for frame_sample in frame_data if frame_sample.type == 'regular']

    @threadsafe_generator
    def data_generator(self, data_locators: Sequence[Any], batch_size: int) -> Generator:
        # output dimentions for x
        # (batchsize, width, height, channels)
        # (16, 640, 400, 3)
        target_size = (640, 400)
        epoch = 0

        while 1:
            epoch += 1
            target_samples = data_locators[:]
            shuffle(target_samples)

            read_count = 0
            while read_count < (len(target_samples) - batch_size):
                batch_frame_samples = target_samples[read_count: read_count + batch_size]
                xs = numpy.array([self.frame_sample_load_image(frame_sample, target_size)
                                  for frame_sample in batch_frame_samples])
                rot_xs = numpy.array([self.normalized_angle(frame_sample.camera.rotation.x) for frame_sample
                                      in batch_frame_samples])
                read_count += batch_size
                yield (xs, rot_xs)

    @staticmethod
    def frame_sample_load_image(frame_sample, target_size):
        frame_image_path = f'{sample_dir}/{frame_sample.imagePath}'
        return img_to_array(load_img(frame_image_path, target_size=target_size)) / 255.0

    '''
        normalize to -180 to 180 (0 to 1)
    '''
    @staticmethod
    def normalized_angle(angle):
        target = angle
        if target > 180.0:
            target = 180.0
        elif target < -180.0:
            target = -180.0

        return (target + 180.0) / 360.0
