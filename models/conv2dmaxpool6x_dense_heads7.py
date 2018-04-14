from typing import Sequence, Any, Generator, Optional, Dict, List
from os.path import isfile
from types import SimpleNamespace as Namespace
import numpy
import json

from keras.models import Model
from keras.preprocessing.image import load_img, img_to_array
from keras import Input, layers
from hyperopt import hp
from random import shuffle

from src.model_descriptor import ModelDescriptor
from src.threadsafe_generator import threadsafe_generator


data_directory_path = './data'
sample_name = '10_min_sample'
sample_id = 'cebd5588-926b-486b-8add-bbe1e74a1226'
sample_dir = f'{data_directory_path}/input/{sample_name}_{sample_id}'


class Conv2dMaxpool6xDenseHeads7(ModelDescriptor):
    _version = 3

    def __init__(self, name: str, model_dir_path: str):
        super().__init__(name, model_dir_path)

    def create_model(self, space: Optional[Dict[str, Any]] = None) -> Model:
        if space:
            print('Using hyperopt space:')
            print(space)

        for_optimization = True if space else False

        img_input = Input(shape=(640, 400, 3), dtype='float32')
        x = layers.Conv2D(32 if not for_optimization else space['Conv2D_0'], 3, activation='relu')(img_input)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Conv2D(64 if not for_optimization else space['Conv2D_1'], 3, activation='relu')(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Conv2D(128 if not for_optimization else space['Conv2D_2'], 3, activation='relu')(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Conv2D(128 if not for_optimization else space['Conv2D_3'], 3, activation='relu')(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Conv2D(128 if not for_optimization else space['Conv2D_4'], 3, activation='relu')(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Conv2D(128 if not for_optimization else space['Conv2D_5'], 3, activation='relu')(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(512, activation='relu')(x)
        pos_x_pred = layers.Dense(1, name='pos_x')(x)
        pos_y_pred = layers.Dense(1, name='pos_y')(x)
        pos_z_pred = layers.Dense(1, name='pos_z')(x)
        rot_x_pred = layers.Dense(1, name='rot_x')(x)
        rot_y_pred = layers.Dense(1, name='rot_y')(x)
        rot_z_pred = layers.Dense(1, name='rot_z')(x)
        fov_pred = layers.Dense(1, name='fov')(x)

        model = Model(img_input, [pos_x_pred, pos_y_pred, pos_z_pred, rot_x_pred, rot_y_pred, rot_z_pred, fov_pred])
        model.compile(optimizer='rmsprop',
                      loss={'pos_x': 'mse',
                            'pos_y': 'mse',
                            'pos_z': 'mse',
                            'rot_x': 'mse',
                            'rot_y': 'mse',
                            'rot_z': 'mse',
                            'fov': 'mse'},
                      metrics=['mae'])
        return model

    def hyperopt_space(self):
        space = {}
        for i in range(6):
            key = f'Conv2D_{i}'
            space[key] = hp.choice(key, [32, 64, 128])

        return space

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
                xs = numpy.array([Conv2dMaxpool6xDenseHeads7.frame_sample_load_image(frame_sample, target_size)
                                  for frame_sample in batch_frame_samples])
                pos_xs = numpy.array([frame_sample.camera.position.x for frame_sample in batch_frame_samples])
                pos_ys = numpy.array([frame_sample.camera.position.y for frame_sample in batch_frame_samples])
                pos_zs = numpy.array([frame_sample.camera.position.z for frame_sample in batch_frame_samples])
                rot_xs = numpy.array([frame_sample.camera.rotation.x for frame_sample in batch_frame_samples])
                rot_ys = numpy.array([frame_sample.camera.rotation.y for frame_sample in batch_frame_samples])
                rot_zs = numpy.array([frame_sample.camera.rotation.z for frame_sample in batch_frame_samples])
                fovs = numpy.array([frame_sample.camera.fov for frame_sample in batch_frame_samples])
                read_count += batch_size
                yield (xs, [pos_xs, pos_ys, pos_zs, rot_xs, rot_ys, rot_zs, fovs])

    @staticmethod
    def frame_sample_load_image(frame_sample, target_size):
        frame_image_path = f'{sample_dir}/{frame_sample.imagePath}'
        return img_to_array(load_img(frame_image_path, target_size=target_size)) / 255.0
