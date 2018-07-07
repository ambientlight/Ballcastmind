from typing import Sequence, Any, Generator, Optional, Dict, List
from types import SimpleNamespace as Namespace
from os.path import isfile
import json
import numpy as np

from hyperopt import hp
from keras.models import Model
from keras import Input, layers

from src.model_descriptor import ModelDescriptor
from src.threadsafe_generator import threadsafe_generator

data_directory_path = './data'
sample_name = '10_min_sample'
sample_id = 'cebd5588-926b-486b-8add-bbe1e74a1226_v2'
sample_dir = f'{data_directory_path}/input/{sample_name}_{sample_id}'


class LineFeaturesTests(ModelDescriptor):
    _version = 0
    _miniBatchSize = 32

    def __init__(self, name: str, model_dir_path: str):
        super().__init__(name, model_dir_path)

    def create_model(self, space: Optional[Dict[str, Any]] = None) -> Model:
        if space:
            print('Using hyperopt space:')
            print(space)

        rec_history = space['REC_HISTORY'] if space and space['REC_HISTORY'] else 10
        dense_units = space['DENSE_SIZE'] if space and space['DENSE_SIZE'] else 2048
        dropout_size = space['DROPOUT'] if space and space['DROPOUT'] else 0.25
        last_dense = space['LAST_DENSE'] if space and space['LAST_DENSE'] else 1

        line_input = Input(shape=(32, 4), dtype='float32')
        if space and space['REC_TYPE'] != 'NONE':
            x = layers.GRU(rec_history)(line_input) if space and space['REC_TYPE'] == 'GRU' else layers.LSTM(rec_history)(line_input)
            x = layers.Dense(dense_units, activation='relu')(x)
        else:
            x = layers.Dense(dense_units, activation='relu')(line_input)

        if space and space['REC_TYPE'] == 'NONE':
            x = layers.Flatten()(x)

        x = layers.Dropout(dropout_size)(x)

        if last_dense == 1:
            rot_x_pred = layers.Dense(last_dense, name='rot_x')(x)
        else:
            x = layers.Dense(last_dense, name='rot_x_pre')(x)
            rot_x_pred = layers.Dense(1, name='rot_x')(x)

        # rot_y_pred = layers.Dense(1, name='rot_y')(x)
        # rot_z_pred = layers.Dense(1, name='rot_z')(x)
        # fov_pred = layers.Dense(1, name='fov')(x)

        model = Model(line_input, rot_x_pred)
        model.compile(optimizer='adam',
                      loss='mse')
        return model

    def hyperopt_space(self):
        space = {}
        space['REC_TYPE'] = hp.choice('REC_TYPE', ['NONE', 'GRU', 'LSTM'])
        space['REC_HISTORY'] = hp.choice('REC_HISTORY', [2, 4, 6, 8, 10, 16, 24, 32])
        space['DENSE_SIZE'] = hp.choice('DENSE_SIZE', [128, 256, 512, 1024, 2048, 4096])
        space['DROPOUT'] = hp.choice('DROPOUT', [0.05, 0.1, 0.2, 0.25, 0.33, 0.5, 0.66, 0.75, 0.8])
        space['LAST_DENSE'] = hp.choice('LAST_DENSE', [1, 2, 4, 8, 16, 32, 64, 128])
        return space

    def data_locators(self, skip_filter: bool = False) -> List[Any]:
        frame_data_path = f'{sample_dir}/frame_data_minimized.json'
        if not isfile(frame_data_path):
            print("frame_data_minimized.json hasn't found in sample directory")
            exit(1)

        with open(frame_data_path, encoding='utf-8') as frame_data_path:
            frame_data = json.load(frame_data_path, object_hook=lambda d: Namespace(**d))

        return [frame_sample for frame_sample in frame_data if skip_filter or frame_sample.type == 'regular']

    @threadsafe_generator
    def data_generator(self, data_locators: Sequence[Any], batch_size: int) -> Generator:
        epoch = 0
        while 1:
            epoch += 1
            target_samples = data_locators[:]
            read_count = 0

            while read_count < (len(target_samples) - batch_size):
                batch_frame_samples = target_samples[read_count: read_count + batch_size]
                xs = np.array([self.load_lines(frame_sample) for frame_sample in batch_frame_samples])
                rot_xs = np.array([frame_sample.minimized_camera.rotation.x if hasattr(frame_sample, 'minimized_camera') else frame_sample.camera.rotation.x for frame_sample in batch_frame_samples])
                # rot_ys = np.array([frame_sample.camera.rotation.y for frame_sample in batch_frame_samples])
                # rot_zs = np.array([frame_sample.camera.rotation.z for frame_sample in batch_frame_samples])
                # fovs = np.array([frame_sample.camera.fov for frame_sample in batch_frame_samples])
                read_count += batch_size
                yield (xs, rot_xs)

    @staticmethod
    def load_lines(frame_sample):
        reduced_lines: List[[int, int, int, int]] = frame_sample.reducedLines
        if len(reduced_lines) > 32:
            reduced_lines = frame_sample.reducedLines[:32]
        else:
            for i in range(len(reduced_lines), 32):
                reduced_lines.append([0, 0, 0, 0])

        target_lines = []
        for x1, y1, x2, y2 in [
            (reduced_line[0] / frame_sample.canvasSize.width, reduced_line[1] / frame_sample.canvasSize.height,
             reduced_line[2] / frame_sample.canvasSize.width, reduced_line[3] / frame_sample.canvasSize.height) for reduced_line in reduced_lines]:
            target_lines.append([x1, y1, x2, y2])

        target = np.array(target_lines, dtype='float32')
        return target
