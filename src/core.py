import os
import json
import math
import numpy
from pprint import pprint
import matplotlib.pyplot as plt
from types import SimpleNamespace as Namespace
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Model
from keras import Input, layers
from datetime import datetime
from random import shuffle

from src.threadsafe_generator import threadsafe_generator

from models.conv_simple_with_single_on_top import ConvSimpleWithSingleOnTopModel

simple_conv = ConvSimpleWithSingleOnTopModel(name='Conv_simple_with_single_on_top_0')
if not simple_conv.is_idle:
    result = simple_conv.resume_if_needed()
    print('Optimization result')
    print(result)

# best_run = simple_conv.optimize(epoch=2, max_evals=2)
# print('Optimization result')
# print(best_run)

# data_directory_path = '../daRta'
# sample_name = '10_min_sample'
# sample_id = 'cebd5588-926b-486b-8add-bbe1e74a1226'
#
# sample_dir = f'{data_directory_path}/input/{sample_name}_{sample_id}'
#
# frame_data_path = f'{sample_dir}/frame_data.json'
# if not os.path.isfile(frame_data_path):
#     print("frame_data.json hasn't been found in sample directory")
#     exit(1)
#
# with open(frame_data_path, encoding='utf-8') as frame_data_file:
#     frame_data = json.load(frame_data_file, object_hook=lambda d: Namespace(**d))
#
# if frame_data is None:
#     print('frame data reading failure')
#     exit(1)
#
# regular_frame_samples = [frame_sample for frame_sample in frame_data if frame_sample.type == 'regular']
#
#
# def frame_sample_load_image(frame_sample, target_size):
#     frame_image_path = f'{sample_dir}/{frame_sample.imagePath}'
#     return img_to_array(load_img(frame_image_path, target_size=target_size)) / 255.0
#
#
# @threadsafe_generator
# def frame_input_generator(frame_samples, target_size, should_shuffle: bool = False):
#
#     # output dimentions for x
#     # (batchsize, width, height, channels)
#     # (16, 640, 400, 3)
#     batchsize = 16
#
#     epoch = 0
#     while 1:
#         epoch += 1
#         target_samples = frame_samples[:]
#         if should_shuffle:
#             shuffle(target_samples)
#
#         read_count = 0
#         while read_count < (len(target_samples) - batchsize):
#             batch_frame_samples = target_samples[read_count: read_count + batchsize]
#             xs = numpy.array([frame_sample_load_image(frame_sample, target_size) for frame_sample in batch_frame_samples])
#             pos_xs = numpy.array([frame_sample.camera.position.x for frame_sample in batch_frame_samples])
#             pos_ys = numpy.array([frame_sample.camera.position.y for frame_sample in batch_frame_samples])
#             pos_zs = numpy.array([frame_sample.camera.position.z for frame_sample in batch_frame_samples])
#             rot_xs = numpy.array([frame_sample.camera.rotation.x for frame_sample in batch_frame_samples])
#             rot_ys = numpy.array([frame_sample.camera.rotation.y for frame_sample in batch_frame_samples])
#             rot_zs = numpy.array([frame_sample.camera.rotation.z for frame_sample in batch_frame_samples])
#             fovs = numpy.array([frame_sample.camera.fov for frame_sample in batch_frame_samples])
#             read_count += batchsize
#             yield (xs, [pos_xs, pos_ys, pos_zs, rot_xs, rot_ys, rot_zs, fovs])
#
#
# split_index = math.floor(len(regular_frame_samples) * 0.8)
# train_generator = frame_input_generator(regular_frame_samples[:split_index], (640, 400), should_shuffle=True)
# dev_generator = frame_input_generator(regular_frame_samples[split_index:], (640, 400))
# print(f'train samples: {len(regular_frame_samples[:split_index])}')
# print(f'dev samples: {len(regular_frame_samples[split_index:])}')
#
# x_debug, y_debug = next(train_generator)
# print(x_debug.shape)
# print(x_debug.dtype)
# exit(0)
#
# model_name = 'Conv_simple_with_single_on_top_0'
# model_dir = f'{data_directory_path}/output/{model_name}'
#
#
# img_input = Input(shape=(640, 400, 3), dtype='float32')
# x = layers.Conv2D(32, (3, 3), activation='relu')(img_input)
# x = layers.MaxPooling2D((2, 2))(x)
# x = layers.Conv2D(64, (3, 3), activation='relu')(x)
# x = layers.MaxPooling2D((2, 2))(x)
# x = layers.Conv2D(128, (3, 3), activation='relu')(x)
# x = layers.MaxPooling2D((2, 2))(x)
# x = layers.Conv2D(128, (3, 3), activation='relu')(x)
# x = layers.MaxPooling2D((2, 2))(x)
# x = layers.Conv2D(128, (3, 3), activation='relu')(x)
# x = layers.MaxPooling2D((2, 2))(x)
# x = layers.Conv2D(128, (3, 3), activation='relu')(x)
# x = layers.MaxPooling2D((2, 2))(x)
# x = layers.Flatten()(x)
# x = layers.Dense(512, activation='relu')(x)
# pos_x_pred = layers.Dense(1, name='pos_x')(x)
# pos_y_pred = layers.Dense(1, name='pos_y')(x)
# pos_z_pred = layers.Dense(1, name='pos_z')(x)
# rot_x_pred = layers.Dense(1, name='rot_x')(x)
# rot_y_pred = layers.Dense(1, name='rot_y')(x)
# rot_z_pred = layers.Dense(1, name='rot_z')(x)
# fov_pred = layers.Dense(1, name='fov')(x)
#
# model = Model(img_input, [pos_x_pred, pos_y_pred, pos_z_pred, rot_x_pred, rot_y_pred, rot_z_pred, fov_pred])
# model.compile(optimizer='rmsprop',
#               loss={'pos_x': 'mse',
#                     'pos_y': 'mse',
#                     'pos_z': 'mse',
#                     'rot_x': 'mse',
#                     'rot_y': 'mse',
#                     'rot_z': 'mse',
#                     'fov': 'mse'},
#               metrics=['mae'])
# model.summary()
#
# history = model.fit_generator(
#     train_generator,
#     steps_per_epoch=53,
#     epochs=100,
#     validation_data=dev_generator,
#     validation_steps=13
# )
#
#
# if not os.path.exists(model_dir):
#     os.makedirs(model_dir)
#
# current_time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
# model.save(f'{model_dir}/{model_name}_{current_time}.h5')
# with open(f'{model_dir}/history_{current_time}.json', "w") as json_file:
#     print(json.dumps(history.history), file=json_file)
#
#
# '''
# # reading history back from json for charts output
# with open(f'{model_dir}/history_2018-03-26.json', encoding='utf-8') as frame_data_file:
#     def history(): pass
#     history.history = json.load(frame_data_file)
# '''
#
# def save_chart(metrics:str, title:str):
#     mae = history.history[f'{metrics}_mean_absolute_error']
#     mae = mae[1 if len(mae) > 1 else 0:]
#     val_mae = history.history[f'val_{metrics}_mean_absolute_error']
#     val_mae = val_mae[1 if len(val_mae) > 1 else 0:]
#     loss = history.history[f'{metrics}_loss']
#     loss = loss[1 if len(loss) > 1 else 0:]
#     val_loss = history.history[f'val_{metrics}_loss']
#     val_loss = val_loss[1 if len(val_loss) > 1 else 0:]
#     epochs = range(1, len(mae) + 1)
#
#     plt.plot(epochs, mae, 'bo', label='Training MAE')
#     plt.plot(epochs, val_mae, 'b', label='Validation MAE')
#     plt.title(f'Training and validation MAE: {title}')
#     plt.legend()
#     plt.savefig(f'{model_dir}/charts_output/{metrics}_mean_absolute_error.png')
#     plt.cla()
#     plt.clf()
#
#     plt.plot(epochs, loss, 'bo', label='Training loss')
#     plt.plot(epochs, val_loss, 'b', label='Validation loss')
#     plt.title(f'Training and validation loss: {title}')
#     plt.legend()
#     plt.savefig(f'{model_dir}/charts_output/{metrics}_loss.png')
#     plt.cla()
#     plt.clf()
#
#
# if not os.path.exists(f'{model_dir}/charts_output'):
#     os.makedirs(f'{model_dir}/charts_output')
#
# save_chart('pos_x', 'Position: X')
# save_chart('pos_y', 'Position: Y')
# save_chart('pos_z', 'Position: Z')
# save_chart('rot_x', 'Rotation: X')
# save_chart('rot_y', 'Rotation: Y')
# save_chart('rot_z', 'Rotation: Z')
# save_chart('fov', 'FOV')
