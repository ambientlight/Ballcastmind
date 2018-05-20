import numpy
import json
from keras.models import Model, load_model
from models.seperableconv2d4x_dense_head1_fit_test import SeperableConv2d4xDenseHead1
# from models.inception_test import InceptionTest
# from keras.utils import plot_model

model_descriptor = SeperableConv2d4xDenseHead1(name='SeperableConv2d4x_Dense_Head1', model_dir_path='./data/output')
data_locators = model_descriptor.data_locators()
lb_progress = 194.3
target_size = (1384, 865)
frame_samples_to_predict = [frame_sample for frame_sample in data_locators if frame_sample.progress >= lb_progress and frame_sample.imagePath]

images = numpy.array([model_descriptor.frame_sample_load_image(frame_sample, target_size) for frame_sample in frame_samples_to_predict])

rot_x_model_path = './data/model_extracts/rot_x_model_ep92_s1.h5'
rot_y_model_path = './data/model_extracts/rot_y_model_ep93_s1.h5'
rot_z_model_path = './data/model_extracts/rot_z_model_ep93_s1.h5'
rot_x_model: Model = load_model(rot_x_model_path)
rot_y_model: Model = load_model(rot_y_model_path)
rot_z_model: Model = load_model(rot_z_model_path)
rot_xs = rot_x_model.predict(x=images, verbose=1)
rot_ys = rot_y_model.predict(x=images, verbose=1)
rot_zs = rot_z_model.predict(x=images, verbose=1)

for i in range(len(frame_samples_to_predict)):
    frame_sample_prog = frame_samples_to_predict[i].progress
    rot_x_expected = frame_samples_to_predict[i].camera.rotation.x
    rot_y_expected = frame_samples_to_predict[i].camera.rotation.y
    rot_z_expected = frame_samples_to_predict[i].camera.rotation.z
    rot_x = rot_xs[i][0]
    rot_y = rot_ys[i][0]
    rot_z = rot_zs[i][0]
    print(f'${frame_sample_prog}: X(p: ${rot_x}, e: ${rot_x_expected}, diff:${rot_x - rot_x_expected}), '
          f'Y(p: ${rot_y}, e: ${rot_y_expected}, diff:${rot_y - rot_y_expected}), '
          f'Z(p: ${rot_z}, e: ${rot_z_expected}, diff:${rot_z - rot_z_expected})')

    frame_samples_to_predict[i].camera.rotation.x = numpy.asscalar(rot_x)
    # frame_samples_to_predict[i].camera.rotation.y = numpy.asscalar(rot_y)
    frame_samples_to_predict[i].camera.rotation.z = numpy.asscalar(rot_z)

json_objects = [{
    'type': frame_sample.type,
    'completeness': frame_sample.completeness,
    'wasCompleted': frame_sample.wasCompleted,
    'camera': {
        'position': {
            'x': frame_sample.camera.position.x,
            'y': frame_sample.camera.position.y,
            'z': frame_sample.camera.position.z
        },
        'rotation': {
            'x': frame_sample.camera.rotation.x,
            'y': frame_sample.camera.rotation.y,
            'z': frame_sample.camera.rotation.z
        },
        'fov': frame_sample.camera.fov,
    },
    'ballPosition': {
        'x': frame_sample.ballPosition.x,
        'y': frame_sample.ballPosition.y,
        'z': frame_sample.ballPosition.z
    },
    'imagePath': frame_sample.imagePath,
    'progress': frame_sample.progress,
    'canvasSize': {
        'width': frame_sample.canvasSize.width,
        'height': frame_sample.canvasSize.height
    },
    'relationIds': frame_sample.relationIds,
    'alternativePositions': {},
    'imported': True
} for frame_sample in frame_samples_to_predict]

with open('./fit_test.json', 'w') as json_file:
    print(json.dumps(json_objects), file=json_file)

