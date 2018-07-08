# from models.line_features_tests import LineFeaturesTests
from models.seperableconv2d4x_dense_head1 import SeperableConv2d4xDenseHead1
# from keras.utils import plot_model

# model = LineFeaturesTests(name='LineFeaturesTests', model_dir_path='./data/output')
model = SeperableConv2d4xDenseHead1(name='SeperableConv2d4xDenseHead1', model_dir_path='./data/output')

shouldOptimize = False
if shouldOptimize:
    res = model.optimize(max_evals=1024, epoch=100)
    print(res)
else:
    model.create_model().summary()
    model.train_validate(epoch=100, from_scratch=True)
