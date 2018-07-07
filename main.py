from models.line_features_tests import LineFeaturesTests
# from keras.utils import plot_model

model = LineFeaturesTests(name='LineFeaturesTests', model_dir_path='./data/output')

shouldOptimize = False
if shouldOptimize:
    model.train_validate(epoch=200, from_scratch=True)
else:
    res = model.optimize(max_evals=1024, epoch=100)
    print(res)
