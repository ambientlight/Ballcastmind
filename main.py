from models.seperableconv2d4x_dense_head4 import SeperableConv2d4xDenseHead4
# from keras.utils import plot_model

model = SeperableConv2d4xDenseHead4(name='SeperableConv2d4x_Dense_Head4', model_dir_path='./data/output')
model.create_model().summary()
model.train_validate(epoch=100)
