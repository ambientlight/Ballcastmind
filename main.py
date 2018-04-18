from models.seperableconv2d4x_dense_head1 import SeperableConv2d4xDenseHead1
# from models.inception_test import InceptionTest
# from keras.utils import plot_model

model = SeperableConv2d4xDenseHead1(name='SeperableConv2d4x_Dense_Head1', model_dir_path='./data/output')
model.create_model().summary()
model.train_prod(epoch=100)

# inception = InceptionTest(name='Inception_Test', model_dir_path='./data/output')
# inception.create_model().summary()
# inception.train_prod(epoch=100)