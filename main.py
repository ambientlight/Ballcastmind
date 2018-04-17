# from models.conv_simple_with_single_on_top import ConvSimpleWithSingleOnTopModel
# from models.conv2dmaxpool6x_dense_heads7 import Conv2dMaxpool6xDenseHeads7
from models.conv2dmaxpool5x_dense_heads1 import Conv2dMaxpool5xDenseHeads1
from models.inception_test import InceptionTest

# from keras.utils import plot_model

model = Conv2dMaxpool5xDenseHeads1(name='Conv_2d_Maxpool5x_Dense_Heads1', model_dir_path='./data/output')
model.create_model().summary()
model.train_prod(epoch=100)

# inception = InceptionTest(name='Inception_Test', model_dir_path='./data/output')
# inception.create_model().summary()
# inception.train_prod(epoch=100)