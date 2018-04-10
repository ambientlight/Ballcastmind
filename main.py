# from models.conv_simple_with_single_on_top import ConvSimpleWithSingleOnTopModel
from models.conv2dmaxpool6x_dense_heads7 import Conv2dMaxpool6xDenseHeads7

initial_7heads = Conv2dMaxpool6xDenseHeads7(name='Conv_2d_Maxpool6x_Dense_Heads7', model_dir_path='./data/output')
initial_7heads.create_model().summary()

validation_score = initial_7heads.train_validate(epoch=20)
print('Score')
print(validation_score)
