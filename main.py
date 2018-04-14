# from models.conv_simple_with_single_on_top import ConvSimpleWithSingleOnTopModel
# from models.conv2dmaxpool6x_dense_heads7 import Conv2dMaxpool6xDenseHeads7
from models.conv2dmaxpool5x_dense_heads1 import Conv2dMaxpool5xDenseHeads1

initial_7heads = Conv2dMaxpool5xDenseHeads1(name='Conv_2d_Maxpool5x_Dense_Heads1', model_dir_path='./data/output')
initial_7heads.create_model().summary()

validation_score = initial_7heads.train_validate(epoch=100)
print('Score')
print(validation_score)
