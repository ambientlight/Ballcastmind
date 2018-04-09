from models.conv_simple_with_single_on_top import ConvSimpleWithSingleOnTopModel

simple_conv = ConvSimpleWithSingleOnTopModel(name='Conv_simple_with_single_on_top_0', model_dir_path='./data/output')
simple_conv.create_model().summary()

validation_score = simple_conv.train_validate()
print('Score')
print(validation_score)
