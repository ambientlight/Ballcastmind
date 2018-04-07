from models.conv_simple_with_single_on_top import ConvSimpleWithSingleOnTopModel

simple_conv = ConvSimpleWithSingleOnTopModel(name='Conv_simple_with_single_on_top_0')
validation_score = simple_conv.resume_if_needed()
print('Score')
print(validation_score)
