from keras import models
from keras.layers import DepthwiseConv2D
from keras.utils import custom_object_scope

def custom_depthwise_conv2d(*args, **kwargs):
    kwargs.pop('groups', None)
    return DepthwiseConv2D(*args, **kwargs)

with custom_object_scope({'DepthwiseConv2D': custom_depthwise_conv2d}):
    model = models.load_model('keras_model-4.h5')
model.summary()