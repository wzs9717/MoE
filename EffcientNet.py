import os
import math
from typing import List

from keras import backend as K
from keras import layers
from keras.models import Model
from keras.utils import get_file, get_source_inputs

from keras_applications.imagenet_utils import _obtain_input_shape
from keras_applications.imagenet_utils import preprocess_input as _preprocess

from keras_efficientnets.config import BlockArgs, get_default_block_list
from keras_efficientnets.custom_objects import EfficientNetConvInitializer
from keras_efficientnets.custom_objects import EfficientNetDenseInitializer
from keras_efficientnets.custom_objects import Swish, 

def EfficientNet(input_shape,
                 block_args_list: List[BlockArgs],
                 width_coefficient: float,
                 depth_coefficient: float,
                 include_top=True,
                 weights=None,
                 input_tensor=None,
                 pooling=None,
                 classes=1000,
                 dropout_rate=0.,
                 drop_connect_rate=0.,
                 batch_norm_momentum=0.99,
                 batch_norm_epsilon=1e-3,
                 depth_divisor=8,
                 min_depth=None,
                 data_format=None,
                 default_size=None,
                 **kwargs):


    if data_format is None:
        data_format = K.image_data_format()

    # if data_format == 'channels_first':
    #     channel_axis = 1
    # else:
    #     channel_axis = -1

    if default_size is None:
        default_size = 224

    if block_args_list is None:
        block_args_list = get_default_block_list()

    # count number of strides to compute min size
    stride_count = 1
    for block_args in block_args_list:
        if block_args.strides is not None and block_args.strides[0] > 1:
            stride_count += 1

    min_size = int(2 ** stride_count)

    # Determine proper input shape and default size.
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=default_size,
                                      min_size=min_size,
                                      data_format=data_format,
                                      require_flatten=include_top,
                                      weights=weights)

    # Stem part
    if input_tensor is None:
        inputs = layers.Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            inputs = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            inputs = input_tensor

    x = inputs
    x = layers.Conv2D(
        filters=round_filters(32, width_coefficient,
                              depth_divisor, min_depth),
        kernel_size=[3, 3],
        strides=[2, 2],
        kernel_initializer=EfficientNetConvInitializer(),
        padding='same',
        use_bias=False)(x)
    x = layers.BatchNormalization(
        axis=channel_axis,
        momentum=batch_norm_momentum,
        epsilon=batch_norm_epsilon)(x)
    x = Swish()(x)

    num_blocks = sum([block_args.num_repeat for block_args in block_args_list])
    drop_connect_rate_per_block = drop_connect_rate / float(num_blocks)

    # Blocks part
    for block_idx, block_args in enumerate(block_args_list):
        assert block_args.num_repeat > 0

        # Update block input and output filters based on depth multiplier.
        block_args.input_filters = round_filters(block_args.input_filters, width_coefficient, depth_divisor, min_depth)
        block_args.output_filters = round_filters(block_args.output_filters, width_coefficient, depth_divisor, min_depth)
        block_args.num_repeat = round_repeats(block_args.num_repeat, depth_coefficient)

        # The first block needs to take care of stride and filter size increase.
        x = MBConvBlock(block_args.input_filters, block_args.output_filters,
                        block_args.kernel_size, block_args.strides,
                        block_args.expand_ratio, block_args.se_ratio,
                        block_args.identity_skip, drop_connect_rate_per_block * block_idx,
                        batch_norm_momentum, batch_norm_epsilon, data_format)(x)

        if block_args.num_repeat > 1:
            block_args.input_filters = block_args.output_filters
            block_args.strides = [1, 1]

        for _ in range(block_args.num_repeat - 1):
            x = MBConvBlock(block_args.input_filters, block_args.output_filters,
                            block_args.kernel_size, block_args.strides,
                            block_args.expand_ratio, block_args.se_ratio,
                            block_args.identity_skip, drop_connect_rate_per_block * block_idx,
                            batch_norm_momentum, batch_norm_epsilon, data_format)(x)

    # Head part
    x = layers.Conv2D(
        filters=round_filters(1280, width_coefficient, depth_coefficient, min_depth),
        kernel_size=[1, 1],
        strides=[1, 1],
        kernel_initializer=EfficientNetConvInitializer(),
        padding='same',
        use_bias=False)(x)
    x = layers.BatchNormalization(
        axis=channel_axis,
        momentum=batch_norm_momentum,
        epsilon=batch_norm_epsilon)(x)
    x = Swish()(x)

    x = layers.GlobalAveragePooling2D(data_format=data_format)(x)

    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(classes, kernel_initializer=EfficientNetDenseInitializer())(x)
    x = layers.Activation('softmax')(x)

    outputs = x

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)

    model = Model(inputs, outputs)

    # Load weights
    
    elif weights is not None:
        model.load_weights(weights)

    return model