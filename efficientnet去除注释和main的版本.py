import tensorflow as tf
import tensorflow.keras.backend as K
import math

def swish(x):
    return x * tf.nn.sigmoid(x)

def SEBlock(input_filters,se_output_filters,se_ratio=0.25):
    def block(inputs):
        num_reduced_filters = max(1, int(input_filters * se_ratio))		
        x = inputs
        x = tf.keras.layers.Lambda(lambda a: K.mean(a, axis=[1, 2], keepdims=True))(x)
        x = tf.keras.layers.Conv2D(num_reduced_filters,kernel_size=(1, 1),padding='same')(x)
        x = swish(x)
        x = tf.keras.layers.Conv2D(se_output_filters,kernel_size=(1, 1),padding='same',)(x)
        x = tf.keras.layers.Activation('sigmoid')(x)
        out = tf.keras.layers.Multiply()([x, inputs])
        return out
    return block

class DropConnect(tf.keras.layers.Layer):
    def __init__(self, drop_connect_rate=0.):
        super().__init__()
        self.drop_connect_rate = drop_connect_rate

    def call(self, inputs, training=None):
        def drop_connect():
            survival_prob = 1.0 - self.drop_connect_rate
            batch_size = tf.shape(inputs)[0]
            random_tensor = survival_prob
            random_tensor += tf.random_uniform([batch_size, 1, 1, 1], dtype=inputs.dtype)
            binary_tensor = tf.floor(random_tensor)
            output = tf.div(inputs, survival_prob) * binary_tensor
            return output
        return K.in_train_phase(drop_connect, inputs, training=training)

def MBConvBlock(input_filters, output_filters,kernel_size, strides,expand_ratio,drop_connect_rate):
    def block(inputs):
        se_output_filters = input_filters * expand_ratio
        if expand_ratio != 1:
            x = tf.keras.layers.Conv2D(se_output_filters,kernel_size=(1, 1),padding='same',use_bias=False)(inputs)
            x = tf.keras.layers.BatchNormalization()(x)
            x = swish(x)
        else:
            x = inputs
        x = tf.keras.layers.DepthwiseConv2D((kernel_size, kernel_size),strides=strides,padding='same',use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = swish(x)
        x = SEBlock(input_filters,se_output_filters)(x)
        x = tf.keras.layers.Conv2D(output_filters,kernel_size=(1, 1),padding='same',use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        if all(s == 1 for s in strides) and (input_filters == output_filters):
            if drop_connect_rate:
                x = DropConnect(drop_connect_rate)(x)

            x = tf.keras.layers.Add()([x, inputs])
        return x
    return block

import collections
BlockArgs = collections.namedtuple('BlockArgs', ['kernel_size', 'num_repeat', 'input_filters', 'output_filters','expand_ratio', 'strides'])
block_args_list = [
    BlockArgs(kernel_size=3, num_repeat=1, input_filters=32, output_filters=16,expand_ratio=1, strides=[1, 1]),
    BlockArgs(kernel_size=3, num_repeat=2, input_filters=16, output_filters=24,expand_ratio=6, strides=[2, 2]),
    BlockArgs(kernel_size=5, num_repeat=2, input_filters=24, output_filters=40,expand_ratio=6, strides=[2, 2]),
    BlockArgs(kernel_size=3, num_repeat=3, input_filters=40, output_filters=80,expand_ratio=6, strides=[2, 2]),
    BlockArgs(kernel_size=5, num_repeat=3, input_filters=80, output_filters=112,expand_ratio=6, strides=[1, 1]),
    BlockArgs(kernel_size=5, num_repeat=4, input_filters=112, output_filters=192,expand_ratio=6, strides=[2, 2]),
    BlockArgs(kernel_size=3, num_repeat=1, input_filters=192, output_filters=320,expand_ratio=6, strides=[1, 1])
]
stride_count = 5
num_blocks = 16

def EfficientNet(input_shape,classes,width_coefficient: float,depth_coefficient: float,include_top=True,dropout_rate=0.,drop_connect_rate=0.): 
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = inputs
    x = tf.keras.layers.Conv2D(filters=int(32*width_coefficient), kernel_size=(3,3),strides=(2,2),padding='same',use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = swish(x)    
    drop_connect_rate_per_block = drop_connect_rate / float(num_blocks)
    for block_idx, block_args in enumerate(block_args_list):
        my_input_filters = int(block_args.input_filters*width_coefficient)
        my_output_filters = int(block_args.output_filters*width_coefficient)
        my_num_repeat = math.ceil(block_args.num_repeat*depth_coefficient)
        x = MBConvBlock(my_input_filters, my_output_filters,block_args.kernel_size, block_args.strides,block_args.expand_ratio,drop_connect_rate_per_block * block_idx)(x)
        if my_num_repeat > 1:
            my_input_filters = my_output_filters
            my_strides = [1, 1]
        for _ in range(my_num_repeat - 1):
            x = MBConvBlock(my_input_filters, my_output_filters,block_args.kernel_size, my_strides,block_args.expand_ratio,drop_connect_rate_per_block * block_idx)(x)
    x = tf.keras.layers.Conv2D(filters=int(1280*width_coefficient),kernel_size=(1, 1),padding='same',use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = swish(x)
    if include_top:
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        if dropout_rate > 0:
            x = tf.keras.layers.Dropout(dropout_rate)(x)
        x = tf.keras.layers.Dense(classes)(x)
        x = tf.keras.layers.Activation('softmax')(x)
    outputs = x
    model = tf.keras.models.Model(inputs, outputs)
    return model

def EfficientNetB0(input_shape,classes,include_top=True,dropout_rate=0.2,drop_connect_rate=0.): 
    return  EfficientNet(input_shape,classes,1.0,1.0,include_top=include_top,dropout_rate=dropout_rate,drop_connect_rate=drop_connect_rate)

def EfficientNetB1(input_shape,classes,include_top=True,dropout_rate=0.2,drop_connect_rate=0.): 
    return  EfficientNet(input_shape,classes,1.0,1.1,include_top=include_top,dropout_rate=dropout_rate,drop_connect_rate=drop_connect_rate)

def EfficientNetB2(input_shape,classes,include_top=True,dropout_rate=0.3,drop_connect_rate=0.): 
    return  EfficientNet(input_shape,classes,1.1,1.2,include_top=include_top,dropout_rate=dropout_rate,drop_connect_rate=drop_connect_rate)

def EfficientNetB3(input_shape,classes,include_top=True,dropout_rate=0.3,drop_connect_rate=0.): 
    return  EfficientNet(input_shape,classes,1.2,1.4,include_top=include_top,dropout_rate=dropout_rate,drop_connect_rate=drop_connect_rate)

def EfficientNetB4(input_shape,classes,include_top=True,dropout_rate=0.4,drop_connect_rate=0.): 
    return  EfficientNet(input_shape,classes,1.4,1.8,include_top=include_top,dropout_rate=dropout_rate,drop_connect_rate=drop_connect_rate)

def EfficientNetB5(input_shape,classes,include_top=True,dropout_rate=0.4,drop_connect_rate=0.): 
    return  EfficientNet(input_shape,classes,1.6,2.2,include_top=include_top,dropout_rate=dropout_rate,drop_connect_rate=drop_connect_rate)

def EfficientNetB6(input_shape,classes,include_top=True,dropout_rate=0.5,drop_connect_rate=0.): 
    return  EfficientNet(input_shape,classes,1.8,2.6,include_top=include_top,dropout_rate=dropout_rate,drop_connect_rate=drop_connect_rate)

def EfficientNetB7(input_shape,classes,include_top=True,dropout_rate=0.5,drop_connect_rate=0.): 
    return  EfficientNet(input_shape,classes,2.0,3.1,include_top=include_top,dropout_rate=dropout_rate,drop_connect_rate=drop_connect_rate)

def EfficientNetB8(input_shape,classes,include_top=True,dropout_rate=0.5,drop_connect_rate=0.): 
    return  EfficientNet(input_shape,classes,2.2,3.6,include_top=include_top,dropout_rate=dropout_rate,drop_connect_rate=drop_connect_rate)