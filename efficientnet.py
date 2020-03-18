#适用于tensorflow2.0;tensorflowtf1.14.0rc也可以用

import tensorflow as tf
import tensorflow.keras.backend as K
import math

'''
efficientnet:
我们也使用了swish激活函数(Ramachandran et al., 2018; Elfwing et al., 2018)，
固定的AutoAugment数据增强策略(Cubuk et al., 2019)，
和随机的深度(Huang et al., 2016)和drop connect率为0.2。
众所周知更大的模型需要更多的正则化，
所以我们线性的增加dropout(Srivastava et al., 2014)的比率从EfficientNet-B0的0.2到EfficientNet-B7的0.5

Swish是Google2017年在10月16号提出的一种新型激活函数,
其原始公式为:f(x)=x * sigmod(x),
变形Swish-B激活函数的公式则为f(x)=x * sigmod(b * x),
拥有不饱和,光滑,非单调性的特征
多项测试表明Swish以及Swish-B激活函数的性能即佳,在不同的数据集上都表现出了要优于当时最佳激活函数的性能。
'''	
#tf.keras.layers.Layer：This is the class from which all layers inherit.
#所以我们后面要用一些layers时，需要用这个来定义
#下面那种对tf1.14.0rc支持的比较好，但是在tf2.0会断掉梯度的反向传播，tf2.0可以用第二种。
'''
class Swish(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
    def call(self, inputs):
        return tf.nn.swish(inputs)
'''

#用这个,直接用tf.nn.swish会报错的！
def swish(x):
    return x * tf.nn.sigmoid(x)



'''
Conv2D的默认构造
__init__(
    filters,
    kernel_size,
    strides=(1, 1),
    padding='valid',
    data_format=None,
    dilation_rate=(1, 1),
    activation=None,
    use_bias=True,
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros',
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    **kwargs
)
'''
#se_ratio固定为0.25
def SEBlock(input_filters,se_output_filters,se_ratio=0.25):
		
	#这个就是(xxxxx)(inputs)的那种函数,所以应该写这种形式
    def block(inputs):
        num_reduced_filters = max(1, int(input_filters * se_ratio))	

		#inputs后面还要用来相乘的		
        x = inputs

        #这里假如用gap的话，还需要进行两次tf.expand_dims。恢复成原来的4维
        # x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Lambda(lambda a: K.mean(a, axis=[1, 2], keepdims=True))(x)

		#默认Glorot正态分布初始化方法，也称作Xavier正态分布初始化，参数由0均值，
		#标准差为sqrt(2 / (fan_in + fan_out))的正态分布产生，
		#其中fan_in和fan_out是权重张量的扇入扇出（即输入和输出单元数目）
        x = tf.keras.layers.Conv2D(num_reduced_filters,kernel_size=(1, 1),padding='same')(x)
        x = swish(x)

        # Excite
        x = tf.keras.layers.Conv2D(se_output_filters,kernel_size=(1, 1),padding='same',)(x)
        x = tf.keras.layers.Activation('sigmoid')(x)
        out = tf.keras.layers.Multiply()([x, inputs])
        return out

    return block



#from	https://github.com/tensorflow/tpu/blob/d6f2ef3edfeb4b1c2039b81014dc5271a7753832/models/official/efficientnet/utils.py#L146
#tf.keras.backend.in_train_phase		Selects x in train phase, and alt otherwise.
class DropConnect(tf.keras.layers.Layer):

    def __init__(self, drop_connect_rate=0.):
        super().__init__()
        self.drop_connect_rate = drop_connect_rate

	#training flag在这，和最后的return搭配使用
    def call(self, inputs, training=None):

        def drop_connect():
            survival_prob = 1.0 - self.drop_connect_rate

            # Compute drop_connect tensor
            batch_size = tf.shape(inputs)[0]
            random_tensor = survival_prob
            random_tensor += tf.random_uniform([batch_size, 1, 1, 1], dtype=inputs.dtype)
            binary_tensor = tf.floor(random_tensor)
            output = tf.div(inputs, survival_prob) * binary_tensor
            return output

        return K.in_train_phase(drop_connect, inputs, training=training)



'''
BatchNormalization的默认构造
__init__(
    axis=-1,#默认就是最后一层了，即按通道了
    momentum=0.99,#滑动平均系数
    epsilon=0.001,#防零最小值
    center=True,
    scale=True,
    beta_initializer='zeros',
    gamma_initializer='ones',
    moving_mean_initializer='zeros',
    moving_variance_initializer='ones',
    beta_regularizer=None,
    gamma_regularizer=None,
    beta_constraint=None,
    gamma_constraint=None,
    renorm=False,
    renorm_clipping=None,
    renorm_momentum=0.99,
    fused=None,
    trainable=True,
    virtual_batch_size=None,
    adjustment=None,
    name=None,
    **kwargs
)
'''
#倒置瓶颈+se模块+残差模块
def MBConvBlock(input_filters, output_filters,kernel_size, strides,expand_ratio,drop_connect_rate):

    def block(inputs):

		#expand_ratio倒置瓶颈的比率
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

        #if has_se:
        x = SEBlock(input_filters,se_output_filters)(x)

        # output phase
        x = tf.keras.layers.Conv2D(output_filters,kernel_size=(1, 1),padding='same',use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)

        #if id_skip: #如果下采样了，就不好用res_block了
        if all(s == 1 for s in strides) and (input_filters == output_filters):

			# only apply drop_connect if skip presents. #默认情况会直接跳过这一层
            if drop_connect_rate:
                x = DropConnect(drop_connect_rate)(x)

            x = tf.keras.layers.Add()([x, inputs])

        return x

    return block



'''
我们知道tuple可以表示不变集合，例如，一个点的二维坐标就可以表示成：
p = (1, 2)
但是，看到(1, 2)，很难看出这个tuple是用来表示一个坐标的。这时，namedtuple就派上了用场。

用法：
namedtuple('名称', [属性list])
使用namedtuple表示一个坐标的例子如下：

from collections import namedtuple
Point = namedtuple('Point', ['x', 'y'])
p = Point(1, 2)
print(p.x，p.y）
输出为：1,2
'''
#def EfficientNet之前的一些配置，这是B0的丹方！！！
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
#min_size = int(2 ** stride_count) #32 - -比32还小的图真是没谁了，话说在MBconv前还降采样了一次，要用的话int(2 ** (stride_count+1))=64好点吧
num_blocks = 16


#这里我和官方的实现有点不一样
#官方的from https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py
#论文中wider这个才是channels的系数，应该是这里的 width_coefficient
#感觉不缩小模型的话这个操作没有意义，直接用int(filters*width_coefficient)代替即可，且符合论文的原理
'''
def round_filters(filters, width_coefficient, depth_divisor):
    """Round number of filters based on depth multiplier."""
    multiplier = float(width_coefficient)   #这玩意本来就是float，这样写读者看得明白些
    divisor = int(depth_divisor)
    filters *= multiplier   #乘以完了系数
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    return int(new_filters)
'''



#include_top=False时里面不加全局平均池化
'''
官方程序段 https://github.com/tensorflow/tpu/blob/8bccd04606e39c2067752f061653bbce5477c37e/models/official/efficientnet/efficientnet_model.py
    for idx, block in enumerate(self._blocks):  #先逐行
      is_reduction = False  # reduction flag for blocks after the stem layer

      if (block.block_args().super_pixel == 1 and idx == 0):
        reduction_idx += 1
        self.endpoints['reduction_%s' % reduction_idx] = outputs

      elif ((idx == len(self._blocks) - 1) or
            self._blocks[idx + 1].block_args().strides[0] > 1):
        is_reduction = True
        reduction_idx += 1

      with tf.variable_scope('blocks_%s' % idx):
        survival_prob = self._global_params.survival_prob
        if survival_prob:
          drop_rate = 1.0 - survival_prob
          survival_prob = 1.0 - drop_rate * float(idx) / len(self._blocks) #也是用了个乘idx
'''
#drop_connect_rate可以在构建模型时手动加上
def EfficientNet(input_shape,classes,width_coefficient: float,depth_coefficient: float,include_top=True,dropout_rate=0.,drop_connect_rate=0.): 

    inputs = tf.keras.layers.Input(shape=input_shape)
    x = inputs

    #丹方中的第一个Conv(3,3),也下采样了
    x = tf.keras.layers.Conv2D(filters=int(32*width_coefficient), kernel_size=(3,3),strides=(2,2),padding='same',use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = swish(x)
    
    drop_connect_rate_per_block = drop_connect_rate / float(num_blocks)

    # Blocks part  #for打在最外头了，一行行剖开吧
    for block_idx, block_args in enumerate(block_args_list):

        # Update block input and output filters based on depth multiplier.  #不要去改元祖，来新建变量
        my_input_filters = int(block_args.input_filters*width_coefficient)
        my_output_filters = int(block_args.output_filters*width_coefficient)
        my_num_repeat = math.ceil(block_args.num_repeat*depth_coefficient)

        # The first block needs to take care of stride and filter size increase.  #越后面的组块drop_connect_rate越高
        x = MBConvBlock(my_input_filters, my_output_filters,block_args.kernel_size, block_args.strides,block_args.expand_ratio,drop_connect_rate_per_block * block_idx)(x)

        if my_num_repeat > 1:
            my_input_filters = my_output_filters
            my_strides = [1, 1]

        for _ in range(my_num_repeat - 1):
            x = MBConvBlock(my_input_filters, my_output_filters,block_args.kernel_size, my_strides,block_args.expand_ratio,drop_connect_rate_per_block * block_idx)(x)

    # Head part
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



'''
官方给的参数  https://github.com/tensorflow/tpu/blob/8bccd04606e39c2067752f061653bbce5477c37e/models/official/efficientnet/efficientnet_builder.py
def efficientnet_params(model_name):
  """Get efficientnet params based on model name."""
  params_dict = {
      # (width_coefficient, depth_coefficient, resolution, dropout_rate)
      'efficientnet-b0': (1.0, 1.0, 224, 0.2),
      'efficientnet-b1': (1.0, 1.1, 240, 0.2),
      'efficientnet-b2': (1.1, 1.2, 260, 0.3),
      'efficientnet-b3': (1.2, 1.4, 300, 0.3),
      'efficientnet-b4': (1.4, 1.8, 380, 0.4),
      'efficientnet-b5': (1.6, 2.2, 456, 0.4),
      'efficientnet-b6': (1.8, 2.6, 528, 0.5),
      'efficientnet-b7': (2.0, 3.1, 600, 0.5),
      'efficientnet-b8': (2.2, 3.6, 672, 0.5),
      'efficientnet-l2': (4.3, 5.3, 800, 0.5),
  }
'''
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



if __name__ == '__main__':

    cls=10
    my_size=(32,32,3)

    model = EfficientNetB0(my_size,cls,include_top=True)
    model.summary()

    (x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()
    y_train = tf.keras.utils.to_categorical(y_train, cls)

    model.compile(loss='categorical_crossentropy',optimizer=tf.keras.optimizers.Adam(lr=2e-4),metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.1)
