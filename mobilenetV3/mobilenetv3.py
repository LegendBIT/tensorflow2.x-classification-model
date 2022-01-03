# 根据tf.keras的官方代码修改的mobilenetv3的网络模型
import tensorflow as tf
from tensorflow.keras import layers, models


"""
    Reference:
    - [Searching for MobileNetV3](https://arxiv.org/pdf/1905.02244.pdf) (ICCV 2019)
    The following table describes the performance of MobileNets v3:
    ------------------------------------------------------------------------
    MACs stands for Multiply Adds
    |Classification Checkpoint|MACs(M)|Parameters(M)|Top1 Accuracy|Pixel1 CPU(ms)|
    |---|---|---|---|---|
    | mobilenet_v3_large_1.0_224              | 217 | 5.4 |   75.6   |   51.2  |
    | mobilenet_v3_large_0.75_224             | 155 | 4.0 |   73.3   |   39.8  |
    | mobilenet_v3_large_minimalistic_1.0_224 | 209 | 3.9 |   72.3   |   44.1  |
    | mobilenet_v3_small_1.0_224              | 66  | 2.9 |   68.1   |   15.8  |
    | mobilenet_v3_small_0.75_224             | 44  | 2.4 |   65.4   |   12.8  |
    | mobilenet_v3_small_minimalistic_1.0_224 | 65  | 2.0 |   61.9   |   12.2  |
    For image classification use cases, see
    [this page for detailed examples](https://keras.io/api/applications/#usage-examples-for-image-classification-models).
    For transfer learning use cases, make sure to read the
    [guide to transfer learning & fine-tuning](https://keras.io/guides/transfer_learning/).
"""

##################################################################################################################################
# 定义V3的完整模型 #################################################################################################################
##################################################################################################################################
def MobileNetV3(input_shape=[224, 224 ,3], classes=1000, dropout_rate=0.2, alpha=1.0, weights=None,
                 model_type='large', minimalistic=False, classifier_activation='softmax', include_preprocessing=False):
    # 如果有权重文件，那就意味着要迁移学习，那就意味着需要让BN层始终处于infer状态，否则解冻整个网络后，会出现acc下降loss上升的现象，终其原因是解冻网络之
    # 前，网络BN层用的是之前数据集的均值和方差，解冻后虽然维护着新的滑动平均和滑动方差，但是单次训练时使用的是当前batch的均值和方差，差异太大造成特征崩塌
    if weights:
        bn_training = False
    else:
        bn_training = None
    bn_decay = 0.99  # BN层的滑动平均系数，这个值的设置需要匹配steps和batchsize否则会出现奇怪现象
    # 确定通道所处维度
    channel_axis = -1
    # 根据是否为mini设置，修改部分配置参数
    if minimalistic:
        kernel = 3
        activation = relu
        se_ratio = None
        name = "mini"
    else:
        kernel = 5
        activation = hard_swish
        se_ratio = 0.25
        name = "norm"
    # 定义模型输入张量
    img_input = layers.Input(shape=input_shape)
    # 是否包含预处理层
    if include_preprocessing:
        x = layers.Rescaling(scale=1. / 127.5, offset=-1.)(img_input)
    else:
        x = img_input
    # 定义整个模型的第一个特征提取层
    x = layers.Conv2D(16, kernel_size=3, strides=(2, 2), padding='same', use_bias=False, name='Conv')(x)
    x = layers.BatchNormalization(axis=channel_axis, epsilon=1e-3, momentum=bn_decay, name='Conv/BatchNorm')(x, training=bn_training)
    x = activation(x)
    # 定义整个模型的骨干特征提取
    if model_type == 'large':
        x = MobileNetV3Large(x, kernel, activation, se_ratio, alpha, bn_training, bn_decay)
        last_point_ch = 1280
    else:
        x = MobileNetV3Small(x, kernel, activation, se_ratio, alpha, bn_training, bn_decay)
        last_point_ch = 1024
    # 定义整个模型的后特征提取
    last_conv_ch = _depth(x.shape[channel_axis] * 6)
    # if the width multiplier is greater than 1 we increase the number of output channels
    if alpha > 1.0:
        last_point_ch = _depth(last_point_ch * alpha)
    x = layers.Conv2D(last_conv_ch, kernel_size=1, padding='same', use_bias=False, name='Conv_1')(x)
    x = layers.BatchNormalization(axis=channel_axis, epsilon=1e-3, momentum=bn_decay, name='Conv_1/BatchNorm')(x, training=bn_training)
    x = activation(x)
    # 如果tf版本大于等于2.6则直接使用下面第一句就可以了，否则使用下面2~3句
    # x = layers.GlobalAveragePooling2D(data_format='channels_last', keepdims=True)(x)
    x = layers.GlobalAveragePooling2D(data_format='channels_last')(x)
    x= tf.expand_dims(tf.expand_dims(x, 1), 1)
    # 定义第一个特征分类层
    x = layers.Conv2D(last_point_ch, kernel_size=1, padding='same', use_bias=True, name='Conv_2')(x)
    x = activation(x)
    # 定义第二个特征分类层
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate)(x)
    x = layers.Conv2D(classes, kernel_size=1, padding='same', name='Logits')(x)
    x = layers.Flatten()(x)
    x = layers.Activation(activation=classifier_activation, name='Predictions')(x)  # 注意损失函数需要与初始权重匹配，否则预训练没有意义
    # 创建模型
    model = models.Model(img_input, x, name='MobilenetV3' + '_' + model_type + '_' + name)
    # 恢复权重
    if weights:
        model.load_weights(weights, by_name=True)
        # print(model.get_layer(name="block_8_project_BN").get_weights()[0][:4])
    
    return model


##################################################################################################################################
# 定义V3的骨干网络，不包含前处理和后处理 ###############################################################################################
##################################################################################################################################

# 定义mobilenetv3-small的骨干部分，不包含第一层的卷积特征提取和后处理
def MobileNetV3Small(x, kernel, activation, se_ratio, alpha, bn_training, mome):

    def depth(d):
      return _depth(d * alpha)

    x = _inverted_res_block(x, 1, depth(16), 3, 2, se_ratio, relu, 0, bn_training, mome)
    x = _inverted_res_block(x, 72. / 16, depth(24), 3, 2, None, relu, 1, bn_training, mome)
    x = _inverted_res_block(x, 88. / 24, depth(24), 3, 1, None, relu, 2, bn_training, mome)
    x = _inverted_res_block(x, 4, depth(40), kernel, 2, se_ratio, activation, 3, bn_training, mome)
    x = _inverted_res_block(x, 6, depth(40), kernel, 1, se_ratio, activation, 4, bn_training, mome)
    x = _inverted_res_block(x, 6, depth(40), kernel, 1, se_ratio, activation, 5, bn_training, mome)
    x = _inverted_res_block(x, 3, depth(48), kernel, 1, se_ratio, activation, 6, bn_training, mome)
    x = _inverted_res_block(x, 3, depth(48), kernel, 1, se_ratio, activation, 7, bn_training, mome)
    x = _inverted_res_block(x, 6, depth(96), kernel, 2, se_ratio, activation, 8, bn_training, mome)
    x = _inverted_res_block(x, 6, depth(96), kernel, 1, se_ratio, activation, 9, bn_training, mome)
    x = _inverted_res_block(x, 6, depth(96), kernel, 1, se_ratio, activation, 10, bn_training, mome)

    return x

# 定义mobilenetv3-large的骨干部分，不包含第一层的卷积特征提取和后处理
def MobileNetV3Large(x, kernel, activation, se_ratio, alpha, bn_training, mome):

    def depth(d):
        return _depth(d * alpha)

    x = _inverted_res_block(x, 1, depth(16), 3, 1, None, relu, 0, bn_training, mome)
    x = _inverted_res_block(x, 4, depth(24), 3, 2, None, relu, 1, bn_training, mome)
    x = _inverted_res_block(x, 3, depth(24), 3, 1, None, relu, 2, bn_training, mome)
    x = _inverted_res_block(x, 3, depth(40), kernel, 2, se_ratio, relu, 3, bn_training, mome)
    x = _inverted_res_block(x, 3, depth(40), kernel, 1, se_ratio, relu, 4, bn_training, mome)
    x = _inverted_res_block(x, 3, depth(40), kernel, 1, se_ratio, relu, 5, bn_training, mome)
    x = _inverted_res_block(x, 6, depth(80), 3, 2, None, activation, 6, bn_training, mome)
    x = _inverted_res_block(x, 2.5, depth(80), 3, 1, None, activation, 7, bn_training, mome)
    x = _inverted_res_block(x, 2.3, depth(80), 3, 1, None, activation, 8, bn_training, mome)
    x = _inverted_res_block(x, 2.3, depth(80), 3, 1, None, activation, 9, bn_training, mome)
    x = _inverted_res_block(x, 6, depth(112), 3, 1, se_ratio, activation, 10, bn_training, mome)
    x = _inverted_res_block(x, 6, depth(112), 3, 1, se_ratio, activation, 11, bn_training, mome)
    x = _inverted_res_block(x, 6, depth(160), kernel, 2, se_ratio, activation, 12, bn_training, mome)
    x = _inverted_res_block(x, 6, depth(160), kernel, 1, se_ratio, activation, 13, bn_training, mome)
    x = _inverted_res_block(x, 6, depth(160), kernel, 1, se_ratio, activation, 14, bn_training, mome)

    return x


##################################################################################################################################
# 定义V3的骨干模块 #################################################################################################################
##################################################################################################################################

# 定义relu函数
def relu(x):
    return layers.ReLU()(x)

# 定义sigmoid函数的近似函数h-sigmoid函数
def hard_sigmoid(x):
    return layers.ReLU(6.)(x + 3.) * (1. / 6.)

# 定义swish函数的近似函数，替换原本的sigmoid函数为新的h-sigmoid函数
def hard_swish(x):
    return layers.Multiply()([x, hard_sigmoid(x)])

# python中变量前加单下划线：是提示程序员该变量或函数供内部使用，但不是强制的，只是提示，但是不能用“from xxx import *”而导入
# python中变量后加单下划线：是避免变量名冲突
# python中变量前加双下划线：是强制该变量或函数供类内部使用，名称会被强制修改，所以原名称无法访问到，新名称可以访问到，所以也不是外部完全无法访问
# python中变量前后双下划线：是用于类内部定义使用，是特殊用途，外部可以直接访问，平时程序员不要这样定义
# 通过函数实现不管v为多大，输出new_v始终能够被divisor整除，且new_v是大于等于min_value且不能太小的四舍五入最接近divisor整除的数
def _depth(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    # 保证new_v一定大于等于min_value，max中第二个值保证是v的四舍五入的能够被divisor整除的数
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # 保证new_v不要太小
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

# 在stride等于2时，计算pad的上下左右尺寸，注:在stride等于1时，无需这么麻烦，直接就是correct，本函数仅仅针对stride=2
def pad_size(inputs, kernel_size):
    input_size = inputs.shape[1:3]
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if input_size[0] is None:
        adjust = (1,1)
    else:
        adjust = (1- input_size[0]%2, 1-input_size[1]%2)
    correct = (kernel_size[0]//2, kernel_size[1]//2)
    return ((correct[0] - adjust[0], correct[0]),
            (correct[1] - adjust[1], correct[1]))

# 定义通道注意力机制模块，这里filters数字应该是要等于inputs的通道数的，否则最后一步的相乘无法完成，se_ratio可以调节缩放比例
def _se_block(inputs, filters, se_ratio, prefix):
    # 如果tf版本大于等于2.6则直接使用下面第一句就可以了，否则使用下面2~3句
    # x = layers.GlobalAveragePooling2D(data_format='channels_last', keepdims=True, name=prefix + 'squeeze_excite/AvgPool')(inputs)
    x = layers.GlobalAveragePooling2D(data_format='channels_last', name=prefix + 'squeeze_excite/AvgPool')(inputs)
    x= tf.expand_dims(tf.expand_dims(x, 1), 1)
    x = layers.Conv2D(_depth(filters * se_ratio), kernel_size=1, padding='same', name=prefix + 'squeeze_excite/Conv')(x)
    x = layers.ReLU(name=prefix + 'squeeze_excite/Relu')(x)
    x = layers.Conv2D(filters, kernel_size=1, padding='same', name=prefix + 'squeeze_excite/Conv_1')(x)
    x = hard_sigmoid(x)
    x = layers.Multiply(name=prefix + 'squeeze_excite/Mul')([inputs, x])
    return x

# 定义V3的基础模块，可以通过expansion调整模块中所有特整层的通道数，se_ratio可以调节通道注意力机制中的缩放系数
def _inverted_res_block(x, expansion, filters, kernel_size, stride, se_ratio, activation, block_id, bn_training, mome):
    channel_axis = -1  # 在tf中通道维度是最后一维
    shortcut = x
    prefix = 'expanded_conv/'
    infilters = x.shape[channel_axis]
    if block_id:
        prefix = 'expanded_conv_{}/'.format(block_id)
        x = layers.Conv2D(_depth(infilters * expansion), kernel_size=1, padding='same', use_bias=False, name=prefix + 'expand')(x)
        x = layers.BatchNormalization(axis=channel_axis, epsilon=1e-3, momentum=mome, name=prefix + 'expand/BatchNorm')(x, training=bn_training)
        x = activation(x)

    if stride == 2:
        x = layers.ZeroPadding2D(padding=pad_size(x, kernel_size), name=prefix + 'depthwise/pad')(x)
    x = layers.DepthwiseConv2D(kernel_size, strides=stride, padding='same' if stride == 1 else 'valid', use_bias=False, name=prefix + 'depthwise')(x)
    x = layers.BatchNormalization(axis=channel_axis, epsilon=1e-3, momentum=mome, name=prefix + 'depthwise/BatchNorm')(x, training=bn_training)
    x = activation(x)

    if se_ratio:
        x = _se_block(x, _depth(infilters * expansion), se_ratio, prefix)

    x = layers.Conv2D(filters, kernel_size=1, padding='same', use_bias=False, name=prefix + 'project')(x)
    x = layers.BatchNormalization(axis=channel_axis, epsilon=1e-3, momentum=mome, name=prefix + 'project/BatchNorm')(x, training=bn_training)

    if stride == 1 and infilters == filters:
        x = layers.Add(name=prefix + 'Add')([shortcut, x])
    return x

## no.1
# 在keras中每个model以及model中的layer都存在trainable属性，如果将trainable属性设置为False，那么相应的model或者layer所对应
# 的参数将不会再改变，但是当前不建议直接对model操作，建议直接对layer进行操作，原因是当前有bug，对model设置后，有可能再对layer进
# 行操作就失效了。后面证明这个并非bug，而是model和layer都存在trainable属性，对model的设置会影响到layer的设置，但是对layer的
# 设置不会影响到model的设置，当首先设置model的trainable属性为False时，后面不管对layer的trainable属性怎么设置，都不会改变model
# 的trainable属性为False这一事实，当调用训练函数时，框架首先检查model的trainable属性，如果该属性为False，那就是终止训练，所以
# 不管内部的layer的trainable属性怎么设置都没用。此外，BN层和Dropout层还存在training的形参，这个形参是用来告诉对应层属于train
# 状态还是infer状态，例如BN层，其在train状态采用的是当前batch的均值和方差，并维护一个滑动平均的均值和方差，在infer状态采用的是之
# 前维护的滑动平均的均值和方差。原本trainable属性和training形参是相互独立的，但是在BN层这里是个例外，就是当BN层的最终trainable
# 属性为True时，一切正常，BN层的线性变换系数可以训练可以被修改，BN层的training设置也符合上面所述。但是当BN层的trainable属性为
# False时，就会出现问题，此时线性变换系数不可以训练不可以被修改，这个正常，但是此时BN层将处在infer状态，即trianing参数被修改为
# False，此时滑动均值和方差不会再修改，也就是说在调用fit()时，BN层将采用之前的滑动均值和方差进行计算，并不是当前batch的均值和方差，
# 且不会维护着滑动平均的均值和方差。这个造成的问题是在迁移学习时，从只是训练最后一层变换到训练整个网络时，整个误差和acc都会剧降，原因
# 就是在冻结训练时，BN层处在不可训练状态，那么其BN一直采用的是旧数据的均值和方差，且没有维护滑动平均的均值和方差，当变换到全网络训练时，
# BN层处在可训练状态，此时BN层采用的当前batch的的均值和方差，且开始维护着滑动平均的均值和方差，这会造成后面的分类层无法使用BN层中的
# 参数巨变，进而对识别精度产生重大影响。所以，问题的根本原因是在BN层处于不可训练状态时，其会自动处在infer状态，解决这一问题最简单的方式
# 是，在定义网络时直接把BN层的training设置为False，这样不管BN层处在何种状态，BN层都是采用旧数据的均值和方差进行计算，不会再更新，
# 这样就不会出现参数巨变也不会出现准确率剧降，也可以直接先计算一下新数据整体的均值和方差，然后在迁移学习时，先把方差和均值恢复进网络里，
# 同时training设置为False。关于BN与training参数和trainable属性的相互影响，详细见自己的CSDN博客。

## no.2
# 测试中间模型准确率，第一次调试时遇到一个问题就是当没有采用迁移学习而是整个网络随机初始化且同时训练时，fit在训练集上进行训练，acc
# 逐步提升很正常，但是同步在验证集和测试集上acc在前7~8轮训练完全不增加，最后增加了，也增加的相当有限，最后排查原因发现，是因为网络
# 中有BN结构造成的，BN结构中存在一个均值和方差，它们是通过步进平滑计算得到的，最终这两个值趋近于全部数据集的整体均值和方差
# (batchsize==1，平滑系数==0.99时，趋近于时间上最近的几百多个数据的类似平均，如果加大batchsize和增大平滑系数，最终趋近于整体的
# 均值和方差，所以其实也可以直接计算整体均值和方差然后赋值)，但是如果刚开始训练时batchsize设置过大，而总数量不足将会导致训练完一轮
# 以后，steps数过小，如果此时平滑系数还很大，那步进计算的均值和方差将非常接近于初始的随机值而不是数据集的平均值，那在测试状态下，网
# 络的输出结果就很差，而在训练状态下，这个均值和方差是通过一个batch实时计算的，后面匹配的线性变换也是实时改变的，所以质量比较好，所
# 以才会出现同样是训练集fit时acc很好，但是evaluate时acc巨差的现象，所以在数据集比较小时，且不是迁移学习时，batchsize可以设置的
# 小一点以及滑动系数设置的小一点。

## no.3
# 需要说明的是在定义网络结构时如果没有指定Dropout和BN层的training属性，那tf会根据所调用函数自动设置，例如调用fit函数则为True，调用evaluate和
# predict函数则为False，调用__call__函数时，默认是False，但是可以手动设置。但是如果在定义网络结构时给予了具体布尔值，则不管调用任何函数，都按照
# 实际设置的属性使用

## no.4
# 更详细的讲解详见CSDN博客<BN(Batch Normalization) 的理论理解以及在tf.keras中的实际应用和总结>
