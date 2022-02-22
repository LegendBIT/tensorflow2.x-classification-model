# 通过tf.keras实现自定义的mobilenetv2网络并通过tensorflow2.x训练
import os
import time
import random
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Dense, Dropout, BatchNormalization, ReLU
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, ZeroPadding2D

############################################################################################################
## 0. 参数设置 ##############################################################################################
############################################################################################################

IMG_SIZE = (128, 128)
BATCH_SIZE = 128
CLASS_NUM = 7
alpha = 1.0           # 模型通道缩放系数
initial_epochs = 4    # 第一轮仅仅训练最后一层
second_epochs = 6     # 第二轮训练整个网络
initial_learning_rate = 0.0001
second_learning_rate = 0.00001
dataset_name = "XXX"
train_dir = "./train"
test_dir = "./test"
weight_path = "./checkpoint/mobilenet_v2_no_top.h5"
output_path = "./checkpoint/h5/"

############################################################################################################
## 1. 读取数据和数据预处理 #####################################################################################
############################################################################################################

# 得到dir路径下所有文件的具体路径以及标签，file_paths_list, labels_list都是list分别对应着每个文件的具体路径和标签编号，class_names对应标签名
def load_file_paths_and_labels(dir):
    print("--> Loading file paths and labels...")
    image_format = ["bmp", "jpg", "jpeg", "png"]  # 可以读取的文件类型
    file_paths_list, labels_list, class_names = [], [], []
    for path, _, files in os.walk(dir):           # path是dir路径下每个文件夹的路径和dir文件夹本身的路径，files是每个文件夹下的文件名列表
        if path == dir: continue                  # dir本身路径不要
        for file in files:
            if file.split(".")[-1] not in image_format:           # 在mac中每个文件夹里有一个隐藏的.DS_Store，予以删除
                continue
            file_paths_list.append(os.path.join(path, file))
            labels_list.append(path.split("/")[-1])
        class_names.append(path.split("/")[-1])
    class_names.sort()                            # 标签名按照字母顺序排列
    labels_list = [class_names.index(label) for label in labels_list]
    randnum = random.randint(0, 666)              # 为了将读取的图片列表和标签列表打乱
    random.seed(randnum)
    random.shuffle(file_paths_list)
    random.seed(randnum)
    random.shuffle(labels_list)
    print("<-- Loading end...")
    return file_paths_list, labels_list, class_names

# 对path路径下的图片进行读取和预处理，当前针对分类问题，没有对标签进行处理，如果是检测问题，也可以对label进行处理
def load_preprosess_image(path, label):
    # 读取图片文件得到字符串
    image = tf.io.read_file(path)
    # 解码字符串得到图片
    image = tf.io.decode_image(image, channels=3)  # 彩色图像为3个channel
    # 对图片尺寸进行等比例拉伸
    image = tf.image.resize_with_pad(image, IMG_SIZE[0], IMG_SIZE[1])
    # 随机左右翻转图像
    image = tf.image.random_flip_left_right(image)
    # 随机改变图像的亮度，随机改变[-0.1 ~ 0.1]大小的值 tf.image.random_brightness(image,max_delta,seed=None)为随机调整亮度函数，
    # 实际上是在原图的基础上随机加上一个值(如果加上的是正值则增亮否则增暗)，此值取自[-max_delta,max_delta)，要求max_delta>=0。
    image = tf.image.random_brightness(image, 0.1)
    # 改变数据类型
    image = tf.cast(image, tf.float32)
    # 对图片进行像素拉伸
    # image = image * (255 / (tf.reduce_max(image) - tf.reduce_min(image)))
    # 对图像进行归一化，归一化的方法有很多，这个是其中一种，归一化到[-1, 1]
    # image = (image - np.array([65, 70, 69]))/(3*np.array([44, 43, 43]))
    image = image/127.5 - 1.0
    # 对标签进行处理得到one-hot形式标签
    label = tf.one_hot(label, depth=CLASS_NUM)
    return image, label

# 得到图片预处理类tf.data.dataset，相当于yolov4自己写的数据预处理类，但是自带的性能更好，由于读取图片列表时已经打乱，所以这里的再次打乱池可以设置的小
# 一点，原理是首先拿到的图片名列表就是打乱的，但是这个仅仅打乱一次，所以如果不设置shuttle属性，那么每个epoch的数据读取顺序都是相同的，shuttle的原理是
# 如果shuttle设置为10，batchsize设置为2，那么第一次读取数据是从这个10个图中随机抽取2个图，然后按照顺序从后面序列中补充两个图组成10个图，然后重复刚刚
# 随机抽取过程，所以一般如果读取图片的列表没有事先打乱，那么shuttle设置为len(data)是最好的，但是如果这样，当数据集过大时，内存要求太大且读取很慢，所以
# 我们选择事先打乱图片名列表，这样shuttle就没必要设置的太大了。prefetch()的原理是设置为当训练网络时同步进行数据集读取，节省总的训练时间。
# 训练集：得到图片预处理类tf.data.dataset，相当于yolov4自己写的数据预处理类，但是自带的性能更好
file_paths_list, labels_list, train_labels = load_file_paths_and_labels(train_dir)     # 读取文件列表
train_dataset = tf.data.Dataset.from_tensor_slices((file_paths_list, labels_list))     # 生成dataset对象
AUTOTUNE = tf.data.experimental.AUTOTUNE                                               # 根据计算机性能进行运算速度的调整
train_dataset = train_dataset.map(load_preprosess_image, num_parallel_calls=AUTOTUNE)  # 设置数据处理函数
train_dataset = train_dataset.shuffle(int(len(train_dataset)/10)).batch(BATCH_SIZE)    # 设置dataset属性，len(train_dataset)太大，所以改为1/10
# train_dataset = train_dataset.shuffle(len(train_dataset)).batch(BATCH_SIZE)            # 设置dataset属性
train_dataset = train_dataset.prefetch(AUTOTUNE)                                       # 预处理一部分处理，准备读取

# 测试集：得到图片预处理类tf.data.dataset，相当于yolov4自己写的数据预处理类，但是自带的性能更好
file_paths_list, labels_list, test_labels = load_file_paths_and_labels(test_dir)       # 读取文件列表
test_dataset = tf.data.Dataset.from_tensor_slices((file_paths_list, labels_list))      # 生成dataset对象
AUTOTUNE = tf.data.experimental.AUTOTUNE                                               # 根据计算机性能进行运算速度的调整
test_dataset = test_dataset.map(load_preprosess_image, num_parallel_calls=AUTOTUNE)    # 设置数据处理函数
# test_dataset = test_dataset.shuffle(len(test_dataset)).batch(BATCH_SIZE)               # 设置dataset属性
test_dataset = test_dataset.batch(BATCH_SIZE)                                          # 设置dataset属性，测试集没必要打乱顺序
test_dataset = test_dataset.prefetch(AUTOTUNE)                                         # 预处理一部分处理，准备读取

############################################################################################################
## 2. 搭建网络结构 ###########################################################################################
############################################################################################################
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


# 保证特征层数为8的倍数，输入为v和divisor，v是除数，divisor是被除数，将输出值改造为最接近v的divisor的倍数
def make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v+divisor/2)//divisor*divisor)   # 四舍五入取倍数值
    if new_v < 0.9*v:
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

# 定义基本的卷积模块
def conv_block(x, nb_filter, kernel=(1,1), stride=(1,1), name=None):
    x = Conv2D(nb_filter, kernel, strides=stride, padding='same', use_bias=False, name=name+'_expand')(x)
    x = BatchNormalization(axis=3, name=name+'_expand_BN')(x, training=False)
    # x = Activation(relu6, name=name+'_expand_relu')(x)  # 采用这种Activation+自定义relu6的方式实现激活函数将导致后面无法量化感知训练
    x = ReLU(max_value=6.0, name=name+'_expand_relu')(x)
    return x

# 定义特有的残差卷积模块
def depthwise_res_block(x, nb_filter, kernel, stride, t, alpha, resdiual=False, name=None):
    # 准备工作
    input_tensor = x                        # 分一只出来等着残差结构的链接
    exp_channels = x.shape[-1]*t            # V2中特有的扩展维度，即先扩展维度再深度可分离卷积再缩小维度
    alpha_channels = int(nb_filter*alpha)   # 压缩维度，针对v2中几乎所有卷积层进行通道维度的压缩，从整个结构的第一层就开始了
    # 特有机构中的第一个卷积层，起到扩展维度的作用
    x = conv_block(x, exp_channels, (1,1), (1,1), name=name)
    # 在深度可分离卷积前，依据stride进行padding操作
    if stride[0] == 2:
        x = ZeroPadding2D(padding=pad_size(x, 3), name=name+'_pad')(x)
    # 进行深度可分离卷积操作，这里没有用集成函数，所以分两步进行卷积
    x = DepthwiseConv2D(kernel, padding='same' if stride[0]==1 else 'valid', strides=stride, depth_multiplier=1, use_bias=False, name=name+'_depthwise')(x)
    x = BatchNormalization(axis=3, name=name+'_depthwise_BN')(x, training=False)
    x = ReLU(max_value=6.0, name=name+'_depthwise_relu')(x)
    # 深度可分离中的第二步，可以减小维度
    x = Conv2D(alpha_channels, (1,1), padding='same', use_bias=False, strides=(1,1), name=name+'_project')(x)
    x = BatchNormalization(axis=3, name=name+'_project_BN')(x, training=False)
    # 是否需要残差结构，如果有残差，需要深度维度一致才能相加
    if resdiual:
        x = layers.add([x, input_tensor], name=name+'_add')
    return x

# 定义整个v2结构，特色就是独有的残差模块
def MovblieNetV2(img_size, nb_classes, alpha=1.0, dropout=0):
    # 输入端口
    img_input = Input(shape=img_size+(3,))
    # 第一个卷积模块是普通卷积
    first_filter = make_divisible(32*alpha, 8)
    x = ZeroPadding2D(padding=pad_size(img_input, 3), name='Conv1_pad')(img_input)
    x = Conv2D(first_filter, (3,3), strides=(2,2), padding='valid', use_bias=False, name='Conv1')(x)
    x = BatchNormalization(axis=3, name='bn_Conv1')(x, training=False)
    x = ReLU(max_value=6.0, name='Conv1_relu')(x)
    # 第一个深度可分离卷积模块，由于膨胀系数等于1，与剩余的深度可分离模块不兼容，所以无法使用depthwise_res_block函数
    x = DepthwiseConv2D((3,3), padding='same', strides=(1,1), depth_multiplier=1, use_bias=False, name='expanded_conv_depthwise')(x)
    x = BatchNormalization(axis=3, name='expanded_conv_depthwise_BN')(x, training=False)
    x = ReLU(max_value=6.0, name='expanded_conv_depthwise_relu')(x)
    x = Conv2D(16, (1,1), padding='same', use_bias=False, strides=(1,1), name='expanded_conv_project')(x)
    x = BatchNormalization(axis=3, name='expanded_conv_project_BN')(x, training=False)
    # 第二组特有深度可分离卷积组
    x = depthwise_res_block(x, 24, (3,3), (2,2), 6, alpha, resdiual=False, name='block_1') 
    x = depthwise_res_block(x, 24, (3,3), (1,1), 6, alpha, resdiual=True, name='block_2') 
    # 第三组特有深度可分离卷积组
    x = depthwise_res_block(x, 32, (3,3), (2,2), 6, alpha, resdiual=False, name='block_3')
    x = depthwise_res_block(x, 32, (3,3), (1,1), 6, alpha, resdiual=True, name='block_4')
    x = depthwise_res_block(x, 32, (3,3), (1,1), 6, alpha, resdiual=True, name='block_5')
    # 第四组特有深度可分离卷积组
    x = depthwise_res_block(x, 64, (3,3), (2,2), 6, alpha, resdiual=False, name='block_6')
    x = depthwise_res_block(x, 64, (3,3), (1,1), 6, alpha, resdiual=True, name='block_7')
    x = depthwise_res_block(x, 64, (3,3), (1,1), 6, alpha, resdiual=True, name='block_8')
    x = depthwise_res_block(x, 64, (3,3), (1,1), 6, alpha, resdiual=True, name='block_9')
    # 第五组特有深度可分离卷积组
    x = depthwise_res_block(x, 96, (3,3), (1,1), 6, alpha, resdiual=False, name='block_10')
    x = depthwise_res_block(x, 96, (3,3), (1,1), 6, alpha, resdiual=True, name='block_11')
    x = depthwise_res_block(x, 96, (3,3), (1,1), 6, alpha, resdiual=True, name='block_12')
    # 第六组特有深度可分离卷积组
    x = depthwise_res_block(x, 160, (3,3), (2,2), 6, alpha, resdiual=False, name='block_13')
    x = depthwise_res_block(x, 160, (3,3), (1,1), 6, alpha, resdiual=True, name='block_14')
    x = depthwise_res_block(x, 160, (3,3), (1,1), 6, alpha, resdiual=True, name='block_15')
    # 第七组特有深度可分离卷积组
    x = depthwise_res_block(x, 320, (3,3), (1,1), 6, alpha, resdiual=False, name='block_16')
    # 通道数计算
    if alpha > 1.0:
        last_filter = make_divisible(1280*alpha,8)
    else:
        last_filter = 1280
    # 特征提取网络的最后一个卷积是普通卷积
    x = Conv2D(last_filter, (1,1), strides=(1,1), use_bias=False, name='Conv_1')(x)
    x = BatchNormalization(axis=3, name='Conv_1_bn')(x, training=False)
    x = ReLU(max_value=6.0, name='out_relu')(x)
    # 通过全局均值池化对接特征提取网络和特征分类网络
    x = GlobalAveragePooling2D()(x)
    # 特征分类网络
    x = Dropout(dropout)(x)
    x = Dense(nb_classes, activation='softmax', use_bias=True, name='Logits')(x)
    # 搭建keras模型
    model = models.Model(img_input, x, name='MobileNetV2')
    # 返回结果模型
    return model
    # 需要说明的是在定义网络结构时如果没有指定Dropout和BN层的training属性，那tf会根据所调用函数自动设置，例如调用fit函数则为True，调用evaluate和
    # predict函数则为False，调用__call__函数时，默认是False，但是可以手动设置。但是如果在定义网络结构时给予了具体布尔值，则不管调用任何函数，都按照
    # 实际设置的属性使用

# 生成整个网络模型
model = MovblieNetV2(IMG_SIZE, CLASS_NUM, alpha, 0.2)
model.summary()

############################################################################################################
## 3. 定义优化器，日志记录器和模型保存函数 #######################################################################
############################################################################################################
optimizer = tf.keras.optimizers.Adam()
logdir = "./log"
global_steps = tf.Variable(0, trainable=False, dtype=tf.int64)  # 用于写日志时记录step
if os.path.exists(logdir): shutil.rmtree(logdir)
writer = tf.summary.create_file_writer(logdir)
# 保存模型参数和结构
def save_model(epoch, acc1, acc2, path, dataset_name, model, alpha):
    localtime = time.strftime("%Y%m%d-%H%M", time.localtime())
    output_model = path + "{}_mobilenetv2-v0.5_{}_{}_{}_{}_{:.4f}_{:.4f}.h5".format(dataset_name, localtime, IMG_SIZE[0], alpha, epoch, acc1, acc2)
    model.save(output_model)
############################################################################################################
## 4. 定义训练函数和测试函数 ###################################################################################
############################################################################################################
## 由于tf.function()的缘故，第一和第二次训练需要完整的copy两份代码
# 单次训练函数
@tf.function
def train_step(gt_x, gt_y, epoch):  # 定义单次训练函数
    # 4.1 计算梯度，loss和acc，这里acc是指训练前的acc
    with tf.GradientTape() as tape:
        y = model(gt_x, training=True)  # model()函数需要指定training属性，该属性影响BN和dropout层，指定当前是训练模式还是测试模式，与trainable属性是两码事
        loss = -tf.reduce_sum(gt_y*tf.math.log(y))  # 自定义的交叉熵损失函数，模型最后一层本身包含softmax操作
    gradients = tape.gradient(loss, model.trainable_variables)
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(gt_y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # 4.2 更新权重
    if epoch < initial_epochs:
        optimizer.lr.assign(initial_learning_rate)
    else:
        optimizer.lr.assign(second_learning_rate)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, accuracy, optimizer.lr    # 这里acc是指训练前的acc

# 单次测试函数
@tf.function
def test_step(gt_x, gt_y):  # 定义单次测试函数
    # 4.3 测试网络
    y = model(gt_x, training=False)
    loss = -tf.reduce_sum(gt_y*tf.math.log(y))  # 自定义的交叉熵损失函数
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(gt_y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return loss, accuracy

# 训练函数
def train(train_dataset, test_dataset, epoch):
    total_train_loss = []
    total_train_acc = []
    total_test_loss = []
    total_test_acc  = []
    # 4.4 训练网络
    for images, labels in train_dataset:
        train_loss, train_acc, train_lr = train_step(images, labels, epoch)  # 这里acc是训练前的acc
        total_train_loss.append(train_loss.numpy())
        total_train_acc.append(train_acc.numpy())
        # 4.5 写一次日志
        global_steps.assign_add(1)
        with writer.as_default():
            tf.summary.scalar("lr", train_lr, step=global_steps)  # global_steps必须这样写，如果改成整形数字n，不仅有警告，还运行贼慢！
            tf.summary.scalar("train_loss", train_loss, step=global_steps)
            tf.summary.scalar("train_acc", train_acc, step=global_steps) 
    writer.flush()
    # 4.6 测试网络
    for images, labels in test_dataset:
        test_loss, test_acc = test_step(images, labels)
        total_test_loss.append(test_loss.numpy())
        total_test_acc.append(test_acc.numpy())
    # 注意训练集acc的计算方式并不是训练完成后统一跑一次训练集
    return np.mean(total_train_loss, 0), np.mean(total_train_acc, 0), np.mean(total_test_loss, 0), np.mean(total_test_acc, 0)

# 测试函数
def test(test_dataset):
    total_test_loss = []
    total_test_acc  = []
    # 4.7 测试网络
    for images, labels in test_dataset:
        test_loss, test_acc = test_step(images, labels)
        total_test_loss.append(test_loss.numpy())
        total_test_acc.append(test_acc.numpy())
    return np.mean(total_test_loss, 0), np.mean(total_test_acc, 0)

## 由于tf.function()的缘故，第一和第二次训练需要完整的copy两份代码
# 单次训练函数
@tf.function
def train_step2(gt_x, gt_y, epoch):  # 定义单次训练函数
    # 4.1 计算梯度，loss和acc，这里acc是指训练前的acc
    with tf.GradientTape() as tape:
        y = model(gt_x, training=True)  # model()函数需要指定training属性，该属性影响BN和dropout层，指定当前是训练模式还是测试模式，与trainable属性是两码事
        loss = -tf.reduce_sum(gt_y*tf.math.log(y))  # 自定义的交叉熵损失函数
    gradients = tape.gradient(loss, model.trainable_variables)
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(gt_y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # 4.2 更新权重
    if epoch < initial_epochs:
        optimizer.lr.assign(initial_learning_rate)
    else:
        optimizer.lr.assign(second_learning_rate)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, accuracy, optimizer.lr    # 这里acc是指训练前的acc

# 单次测试函数
@tf.function
def test_step2(gt_x, gt_y):  # 定义单次测试函数
    # 4.3 测试网络
    y = model(gt_x, training=False)
    loss = -tf.reduce_sum(gt_y*tf.math.log(y))  # 自定义的交叉熵损失函数
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(gt_y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return loss, accuracy

# 训练函数
def train2(train_dataset, test_dataset, epoch):
    total_train_loss = []
    total_train_acc = []
    total_test_loss = []
    total_test_acc  = []
    # 4.4 训练网络
    for images, labels in train_dataset:
        train_loss, train_acc, train_lr = train_step2(images, labels, epoch)  # 这里acc是训练前的acc
        total_train_loss.append(train_loss.numpy())
        total_train_acc.append(train_acc.numpy())
        # 4.5 写一次日志
        global_steps.assign_add(1)
        with writer.as_default():
            tf.summary.scalar("lr", train_lr, step=global_steps)  # global_steps必须这样写，如果改成整形数字n，不仅有警告，还运行贼慢！
            tf.summary.scalar("train_loss", train_loss, step=global_steps)
            tf.summary.scalar("train_acc", train_acc, step=global_steps) 
    writer.flush()
    # 4.6 测试网络
    for images, labels in test_dataset:
        test_loss, test_acc = test_step2(images, labels)
        total_test_loss.append(test_loss.numpy())
        total_test_acc.append(test_acc.numpy())
    # 注意训练集acc的计算方式并不是训练完成后统一跑一次训练集
    return np.mean(total_train_loss, 0), np.mean(total_train_acc, 0), np.mean(total_test_loss, 0), np.mean(total_test_acc, 0)

# 测试函数
def test2(test_dataset):
    total_test_loss = []
    total_test_acc  = []
    # 4.7 测试网络
    for images, labels in test_dataset:
        test_loss, test_acc = test_step2(images, labels)
        total_test_loss.append(test_loss.numpy())
        total_test_acc.append(test_acc.numpy())
    return np.mean(total_test_loss, 0), np.mean(total_test_acc, 0)   

############################################################################################################
## 5. 训练神经网络 ###########################################################################################
############################################################################################################

# 恢复网络权重，不加by_name，则要求整个网络结构与权重文件完全一致才可以恢复(不包含参数的层不考虑在内)，模型中层的命名可以不一致；
# 加了by_name，其按照命名恢复权重，一定要仔细检查命名是否一致，第一次调试的时候遇到当冻结所有层仅仅保留最后一层进行训练时，网络
# 不收敛，甚至acc完全不增加，原因就是我们采用了(加了by_name)的方式恢复权重，但是权重文件中有部分层的命名与我们自定义的网络不一致，
# 导致冻结层的输出完全不对，所以最后一层无法迁移训练，采用by_name恢复权重时，命名不一致也不会提醒，只能自己排查，其找到命名以及
# 结构一致的层进行参数恢复，名字一样但结构不一样会报错，命名不一致自动跳过
model.load_weights(weight_path, by_name=True)
# print(model.get_layer(name="block_8_project_BN").get_weights()[0][:4])

# 控制仅仅最后的分类层可训练，实际测试发现如果model.trainable = False然后放开最后一层，这样操作不可以，网络不认可，也许是个bug
for layer in model.layers[:-1]:   # 改变trainable属性后，必须调用model.compile才能使其生效
    layer.trainable = False
    
# 编译模型，以便后面可以使用evaluate评估函数
model.compile(loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])

# 测试初始模型准确率
loss0, accuracy0 = test(test_dataset)
print("")
print("initial loss: {:.4f}".format(loss0))
print("initial accuracy: {:.4f}".format(accuracy0))
print("")

# 训练模型，只是训练最后一层，冻结前面所有层
for i in range(initial_epochs):
    loss, acc, val_loss, val_acc = train(train_dataset, test_dataset, i)
    save_model(i+1, acc, val_acc, "./tmp/", dataset_name, model, alpha)
    print("epoch: %d, loss: %0.5f, acc: %0.4f, val_loss: %0.5f, val_acc: %0.4f" % (i+1, loss, acc, val_loss, val_acc))
    print("")
# print(model.get_layer(name="block_8_project_BN").get_weights()[0][:4])

# 测试中间模型准确率，第一次调试时遇到一个问题就是当没有采用迁移学习而是整个网络随机初始化且同时训练时，fit在训练集上进行训练，acc
# 逐步提升很正常，但是同步在验证集和测试集上acc在前7~8轮训练完全不增加，最后增加了，也增加的相当有限，最后排查原因发现，是因为网络
# 中有BN结构造成的，BN结构中存在一个均值和方差，它们是通过步进平滑计算得到的，最终这两个值趋近于全部数据集的整体均值和方差
# (batchsize==1，平滑系数==0.99时，趋近于时间上最近的几百多个数据的类似平均，如果加大batchsize和增大平滑系数，最终趋近于整体的
# 均值和方差，所以其实也可以直接计算整体均值和方差然后赋值)，但是如果刚开始训练时batchsize设置过大，而总数量不足将会导致训练完一轮
# 以后，steps数过小，如果此时平滑系数还很大，那步进计算的均值和方差将非常接近于初始的随机值而不是数据集的平均值，那在测试状态下，网
# 络的输出结果就很差，而在训练状态下，这个均值和方差是通过一个batch实时计算的，后面匹配的线性变换也是实时改变的，所以质量比较好，所
# 以才会出现同样是训练集fit时acc很好，但是evaluate时acc巨差的现象，所以在数据集比较小时，且不是迁移学习时，batchsize可以设置的
# 小一点以及滑动系数设置的小一点。
loss1, accuracy1 = test(test_dataset)
print("middle loss: {:.4f}".format(loss1))
print("middle accuracy: {:.4f}".format(accuracy1))
print("")

# 放开全部网络层，开始第二轮训练
model.trainable = True
for layer in model.layers[:-1]:   # 改变trainable属性后，必须调用model.compile才能使其生效
    layer.trainable = True

# 开始第二轮训练；回调函数的顺序有影响，需要先学习率再自定义的，这样logs里才有lr
total_epochs =  initial_epochs + second_epochs
for i in range(initial_epochs, total_epochs):
    loss, acc, val_loss, val_acc = train2(train_dataset, test_dataset, i)
    save_model(i+1, acc, val_acc, "./tmp/", dataset_name, model, alpha)
    print("epoch: %d, loss: %0.5f, acc: %0.4f, val_loss: %0.5f, val_acc: %0.4f" % (i+1, loss, acc, val_loss, val_acc))
    print("")

# 测试输出模型准确率
loss2, accuracy2 = test2(test_dataset)
print("last loss: {:.4f}".format(loss2))
print("last accuracy: {:.4f}".format(accuracy2))

# 保存模型参数和结构
save_model(total_epochs, acc, accuracy2, output_path, dataset_name, model, alpha)
