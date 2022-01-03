# 20211202 根据tf.keras的官方代码修改的mobilenetv3的网络模型，并通过tensorflow2.x的方式进行迁移学习
import os
import time
import shutil
import numpy as np
import tensorflow as tf
from mobilenetv3 import MobileNetV3
from data_read_and_preprocess import data_read_and_preprocess

############################################################################################################
## 0. 参数设置 ##############################################################################################
############################################################################################################

IMG_SIZE = (128, 128, 3)
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
weight_path = "./checkpoint/weights_mobilenet_v3_large_1.0_224_notop.h5"
output_path = "./checkpoint/h5/"
############################################################################################################
## 1. 读取数据和数据预处理 #####################################################################################
############################################################################################################
train_dataset, test_dataset = data_read_and_preprocess(IMG_SIZE, BATCH_SIZE, CLASS_NUM, train_dir, test_dir, True)

############################################################################################################
## 2. 搭建网络结构 ###########################################################################################
############################################################################################################
model = MobileNetV3(input_shape=IMG_SIZE, classes=CLASS_NUM, dropout_rate=0.2, alpha=alpha, 
                                                weights=weight_path, model_type='large', minimalistic=False)
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
    output_model = path + "{}_mobilenetv3-v0.5_{}_{}_{}_{}_{:.4f}_{:.4f}.h5".format(dataset_name, localtime, IMG_SIZE[0], alpha, epoch, acc1, acc2)
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

# 控制仅仅最后的分类层可训练，实际测试发现如果model.trainable = False然后放开最后一层，这样操作不可以，网络不认可，也许是个bug
for layer in model.layers[:-3]:   # 改变trainable属性后，必须调用model.compile才能使其生效
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
