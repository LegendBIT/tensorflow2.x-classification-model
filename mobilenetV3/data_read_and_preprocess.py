# 将数据读取和预处理单独拿出来
import os
import random
import tensorflow as tf


def data_read_and_preprocess(IMG_SIZE, BATCH_SIZE, CLASS_NUM, train_dir, test_dir, is_onehot):
    
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
        print("<-- Loading end...（total num = %d）" % len(labels_list))
        return file_paths_list, labels_list, class_names

    # 对path路径下的图片进行读取和预处理，当前针对分类问题，没有对标签进行处理，如果是检测问题，也可以对label进行处理
    # 由于当前下面函数中读取图片和图片增强都采用了tf的函数，造成如果想在该函数中基于图片的内容做出一些if判断，然后做出
    # 不同动作是不能实现的，当前只能通过tf.cond()等一系列tf函数来实现基于图片内容的if判断，例如在8朝向估计中，基于左
    # 右图片对调增强修改标签就是这样实现的，具体原因应该是tf2.6依然没有实现完全的eager模式运行，所以另外一种解决办法就
    # 是把图片读取和处理完全换成numpy语句，这样就可以用基于图片内容的if语句了，当前对于label标签可以随便用if语句
    def load_preprosess_image(path, label):
        # 读取图片文件得到字符串
        image = tf.io.read_file(path)
        # 解码字符串得到图片
        image = tf.io.decode_image(image, channels=3)  # 彩色图像为3个channel
        # 对图片尺寸进行等比例拉伸
        image = tf.image.resize_with_pad(image, IMG_SIZE[0], IMG_SIZE[1])
        # 随机左右翻转图像
        image = tf.image.random_flip_left_right(image)
        # 随机改变图像的亮度，随机改变[-1 ~ 1]大小的值 tf.image.random_brightness(image,max_delta,seed=None)为随机调整亮度函数，
        # 实际上是在原图的基础上随机加上一个值(如果加上的是正值则增亮否则增暗)，此值取自[-max_delta,max_delta)，要求max_delta>=0。
        image = tf.image.random_brightness(image, 1)
        # 改变数据类型
        image = tf.cast(image, tf.float32)
        # 对图片进行像素拉伸
        # image = image * (255 / (tf.reduce_max(image) - tf.reduce_min(image)))
        # 对图像进行归一化，归一化的方法有很多，这个是其中一种，归一化到[-1, 1]
        # image = (image - np.array([65, 70, 69]))/(3*np.array([44, 43, 43]))
        image = image/127.5 - 1.0
        # 对标签进行处理得到one-hot形式标签
        if is_onehot:
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

    return train_dataset, test_dataset
