
# coding: utf-8

# DenseNet on CIFAR-10

# 如果训练下载的Kaggle的完整数据集，把下面改False
demo = False
if demo:
    import zipfile
    for fin in ['train_tiny.zip', 'test_tiny.zip', 'trainLabels.csv.zip']:
        with zipfile.ZipFile('../data/kaggle_cifar10/' + fin, 'r') as zin:
            zin.extractall('../data/kaggle_cifar10/')




import os
import shutil

def reorg_cifar10_data(data_dir, label_file, train_dir, test_dir, input_dir, valid_ratio):
    # 读取训练数据标签。
    with open(os.path.join(data_dir, label_file), 'r') as f:
        # 跳过文件头行（栏名称）。
        lines = f.readlines()[1:]
        tokens = [l.rstrip().split(',') for l in lines]
        idx_label = dict(((int(idx), label) for idx, label in tokens))
    labels = set(idx_label.values())

    num_train = len(os.listdir(os.path.join(data_dir, train_dir)))
    num_train_tuning = int(num_train * (1 - valid_ratio))
    assert 0 < num_train_tuning < num_train
    num_train_tuning_per_label = num_train_tuning // len(labels)
    label_count = dict()

    def mkdir_if_not_exist(path):
        if not os.path.exists(os.path.join(*path)):
            os.makedirs(os.path.join(*path))

    # 整理训练和验证集。
    for train_file in os.listdir(os.path.join(data_dir, train_dir)):
        idx = int(train_file.split('.')[0])
        label = idx_label[idx]
        mkdir_if_not_exist([data_dir, input_dir, 'train_valid', label])
        shutil.copy(os.path.join(data_dir, train_dir, train_file),
                    os.path.join(data_dir, input_dir, 'train_valid', label))
        if label not in label_count or label_count[label] < num_train_tuning_per_label:
            mkdir_if_not_exist([data_dir, input_dir, 'train', label])
            shutil.copy(os.path.join(data_dir, train_dir, train_file),
                        os.path.join(data_dir, input_dir, 'train', label))
            label_count[label] = label_count.get(label, 0) + 1
        else:
            mkdir_if_not_exist([data_dir, input_dir, 'valid', label])
            shutil.copy(os.path.join(data_dir, train_dir, train_file),
                        os.path.join(data_dir, input_dir, 'valid', label))

    # 整理测试集。
    mkdir_if_not_exist([data_dir, input_dir, 'test', 'unknown'])
    for test_file in os.listdir(os.path.join(data_dir, test_dir)):
        shutil.copy(os.path.join(data_dir, test_dir, test_file),
                    os.path.join(data_dir, input_dir, 'test', 'unknown'))




if demo:
    # 注意：此处使用小训练集为便于网页编译。Kaggle的完整数据集应包括5万训练样本。
    train_dir = 'train_tiny'
    # 注意：此处使用小测试集为便于网页编译。Kaggle的完整数据集应包括30万测试样本。
    test_dir = 'test_tiny'
    # 注意：此处相应使用小批量。对Kaggle的完整数据集可设较大的整数，例如128。
    batch_size = 1
else:
    train_dir = 'train'
    test_dir = 'test'
    batch_size = 128

data_dir = '../data/kaggle_cifar10'
label_file = 'trainLabels.csv'
input_dir = 'train_valid_test'
valid_ratio = 0.1
reorg_cifar10_data(data_dir, label_file, train_dir, test_dir, input_dir, valid_ratio)





from mxnet import autograd
from mxnet import gluon
from mxnet import image
from mxnet import init
from mxnet import nd
from mxnet.gluon.data import vision
import numpy as np

def transform_train(data, label):
    im = data.astype('float32') / 255
    auglist = image.CreateAugmenter(data_shape=(3, 32, 32), resize=0, 
                        rand_crop=False, rand_resize=False, rand_mirror=True,
                        mean=np.array([0.4914, 0.4822, 0.4465]), 
                        std=np.array([0.2023, 0.1994, 0.2010]), 
                        brightness=0, contrast=0, 
                        saturation=0, hue=0, 
                        pca_noise=0, rand_gray=0, inter_method=2)
    for aug in auglist:
        im = aug(im)
    # 将数据格式从"高*宽*通道"改为"通道*高*宽"。
    im = nd.transpose(im, (2,0,1))
    return (im, nd.array([label]).asscalar().astype('float32'))

# 测试时，无需对图像做标准化以外的增强数据处理。
def transform_test(data, label):
    im = data.astype('float32') / 255
    auglist = image.CreateAugmenter(data_shape=(3, 32, 32), 
                        mean=np.array([0.4914, 0.4822, 0.4465]), 
                        std=np.array([0.2023, 0.1994, 0.2010]))
    for aug in auglist:
        im = aug(im)
    im = nd.transpose(im, (2,0,1))
    return (im, nd.array([label]).asscalar().astype('float32'))





input_str = data_dir + '/' + input_dir + '/'

# 读取原始图像文件。flag=1说明输入图像有三个通道（彩色）。
train_ds = vision.ImageFolderDataset(input_str + 'train', flag=1, 
                                     transform=transform_train)
valid_ds = vision.ImageFolderDataset(input_str + 'valid', flag=1, 
                                     transform=transform_test)
train_valid_ds = vision.ImageFolderDataset(input_str + 'train_valid', 
                                           flag=1, transform=transform_train)
test_ds = vision.ImageFolderDataset(input_str + 'test', flag=1, 
                                     transform=transform_test)

loader = gluon.data.DataLoader
train_data = loader(train_ds, batch_size, shuffle=True, last_batch='keep')
valid_data = loader(valid_ds, batch_size, shuffle=True, last_batch='keep')
train_valid_data = loader(train_valid_ds, batch_size, shuffle=True, last_batch='keep')
test_data = loader(test_ds, batch_size, shuffle=False, last_batch='keep')

# 交叉熵损失函数。
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()





from mxnet.gluon import nn
from mxnet import nd

def conv_block(channels):
    out = nn.Sequential()
    out.add(
        nn.BatchNorm(),
        nn.Activation('relu'),
        nn.Conv2D(channels, kernel_size=3, padding=1)
    )
    return out
        
class DenseBlock(nn.Block):
    def __init__(self, layers, growth_rate, **kwargs):
        super(DenseBlock, self).__init__(**kwargs)
        self.net = nn.Sequential()
        for i in range(layers):
            self.net.add(conv_block(growth_rate))
        
    def forward(self, x):
        for layer in self.net:
            out = layer(x)
            x = nd.concat(x, out, dim=1)
        return x
    
def transition_block(channels):
    out = nn.Sequential()
    out.add(
        nn.BatchNorm(),
        nn.Activation('relu'),
        nn.Conv2D(channels, kernel_size=1),
        nn.AvgPool2D(pool_size=2, strides=2)
    )
    return out

init_channels = 32
growth_rate = 32
block_layers = [6, 12, 24, 16]
num_classes = 10

def dense_net():
    net = nn.Sequential()
    with net.name_scope():
        # first block
        net.add(
            nn.Conv2D(init_channels, kernel_size=3, 
                      strides=1, padding=1),
            nn.BatchNorm(),
            nn.Activation('relu'),
            nn.MaxPool2D(pool_size=3, strides=2, padding=1)
        )
        # dense blocks
        channels = init_channels
        for i, layers in enumerate(block_layers):
            net.add(DenseBlock(layers, growth_rate))
            channels += layers * growth_rate
            if i != len(block_layers)-1:
                net.add(transition_block(channels//2))
        # last block
        net.add(
            nn.BatchNorm(),
            nn.Activation('relu'),
            nn.AvgPool2D(pool_size=1),
            nn.Flatten(),
            nn.Dense(num_classes)
        )    
    return net

def get_net(ctx):
    num_outputs = 10
    net = dense_net()
    net.initialize(ctx=ctx, init=init.Xavier())
    return net





import datetime
import sys
sys.path.append('..')
import utils

def train(net, train_data, valid_data, num_epochs, lr, wd, ctx, lr_period, lr_decay):
    trainer = gluon.Trainer(
        net.collect_params(), 'sgd', {'learning_rate': lr, 'momentum': 0.9, 'wd': wd})

    prev_time = datetime.datetime.now()
    for epoch in range(num_epochs):
        train_loss = 0.0
        train_acc = 0.0
        if epoch > 0 and epoch % lr_period == 0:
            trainer.set_learning_rate(trainer.learning_rate * lr_decay)
        for data, label in train_data:
            label = label.as_in_context(ctx)
            with autograd.record():
                output = net(data.as_in_context(ctx))
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            trainer.step(batch_size)
            train_loss += nd.mean(loss).asscalar()
            train_acc += utils.accuracy(output, label)
        cur_time = datetime.datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        if valid_data is not None:
            valid_acc = utils.evaluate_accuracy(valid_data, net, ctx)
            epoch_str = ("Epoch %d. Loss: %f, Train acc %f, Valid acc %f, "
                         % (epoch, train_loss / len(train_data),
                            train_acc / len(train_data), valid_acc))
        else:
            epoch_str = ("Epoch %d. Loss: %f, Train acc %f, "
                         % (epoch, train_loss / len(train_data),
                            train_acc / len(train_data)))
        prev_time = cur_time
        print(epoch_str + time_str + ', lr ' + str(trainer.learning_rate))





ctx = utils.try_gpu()
num_epochs = 50
learning_rate = 0.1
weight_decay = 5e-4
lr_period = 40
lr_decay = 0.1

net = get_net(ctx)





import numpy as np
import pandas as pd

net = get_net(ctx)
train(net, train_valid_data, None, num_epochs, learning_rate, 
      weight_decay, ctx, lr_period, lr_decay)

preds = []
for data, label in test_data:
    output = net(data.as_in_context(ctx))
    preds.extend(output.argmax(axis=1).astype(int).asnumpy())

sorted_ids = list(range(1, len(test_ds) + 1))
sorted_ids.sort(key = lambda x:str(x))

df = pd.DataFrame({'id': sorted_ids, 'label': preds})
df['label'] = df['label'].apply(lambda x: train_valid_ds.synsets[x])
df.to_csv('submission_dense.csv', index=False)

print("Done. :)")

