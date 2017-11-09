# -*- coding: utf-8 -*-

from skimage import io,transform
import glob
import os
import tensorflow as tf
import numpy as np
import time

w=256
h=256
c=3
#-----network------
x= tf.placeholder(tf.float32, shape=[None, w, h, c], name='x')
y_ = tf.placeholder(tf.int32, shape=[None, ], name='y_')
#conv0 256->128
conv0 = tf.nn.conv2d(
    input=x,
    filter=tf.Variable(tf.truncated_normal([5,5,3,12],stddev=0.01)),
    strides=[1,1,1,1],
    padding="SAME"
    )
#进行relu激励
re_conv0=tf.nn.relu(conv0)
#进行池化(64-->32)
conv0_pool=tf.nn.max_pool(
    value=re_conv0,
    ksize=[1,2,2,1],
    strides=[1,2,2,1],
    padding='SAME'
    )
# 第一个卷积层（128——>64)
conv1 = tf.nn.conv2d(
    input=conv0_pool,
    filter=tf.Variable(tf.truncated_normal([5,5,12,10],stddev=0.01)),
    strides=[1,2,2,1],
    padding="SAME"
    )
#进行relu激励
re_conv1=tf.nn.relu(conv1)
#进行池化(64-->32)
conv1_pool=tf.nn.max_pool(
    value=re_conv1,
    ksize=[1,2,2,1],
    strides=[1,2,2,1],
    padding='SAME'
    )
# 第二个卷积层（32——>16)
conv2 = tf.nn.conv2d(
    input=conv1_pool,
    filter=tf.Variable(tf.truncated_normal([3,3,10,8],stddev=0.01)),
    strides=[1,2,2,1],
    padding="SAME"
    )
#进行relu激励
re_conv2=tf.nn.relu(conv2)
#进行池化
# conv2_pool=tf.nn.max_pool(
#     value=re_conv2,
#     ksize=[1,2,2,1],
#     strides=[1,2,2,1],
#     padding='SAME'
#     )

# 第三个卷积层（16——>8)
conv3 = tf.nn.conv2d(
    input=re_conv2,
    filter=tf.Variable(tf.truncated_normal([5,5,8,6],stddev=0.01)),
    strides=[1,2,2,1],
    padding="SAME"
    )
#进行relu激励
re_conv3=tf.nn.relu(conv3)
#进行池化
conv3_pool=tf.nn.max_pool(
    value=re_conv3,
    ksize=[1,2,2,1],
    strides=[1,1,1,1],
    padding='SAME'
    )

#全连接层
conv_link = tf.nn.conv2d(
    input=conv3_pool,
    filter=tf.Variable(tf.truncated_normal([8,8,6,2],stddev=0.01)),
    strides=[1,1,1,1],
    padding="VALID"
    )
conv_link_out=conv_link[:,0,0,:]
#定义lost函数
loss=tf.losses.sparse_softmax_cross_entropy(y_,conv_link_out)
#定义optimizer
train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
#--------网络结束-------

#-------网络数值输出-----
correct_prediction = tf.equal(tf.cast(tf.argmax(conv_link_out, 1), tf.int32), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#--------数据读入-------
path='/media/ziv/OS/flower_photos'
cate = [path +'/'+ m for m in os.listdir(path) if os.path.isdir(path + '/'+m)]
imgfile=[]
label=[]
for i in range(len(cate)):
    print i
    for im in glob.glob(cate[i]+'/*.jpg'):
        imgfile.append(im)
        label.append(i)
num_example=len(label)
arr = np.arange(num_example)
np.random.shuffle(arr)
label=np.asarray(label, np.int32)
imgfile=np.asarray(imgfile)
label= label[arr]
imgfile=imgfile[arr]
#np.asarray(imgfile)
# serial= np.arange(n_data)
# for i in serial:
#     print i
# np.random.shuffle(serial)
# for i in serial:
#     print i
# #
# label=label[serial]
# label = label[arr]
# # training_data, validation_data, test_data=mnist_loader.load_data()
#将数据分为训练集和验证集
ratio=0.5
s=np.int(num_example*ratio)
x_train=imgfile[:s]
y_train=label[:s]
x_val=imgfile[s:]
y_val=label[s:]

def minibatches(inputs=None,labels=None,batch_size=None,shuffle=True):
    assert len(inputs)==len(labels)
    if shuffle:
        indice=np.arange(len(inputs))
        np.random.shuffle(indice)
    for sta_idx in range(0,len(inputs)-batch_size+1,batch_size):
        if shuffle:
            excerpt=indice[sta_idx:sta_idx+batch_size]
        else:
            excerpt=slice(sta_idx,sta_idx+batch_size)
        yield inputs[excerpt],labels[excerpt]
#---------feed data---------
n_epoch=100
batch_size=60

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer()) #数据初始化

for epoch in range(n_epoch):
    print  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),'epoch:',epoch
    train_loss, train_acc, n_batch = 0, 0, 0
    for x_train_path, y_train_a in minibatches(x_train, y_train, batch_size, shuffle=True):
        x_train_a = []
        for img in x_train_path:
            x_train_a.append(transform.resize(io.imread(img), (w, h)))
        _, err, ac = sess.run([train_op, loss, acc], feed_dict={x: x_train_a, y_: y_train_a})
        train_loss += err
        train_acc += ac
        n_batch += 1
    print("   train loss: %f" % (train_loss / n_batch))
    print("   train acc: %f" % (train_acc / n_batch))

    # validation
    val_loss, val_acc, n_batch = 0, 0, 0
    for x_val_path, y_val_a in minibatches(x_val, y_val, batch_size, shuffle=True):
        x_val_a = []
        for img in x_val_path:
            x_val_a.append(transform.resize(io.imread(img), (w, h)))
        err, ac = sess.run([loss, acc], feed_dict={x: x_val_a, y_: y_val_a})
        val_loss += err;
        val_acc += ac;
        n_batch += 1
    print("   validation loss: %f" % (val_loss / n_batch))
    print("   validation acc: %f" % (val_acc / n_batch))

sess.close()