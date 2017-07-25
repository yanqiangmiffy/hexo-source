---
title: Keras实现简单的手写数字识别
---
<Excerpt in index | 首页摘要> 
Keras实现简单的手写数字识别：构建模型、编译模型、训练数据、输出
<!-- more -->

[参考](http://www.cnblogs.com/yqtm/p/6924939.html)
文中代码有点小bug,加以改正。顺带才了下数据集的坑

## 导入需要的函数和包


```python
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
from keras.optimizers import SGD
from keras.datasets import mnist
import numpy as np
```


```python
#从s3.amazonaws.com/img-datasets/mnist.npz下载数据太慢了。挂了代理，结果程序运行崩溃，只好写一个加载本地的文件函数
def load_data(path='mnist.npz'):
    f=np.load(path)
    x_train,y_train=f['x_train'],f['y_train']
    x_test,y_test=f['x_test'],f['y_test']
    f.close()
    return (x_train,y_train),(x_test,y_test)
```

Sequential是序贯模型，Dense是用于添加模型的层数，SGD是用于模型变异的时候优化器参数,
mnist是用于加载手写识别的数据集，需要在网上下载,下面是mnist.py

```
from ..utils.data_utils import get_file
import numpy as np


def load_data(path='mnist.npz'):
    """Loads the MNIST dataset.

    # Arguments
        path: path where to cache the dataset locally
            (relative to ~/.keras/datasets).

    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    path = get_file(path, origin='https://s3.amazonaws.com/img-datasets/mnist.npz')
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)
    
```

## 构建模型


```python
model=Sequential()
model.add(Dense(500,input_shape=(784,)))#输入层
model.add(Activation('tanh'))
model.add(Dropout(0.5))

model.add(Dense(500))#隐藏层
model.add(Activation('tanh'))
model.add(Dropout(0.5))

model.add(Dense(10))
model.add(Activation('softmax'))
```

1. Dense()设定该层的结构，第一个参数表示输出的个数，第二个参数是接受的输入数据的格式。第一层中需要指定输入的格式，在之后的增加的层中输入层节点数默认是上一层的输出个数
2. Activation()指定预定义激活函数：softmax，elu、softplus、softsign、relu、、sigmoid、hard_sigmoid、linear<br>
3. Dropout()用于指定每层丢掉的信息百分比。

## 编译模型


```python
sgd=SGD(lr=0.01,decay=1e-6,momentum=0.9,nesterov=True)#设定学习效率等参数
model.compile(loss='categorical_crossentropy',optimizer=sgd)
#model.compile(loss = 'categorical_crossentropy', optimizer=sgd, class_mode='categorical') #使用交叉熵作为loss
```

调用model.compile()之前初始化一个优化器对象，然后传入该函数,使用优化器sgd来编译模型，用来指定学习效率等参数。编译时指定loss函数，这里使用交叉熵函数作为loss函数。

*SGD*

```
keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
```
随机梯度下降法，支持动量参数，支持学习衰减率，支持Nesterov动量

参数

- `lr`：大于0的浮点数，学习率
- `momentum`：大于0的浮点数，动量参数
- `decay`：大于0的浮点数，每次更新后的学习率衰减值
- `nesterov`：布尔值，确定是否使用Nesterov动量

## 读取训练集和测试集


```python
(x_train,y_train),(x_test,y_test)=load_data()#直接加载本地文件
#(x_train,y_train),(x_test,y_test)=mnist.load_data()#不使用mnist提供的load_data函数，
X_train=x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2])
X_test=x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[2])
Y_train=(np.arange(10)==y_train[:,None]).astype(int)#将index转换成一个one_hot矩阵
Y_test=(np.arange(10)==y_test[:,None]).astype(int)
```


```python
print(x_train.shape)
print(x_train)
print(x_test.shape)
print("y_train:",y_train,len(y_train))
print(y_train[:None])
print(y_train[:,None]==np.arange(10))
print(np.arange(10))
```

    (60000, 28, 28)
    [[[0 0 0 ..., 0 0 0]
      [0 0 0 ..., 0 0 0]
      [0 0 0 ..., 0 0 0]
      ..., 
      [0 0 0 ..., 0 0 0]
      [0 0 0 ..., 0 0 0]
      [0 0 0 ..., 0 0 0]]
    
     [[0 0 0 ..., 0 0 0]
      [0 0 0 ..., 0 0 0]
      [0 0 0 ..., 0 0 0]
      ..., 
      [0 0 0 ..., 0 0 0]
      [0 0 0 ..., 0 0 0]
      [0 0 0 ..., 0 0 0]]
    
     [[0 0 0 ..., 0 0 0]
      [0 0 0 ..., 0 0 0]
      [0 0 0 ..., 0 0 0]
      ..., 
      [0 0 0 ..., 0 0 0]
      [0 0 0 ..., 0 0 0]
      [0 0 0 ..., 0 0 0]]
    
     ..., 
     [[0 0 0 ..., 0 0 0]
      [0 0 0 ..., 0 0 0]
      [0 0 0 ..., 0 0 0]
      ..., 
      [0 0 0 ..., 0 0 0]
      [0 0 0 ..., 0 0 0]
      [0 0 0 ..., 0 0 0]]
    
     [[0 0 0 ..., 0 0 0]
      [0 0 0 ..., 0 0 0]
      [0 0 0 ..., 0 0 0]
      ..., 
      [0 0 0 ..., 0 0 0]
      [0 0 0 ..., 0 0 0]
      [0 0 0 ..., 0 0 0]]
    
     [[0 0 0 ..., 0 0 0]
      [0 0 0 ..., 0 0 0]
      [0 0 0 ..., 0 0 0]
      ..., 
      [0 0 0 ..., 0 0 0]
      [0 0 0 ..., 0 0 0]
      [0 0 0 ..., 0 0 0]]]
    (10000, 28, 28)
    y_train: [5 0 4 ..., 5 6 8] 60000
    [5 0 4 ..., 5 6 8]
    [[False False False ..., False False False]
     [ True False False ..., False False False]
     [False False False ..., False False False]
     ..., 
     [False False False ..., False False False]
     [False False False ..., False False False]
     [False False False ..., False  True False]]
    [0 1 2 3 4 5 6 7 8 9]
    

1. 读取minst数据集，通过reshape()函数转换数据的格式。
2. 如果我们打印x_train.shape会发现它是(60000,28,28)，即一共60000个数据，每个数据是28*28的图片。通过reshape转换为(60000,784)的线性张量。
3. 如果我们打印y_train会发现它是一组表示每张图片的表示数字的数组，通过numpy的arange()和astype()函数将每个数字转换为一组长度为10的张量，代表的数字的位置是1，其它位置为0.

## 训练模型


```python
model.fit(X_train,Y_train,batch_size=200,epochs=100,shuffle=True,verbose=1,validation_split=0.3)
```

    Train on 42000 samples, validate on 18000 samples
    Epoch 1/100
    42000/42000 [==============================] - 5s - loss: 1.2457 - val_loss: 0.5666
    Epoch 2/100
    42000/42000 [==============================] - 4s - loss: 0.9481 - val_loss: 0.4958
    Epoch 3/100
    42000/42000 [==============================] - 4s - loss: 0.8623 - val_loss: 0.4659
    Epoch 4/100
    42000/42000 [==============================] - 4s - loss: 0.8145 - val_loss: 0.4691
    Epoch 5/100
    42000/42000 [==============================] - 4s - loss: 0.7788 - val_loss: 0.4342
    Epoch 6/100
    42000/42000 [==============================] - 4s - loss: 0.7225 - val_loss: 0.4105
    Epoch 7/100
    42000/42000 [==============================] - 4s - loss: 0.7338 - val_loss: 0.3970
    Epoch 8/100
    42000/42000 [==============================] - 4s - loss: 0.6848 - val_loss: 0.3961
    Epoch 9/100
    42000/42000 [==============================] - 4s - loss: 0.6693 - val_loss: 0.3875
    Epoch 10/100
    42000/42000 [==============================] - 4s - loss: 0.6544 - val_loss: 0.3751
    Epoch 11/100
    42000/42000 [==============================] - 4s - loss: 0.6276 - val_loss: 0.3681
    Epoch 12/100
    42000/42000 [==============================] - 4s - loss: 0.6605 - val_loss: 0.3660
    Epoch 13/100
    42000/42000 [==============================] - 4s - loss: 0.6487 - val_loss: 0.3515
    Epoch 14/100
    42000/42000 [==============================] - 4s - loss: 0.6426 - val_loss: 0.3646
    Epoch 15/100
    42000/42000 [==============================] - 4s - loss: 0.6292 - val_loss: 0.3424
    Epoch 16/100
    42000/42000 [==============================] - 4s - loss: 0.6074 - val_loss: 0.3378
    Epoch 17/100
    42000/42000 [==============================] - 4s - loss: 0.5844 - val_loss: 0.3320
    Epoch 18/100
    42000/42000 [==============================] - 4s - loss: 0.5753 - val_loss: 0.3363
    Epoch 19/100
    42000/42000 [==============================] - 4s - loss: 0.5570 - val_loss: 0.3199
    Epoch 20/100
    42000/42000 [==============================] - 4s - loss: 0.5452 - val_loss: 0.3108
    Epoch 21/100
    42000/42000 [==============================] - 4s - loss: 0.5320 - val_loss: 0.3108
    Epoch 22/100
    42000/42000 [==============================] - 4s - loss: 0.5354 - val_loss: 0.3024
    Epoch 23/100
    42000/42000 [==============================] - 4s - loss: 0.5172 - val_loss: 0.2973
    Epoch 24/100
    42000/42000 [==============================] - 4s - loss: 0.5222 - val_loss: 0.3037
    Epoch 25/100
    42000/42000 [==============================] - 4s - loss: 0.5208 - val_loss: 0.2940
    Epoch 26/100
    42000/42000 [==============================] - 4s - loss: 0.5154 - val_loss: 0.2948
    Epoch 27/100
    42000/42000 [==============================] - 4s - loss: 0.5258 - val_loss: 0.2918
    Epoch 28/100
    42000/42000 [==============================] - 4s - loss: 0.5033 - val_loss: 0.2889
    Epoch 29/100
    42000/42000 [==============================] - 4s - loss: 0.4962 - val_loss: 0.2828
    Epoch 30/100
    42000/42000 [==============================] - 4s - loss: 0.4848 - val_loss: 0.2761
    Epoch 31/100
    42000/42000 [==============================] - 4s - loss: 0.4884 - val_loss: 0.2881
    Epoch 32/100
    42000/42000 [==============================] - 4s - loss: 0.4873 - val_loss: 0.2794
    Epoch 33/100
    42000/42000 [==============================] - 4s - loss: 0.4823 - val_loss: 0.2686
    Epoch 34/100
    42000/42000 [==============================] - 4s - loss: 0.4781 - val_loss: 0.2788
    Epoch 35/100
    42000/42000 [==============================] - 4s - loss: 0.4781 - val_loss: 0.2732
    Epoch 36/100
    42000/42000 [==============================] - 4s - loss: 0.4786 - val_loss: 0.2880
    Epoch 37/100
    42000/42000 [==============================] - 4s - loss: 0.4829 - val_loss: 0.2729
    Epoch 38/100
    42000/42000 [==============================] - 4s - loss: 0.4744 - val_loss: 0.2731
    Epoch 39/100
    42000/42000 [==============================] - 4s - loss: 0.4564 - val_loss: 0.2698
    Epoch 40/100
    42000/42000 [==============================] - 4s - loss: 0.4614 - val_loss: 0.2629
    Epoch 41/100
    42000/42000 [==============================] - 4s - loss: 0.4673 - val_loss: 0.2586
    Epoch 42/100
    42000/42000 [==============================] - 4s - loss: 0.4666 - val_loss: 0.2524
    Epoch 43/100
    42000/42000 [==============================] - 4s - loss: 0.4545 - val_loss: 0.2682
    Epoch 44/100
    42000/42000 [==============================] - 4s - loss: 0.4550 - val_loss: 0.2653
    Epoch 45/100
    42000/42000 [==============================] - 4s - loss: 0.4426 - val_loss: 0.2537
    Epoch 46/100
    42000/42000 [==============================] - 4s - loss: 0.4322 - val_loss: 0.2523
    Epoch 47/100
    42000/42000 [==============================] - 4s - loss: 0.4541 - val_loss: 0.2552
    Epoch 48/100
    42000/42000 [==============================] - 4s - loss: 0.4465 - val_loss: 0.2493
    Epoch 49/100
    42000/42000 [==============================] - 4s - loss: 0.4366 - val_loss: 0.2445
    Epoch 50/100
    42000/42000 [==============================] - 4s - loss: 0.4362 - val_loss: 0.2458
    Epoch 51/100
    42000/42000 [==============================] - 4s - loss: 0.4388 - val_loss: 0.2446
    Epoch 52/100
    42000/42000 [==============================] - 4s - loss: 0.4440 - val_loss: 0.2551
    Epoch 53/100
    42000/42000 [==============================] - 4s - loss: 0.4278 - val_loss: 0.2469
    Epoch 54/100
    42000/42000 [==============================] - 4s - loss: 0.4185 - val_loss: 0.2416
    Epoch 55/100
    42000/42000 [==============================] - 4s - loss: 0.4086 - val_loss: 0.2332
    Epoch 56/100
    42000/42000 [==============================] - 4s - loss: 0.4005 - val_loss: 0.2407
    Epoch 57/100
    42000/42000 [==============================] - 4s - loss: 0.4064 - val_loss: 0.2396
    Epoch 58/100
    42000/42000 [==============================] - 4s - loss: 0.4063 - val_loss: 0.2384
    Epoch 59/100
    42000/42000 [==============================] - 4s - loss: 0.4020 - val_loss: 0.2358
    Epoch 60/100
    42000/42000 [==============================] - 4s - loss: 0.4008 - val_loss: 0.2332
    Epoch 61/100
    42000/42000 [==============================] - 4s - loss: 0.4045 - val_loss: 0.2338
    Epoch 62/100
    42000/42000 [==============================] - 4s - loss: 0.4153 - val_loss: 0.2346
    Epoch 63/100
    42000/42000 [==============================] - 4s - loss: 0.4102 - val_loss: 0.2279
    Epoch 64/100
    42000/42000 [==============================] - 4s - loss: 0.4013 - val_loss: 0.2337
    Epoch 65/100
    42000/42000 [==============================] - 4s - loss: 0.3945 - val_loss: 0.2312
    Epoch 66/100
    42000/42000 [==============================] - 4s - loss: 0.3917 - val_loss: 0.2243
    Epoch 67/100
    42000/42000 [==============================] - 4s - loss: 0.3780 - val_loss: 0.2219
    Epoch 68/100
    42000/42000 [==============================] - 4s - loss: 0.3781 - val_loss: 0.2249
    Epoch 69/100
    42000/42000 [==============================] - 4s - loss: 0.3755 - val_loss: 0.2192
    Epoch 70/100
    42000/42000 [==============================] - 4s - loss: 0.3814 - val_loss: 0.2164
    Epoch 71/100
    42000/42000 [==============================] - 4s - loss: 0.3843 - val_loss: 0.2197
    Epoch 72/100
    42000/42000 [==============================] - 4s - loss: 0.3835 - val_loss: 0.2228
    Epoch 73/100
    42000/42000 [==============================] - 4s - loss: 0.3908 - val_loss: 0.2281
    Epoch 74/100
    42000/42000 [==============================] - 4s - loss: 0.3881 - val_loss: 0.2185
    Epoch 75/100
    42000/42000 [==============================] - 4s - loss: 0.3870 - val_loss: 0.2108
    Epoch 76/100
    42000/42000 [==============================] - 4s - loss: 0.3731 - val_loss: 0.2112
    Epoch 77/100
    42000/42000 [==============================] - 4s - loss: 0.3685 - val_loss: 0.2069
    Epoch 78/100
    42000/42000 [==============================] - 4s - loss: 0.3633 - val_loss: 0.2059
    Epoch 79/100
    42000/42000 [==============================] - 4s - loss: 0.3626 - val_loss: 0.2073
    Epoch 80/100
    42000/42000 [==============================] - 4s - loss: 0.3594 - val_loss: 0.2053
    Epoch 81/100
    42000/42000 [==============================] - 4s - loss: 0.3489 - val_loss: 0.2001
    Epoch 82/100
    42000/42000 [==============================] - 4s - loss: 0.3521 - val_loss: 0.2007
    Epoch 83/100
    42000/42000 [==============================] - 4s - loss: 0.3488 - val_loss: 0.2029
    Epoch 84/100
    42000/42000 [==============================] - 4s - loss: 0.3531 - val_loss: 0.1984
    Epoch 85/100
    42000/42000 [==============================] - 4s - loss: 0.3545 - val_loss: 0.2034
    Epoch 86/100
    42000/42000 [==============================] - 4s - loss: 0.3559 - val_loss: 0.2053
    Epoch 87/100
    42000/42000 [==============================] - 4s - loss: 0.3551 - val_loss: 0.2019
    Epoch 88/100
    42000/42000 [==============================] - 4s - loss: 0.3538 - val_loss: 0.2043
    Epoch 89/100
    42000/42000 [==============================] - 4s - loss: 0.3498 - val_loss: 0.2050
    Epoch 90/100
    42000/42000 [==============================] - 4s - loss: 0.3566 - val_loss: 0.2076
    Epoch 91/100
    42000/42000 [==============================] - 4s - loss: 0.3573 - val_loss: 0.2052
    Epoch 92/100
    42000/42000 [==============================] - 4s - loss: 0.3633 - val_loss: 0.1994
    Epoch 93/100
    42000/42000 [==============================] - 4s - loss: 0.3561 - val_loss: 0.2004
    Epoch 94/100
    42000/42000 [==============================] - 4s - loss: 0.3473 - val_loss: 0.2015
    Epoch 95/100
    42000/42000 [==============================] - 4s - loss: 0.3463 - val_loss: 0.1951
    Epoch 96/100
    42000/42000 [==============================] - 4s - loss: 0.3485 - val_loss: 0.1985
    Epoch 97/100
    42000/42000 [==============================] - 4s - loss: 0.3357 - val_loss: 0.1994
    Epoch 98/100
    42000/42000 [==============================] - 4s - loss: 0.3399 - val_loss: 0.1965
    Epoch 99/100
    42000/42000 [==============================] - 4s - loss: 0.3408 - val_loss: 0.1931
    Epoch 100/100
    42000/42000 [==============================] - 4s - loss: 0.3366 - val_loss: 0.1956
    




    <keras.callbacks.History at 0x2a5fdb3d278>



- batch_size表示每个训练块包含的数据个数，
- epochs表示训练的次数，
- shuffle表示是否每次训练后将batch打乱重排，
- verbose表示是否输出进度log，
- validation_split指定验证集占比

## 输出测试结果


```python
print("test set")
scores = model.evaluate(X_test,Y_test,batch_size=200,verbose=1)
print("")
print("The test loss is %f" % scores)
result = model.predict(X_test,batch_size=200,verbose=1)

result_max = np.argmax(result, axis = 1)
test_max = np.argmax(Y_test, axis = 1)

result_bool = np.equal(result_max, test_max)
true_num = np.sum(result_bool)
print("")
print("The accuracy of the model is %f" % (true_num/len(result_bool)))
```

    test set
     8800/10000 [=========================>....] - ETA: 0s
    The test loss is 0.185958
    10000/10000 [==============================] - 0s     
    
    The accuracy of the model is 0.943400
    

- model.evaluate()计算了测试集中的识别的loss值。
- 通过model.predict()，我们可以得到对于测试集中每个数字的识别结果，每个数字对应一个表示每个数字都是多少概率的长度为10的张量。
- 通过np.argmax()，我们得到每个数字的识别结果和期望的识别结果

- 通过np.equal()，我们得到每个数字是否识别正确

- 通过np.sum()得到识别正确的总的数字个数
