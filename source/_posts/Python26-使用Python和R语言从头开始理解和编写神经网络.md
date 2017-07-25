title: 使用Python和R语言从头开始理解和编写神经网络
date: 2017/7/24 14:46:25
---
<Excerpt in index | 首页摘要> 
本篇文章是[原文](https://www.analyticsvidhya.com/blog/2017/05/neural-network-from-scratch-in-python-and-r/)的翻译过来的，自己在学习和阅读之后觉得文章非常不错，文章结构清晰，由浅入深、从理论到代码实现，最终将神经网络的概念和工作流程呈现出来。
<!-- more -->

本篇文章是[原文](https://www.analyticsvidhya.com/blog/2017/05/neural-network-from-scratch-in-python-and-r/)的翻译过来的，自己在学习和阅读之后觉得文章非常不错，文章结构清晰，由浅入深、从理论到代码实现，最终将神经网络的概念和工作流程呈现出来。自己将其翻译成中文，以便以后阅读和复习和网友参考。因时间（文字纯手打加配图）紧促和翻译水平有限，文章有不足之处请大家指正。
# 介绍

你可以通过两种方式学习和实践一个概念：

- 选项1：您可以了解一个特定主题的整个理论，然后寻找应用这些概念的方法。所以，你阅读整个算法的工作原理，背后的数学知识、假设理论、局限，然后去应用它。这样学习稳健但是需要花费大量的时间去准备。
- 选项2：从简单的基础开始，并就此主题研究直觉上的知识。接下来，选择一个问题并开始解决它。在解决问题的同时了解这些概念，保持调整并改善您对此问题的理解。所以，你去了解如何应用一个算法——实践并应用它。一旦你知道如何应用它，请尝试使用不同的参数和测试值，极限值去测试算法和继续优化对算法的理解。

我更喜欢选项2，并采取这种方法来学习任何新的话题。我可能无法告诉你算法背后的整个数学，但我可以告诉你直觉上的知识以及基于实验和理解来应用算法的最佳场景。

在与其他人交流的过程中，我发现人们不用花时间来发展这种直觉，所以他们能够以正确的方式努力地去解决问题。

在本文中，我将从头开始讨论一个神经网络的构建，更多地关注研究这种直觉上的知识来实现神经网络。我们将在“Python”和“R”中编写代码。读完本篇文章后，您将了解神经网络如何工作，如何初始化权重，以及如何使用反向传播进行更新。

让我们开始吧

# 目录

- 神经网络背后的简单直觉知识
- 多层感知器及其基础知识
- 涉及神经网络方法的步骤
- 可视化神经网络工作方法的步骤
- 使用Numpy（Python）实现NN
- 使用R实现NN
- [可选]反向传播算法的数学观点

# 神经网络背后的直观知识

如果您是开发人员或了解一种工作——知道如何在代码中调试错误。您可以通过改变输入或条件来触发各种测试用例，并查找输出，输出的变化提供了一个提示：在代码中，去哪里寻找bug？ - 哪个模块要检查，哪些行要阅读。找到bug后，您进行更改并继续运行，直到您能够运行正确的代码或者实现应用程序。

神经网络的工作方式非常相似。它需要多个输入，通过来自多个隐藏层的多个神经元进行处理，并使用输出层返回结果。这个结果估计过程在技术上被称为“前向传播”。

接下来，我们将结果与实际输出进行比较。任务是使神经网络的输出接近实际（期望的）输出。在这些神经元中，每一个都会对最终输出产生一些误差，你如何减少这些误差呢？

我们尝试最小化那些对错误“贡献”更多的神经元的值和权重，并且在返回到神经网络的神经元并发现误差在哪里时发生。这个过程被称为“向后传播”。

为了减少迭代次数来实现最小化误差，神经网络通常使用称为“梯度下降”的算法，来快速有效地优化任务。

的确 ，这就是神经网络如何工作的！我知道这是一个非常简单的表示，但它可以帮助您以简单的方式理解事物。

# 多层感知器及其基础知识

就像原子是形成地球上任何物质的基础 - 神经网络的基本形成单位是感知器。 那么，什么是感知器呢？

感知器可以被理解为需要多个输入并产生一个输出的任何东西。 例如，看下面的图片
![感知器](https://i.loli.net/2017/07/24/59756c063bbec.png "感知器")
上述结构需要三个输入并产生一个输出，下一个逻辑问题是输入和输出之间的关系是什么？让我们从基本的方式着手，寻求更复杂的方法。

下面我讨论了三种创建输入输出关系的方法：

1. 通过直接组合输入和计算基于阈值的输出。例如：取x1 = 0，x2 = 1，x3 = 1并设置阈值= 0。因此，如果`x1 + x2 + x3> 0`，则输出为1，否则为0.可以看出，在这种情况下，感知器会将输出计算为1。
2. 接下来，让我们为输入添加权重。权重重视输入。例如，您分别为x1，x2和x3分配w1 = 2，w2 = 3和w3 = 4。为了计算输出，我们将输入与相应权重相乘，并将其与阈值进行比较，如w1 * x1 + w2 * x2 + w3 * x3>阈值。与x1和x2相比，这些权重对于x3显得更重要。
3. 最后，让我们添加偏置量：每个感知器也有一个偏置量，可以被认为是感知器多么灵活。它与某种线性函数y = ax + b的常数b类似，它允许我们上下移动线以适应数据更好的预测。假设没有b，线将始终通过原点（0，0），并且可能会得到较差的拟合。例如，感知器可以具有两个输入，在这种情况下，它需要三个权重。每个输入一个，偏置一个。现在输入的线性表示将如下所示：w1 * x1 + w2 * x2 + w3 * x3 + 1 * b。

但是，上面所讲的感知器之间的关系都是线性的，并没有那么有趣。所以，人们认为将感知器演化成现在所谓的人造神经元，对于输入和偏差，神经元将使用非线性变换（激活函数）。

# 什么是激活函数？

激活函数将加权输入`（w1 * x1 + w2 * x2 + w3 * x3 + 1 * b）`的和作为参数，并返回神经元的输出。
![激活函数](https://i.loli.net/2017/07/24/59757bc8a16d3.png "激活函数")

在上式中，我们用x0表示1，w0表示b。

激活函数主要用于进行非线性变换，使我们能够拟合非线性假设或估计复杂函数。 有多种激活功能，如：`“Sigmoid”`，`“Tanh”`，`ReLu`等等。

# 前向传播，反向传播和训练次数(epochs)

到目前为止，我们已经计算了输出，这个过程被称为“正向传播”。 但是如果估计的输出远离实际输出（非常大的误差）怎么办？ 下面正是我们在神经网络中所做的：基于错误更新偏差和权重。 这种权重和偏差更新过程被称为“反向传播”。

反向传播（BP）算法通过确定输出处的损耗（或误差），然后将其传播回网络来工作， 更新权重以最小化每个神经元产生的错误。 最小化误差的第一步是确定每个节点w.r.t.的梯度（Derivatives），最终实现输出。 要获得反向传播的数学视角，请参阅下面的部分。

这一轮的前向和后向传播迭代被称为一个训练迭代也称为“Epoch”。`ps:e（一）poch（波）的意思;一个epoch是指把所有训练数据完整的过一遍`

# 多层感知器

现在，我们来看看多层感知器。 到目前为止，我们已经看到只有一个由3个输入节点组成的单层，即x1，x2和x3，以及由单个神经元组成的输出层。 但是，出于实际，单层网络只能做到这一点。 如下所示，MLP由层叠在输入层和输出层之间的许多隐层组成。
![多层感知器](https://i.loli.net/2017/07/24/59757fa6a6e02.png "多层感知器")
上面的图像只显示一个单一的隐藏层，但实际上可以包含多个隐藏层。 在MLP的情况下要记住的另一点是，所有层都完全连接，即层中的每个节点（输入和输出层除外）连接到上一层和下一层中的每个节点。让我们继续下一个主题，即神经网络的训练算法（最小化误差）。 在这里，我们将看到最常见的训练算法称为梯度下降。

# 全批量梯度下降和随机梯度下降

Gradient Descent的第二个变体通过使用相同的更新算法执行更新MLP的权重的相同工作，但差异在于用于更新权重和偏差的训练样本的数量。

全部批量梯度下降算法作为名称意味着使用所有的训练数据点来更新每个权重一次，而随机渐变使用1个或更多（样本），但从不使整个训练数据更新权重一次。

让我们用一个简单的例子来理解这个10个数据点的数据集，它们有两个权重w1和w2。

- 全批：您可以使用10个数据点（整个训练数据），并计算w1（Δw1）的变化和w2（Δw2）的变化，并更新w1和w2。

- SGD：使用第一个数据点并计算w1（Δw1）的变化，并改变w2（Δw2）并更新w1和w2。 接下来，当您使用第二个数据点时，您将处理更新的权重

# 神经网络方法的步骤

![多层感知器](https://i.loli.net/2017/07/24/59757fa6a6e02.png "多层感知器")
我们来看一步一步地构建神经网络的方法（MLP与一个隐藏层，类似于上图所示的架构）。 在输出层，我们只有一个神经元，因为我们正在解决二进制分类问题（预测0或1）。 我们也可以有两个神经元来预测两个类的每一个。

先看一下广泛的步骤：

1. 我们输入和输出

    - X作为输入矩阵
    - y作为输出矩阵
2. 我们用随机值初始化权重和偏差（这是一次启动，在下一次迭代中，我们将使用更新的权重和偏差）。 让我们定义：

    - wh作为权重矩阵隐藏层
    - bh作为隐藏层的偏置矩阵
    - wout作为输出层的权重矩阵
    - bout作为偏置矩阵作为输出层
3. 我们将输入和权重的矩阵点积分配给输入和隐藏层之间的边，然后将隐层神经元的偏差添加到相应的输入，这被称为线性变换：

      `hidden_layer_input= matrix_dot_product(X,wh) + bh`
4. 使用激活函数（Sigmoid）执行非线性变换。 Sigmoid将返回输出1/(1 + exp(-x)).
      `hiddenlayer_activations = sigmoid(hidden_layer_input)`
5. 对隐藏层激活进行线性变换（取矩阵点积，并加上输出层神经元的偏差），然后应用激活函数（再次使用Sigmoid，但是根据您的任务可以使用任何其他激活函数 ）来预测输出

      `output_layer_input = matrix_dot_product (hiddenlayer_activations * wout ) + bout`
      
      `output = sigmoid(output_layer_input)`
      
  <strong>所有上述步骤被称为“前向传播”（Forward Propagation）</strong>
6. 将预测与实际输出进行比较，并计算误差梯度（实际预测值）。 误差是均方损失= ((Y-t)^2)/2
      `E = y – output`
7. 计算隐藏和输出层神经元的斜率/斜率（为了计算斜率，我们计算每个神经元的每层的非线性激活x的导数）。 S形梯度可以返回 `x * (1 – x)`.

      `slope_output_layer = derivatives_sigmoid(output)`

      `slope_hidden_layer = derivatives_sigmoid(hiddenlayer_activations)`
8. 计算输出层的变化因子（delta），取决于误差梯度乘以输出层激活的斜率
      `d_output = E * slope_output_layer`
9. 在这一步，错误将传播回网络，这意味着隐藏层的错误。 为此，我们将采用输出层三角形的点积与隐藏层和输出层之间的边缘的重量参数（wout.T）。
      `Error_at_hidden_layer = matrix_dot_product(d_output, wout.Transpose)`
10. 计算隐层的变化因子（delta），将隐层的误差乘以隐藏层激活的斜率
      `d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer`
11. 在输出和隐藏层更新权重：网络中的权重可以从为训练示例计算的错误中更新。
       `wout = wout + matrix_dot_product(hiddenlayer_activations.Transpose, d_output)*learning_rate`
       
       `wh =  wh + matrix_dot_product(X.Transpose,d_hiddenlayer)*learning_rate`
learning_rate：权重更新的量由称为学习率的配置参数控制）
12. 在输出和隐藏层更新偏差：网络中的偏差可以从该神经元的聚合错误中更新。
     - bias at output_layer =bias at output_layer + sum of delta of output_layer at row-wise * learning_rate
     - bias at hidden_layer =bias at hidden_layer + sum of delta of output_layer at row-wise * learning_rate  
     
     `bh = bh + sum(d_hiddenlayer, axis=0) * learning_rate`
        
     `bout = bout + sum(d_output, axis=0)*learning_rate`
     
   <strong>从6到12的步骤被称为“向后传播”(Backward Propagation)</strong>

一个正向和反向传播迭代被认为是一个训练周期。 如前所述，我们什么时候训练第二次，然后更新权重和偏差用于正向传播。

以上，我们更新了隐藏和输出层的权重和偏差，我们使用了全批量梯度下降算法。

# 神经网络方法的可视化步骤

我们将重复上述步骤，可视化输入，权重，偏差，输出，误差矩阵，以了解神经网络（MLP）的工作方法。

- <strong>注意：</strong>

    - 对于良好的可视化图像，我有2或3个位置的十进制小数位。
    - 黄色填充的细胞代表当前活动细胞
    - 橙色单元格表示用于填充当前单元格值的输入


- 步骤1：读取输入和输出
![Step 1](https://i.loli.net/2017/07/24/597588fe182b4.png)
- 步骤2：用随机值初始化权重和偏差（有初始化权重和偏差的方法，但是现在用随机值初始化）
![Step 2](https://i.loli.net/2017/07/24/597589475b8bb.png)
- 步骤3：计算隐层输入： <br />
`hidden_layer_input= matrix_dot_product(X,wh) + bh`
![Step 3](https://i.loli.net/2017/07/24/597589a62c037.png)
- 步骤4：对隐藏的线性输入进行非线性变换 <br />
`hiddenlayer_activations = sigmoid(hidden_layer_input)`
![Step 4](https://i.loli.net/2017/07/24/59758a0111ed8.png)
- 步骤5：在输出层执行隐层激活的线性和非线性变换 <br />
`output_layer_input = matrix_dot_product (hiddenlayer_activations * wout ) + bout` <br />
`output = sigmoid(output_layer_input)`
![Step 5](https://i.loli.net/2017/07/24/59758a58893ba.png)

- 步骤6：计算输出层的误差（E）梯度 <br />
`E = y-output`
![Step 6](https://i.loli.net/2017/07/24/59758ad4a72ff.png)
- 步骤7：计算输出和隐藏层的斜率 <br />
`Slope_output_layer= derivatives_sigmoid(output)` <br />
`Slope_hidden_layer = derivatives_sigmoid(hiddenlayer_activations)`
![py26-10.png](https://i.loli.net/2017/07/24/59758b26893ef.png)
- 步骤8：计算输出层的增量 <br />
`d_output = E * slope_output_layer*lr`
![py26-11.png](https://i.loli.net/2017/07/24/59758b61227a9.png)
- 步骤9：计算隐藏层的误差 <br />
`Error_at_hidden_layer = matrix_dot_product(d_output, wout.Transpose)`
![py26-12.png](https://i.loli.net/2017/07/24/59758ba276123.png)
- 步骤10：计算隐藏层的增量 <br />
`d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer`
![py26-13.png](https://i.loli.net/2017/07/24/59758bd705865.png)
- 步骤11：更新输出和隐藏层的权重 <br />
`wout = wout + matrix_dot_product(hiddenlayer_activations.Transpose, d_output)*learning_rate` <br />
`wh =  wh+ matrix_dot_product(X.Transpose,d_hiddenlayer)*learning_rate`
![py26-14.png](https://i.loli.net/2017/07/24/59758c13cc478.png)
- 步骤12：更新输出和隐藏层的偏置量<br />
`bh = bh + sum(d_hiddenlayer, axis=0) * learning_rate`<br />
`bout = bout + sum(d_output, axis=0)*learning_rate`
![py26-15.png](https://i.loli.net/2017/07/24/59758c71210be.png)

以上，您可以看到仍然有一个很好的误差而不接近于实际目标值，因为我们已经完成了一次训练迭代。 如果我们多次训练模型，那么这将是一个非常接近的实际结果。 我完成了数千次迭代，我的结果接近实际的目标值（`[[0.98032096] [0.96845624] [0.04532167]]`）。

# 使用Numpy（Python）实现NN


```python
import numpy as np

#Input array
X=np.array([[1,0,1,0],[1,0,1,1],[0,1,0,1]])

#Output
y=np.array([[1],[1],[0]])

#Sigmoid Function
def sigmoid (x):
    return 1/(1 + np.exp(-x))

#Derivative of Sigmoid Function
def derivatives_sigmoid(x):
    return x * (1 - x)

#Variable initialization
epoch=5000 #Setting training iterations
lr=0.1 #Setting learning rate
inputlayer_neurons = X.shape[1] #number of features in data set
hiddenlayer_neurons = 3 #number of hidden layers neurons
output_neurons = 1 #number of neurons at output layer

#weight and bias initialization
wh=np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons))
bh=np.random.uniform(size=(1,hiddenlayer_neurons))
wout=np.random.uniform(size=(hiddenlayer_neurons,output_neurons))
bout=np.random.uniform(size=(1,output_neurons))

for i in range(epoch):
    #Forward Propogation
    hidden_layer_input1=np.dot(X,wh)
    hidden_layer_input=hidden_layer_input1 + bh
    hiddenlayer_activations = sigmoid(hidden_layer_input)
    output_layer_input1=np.dot(hiddenlayer_activations,wout)
    output_layer_input= output_layer_input1+ bout
    output = sigmoid(output_layer_input)
    #Backpropagation
    E = y-output
    slope_output_layer = derivatives_sigmoid(output)
    slope_hidden_layer = derivatives_sigmoid(hiddenlayer_activations)
    d_output = E * slope_output_layer
    Error_at_hidden_layer = d_output.dot(wout.T)
    d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer
    wout += hiddenlayer_activations.T.dot(d_output) *lr
    bout += np.sum(d_output, axis=0,keepdims=True) *lr
    wh += X.T.dot(d_hiddenlayer) *lr
    bh += np.sum(d_hiddenlayer, axis=0,keepdims=True) *lr

print("output of Forward Propogation:\n{}".format(output))
print("wout,bout of Backpropagation:\n{},\n{}".format(wout,bout))
```

    output of Forward Propogation:
    [[ 0.98497471]
     [ 0.96956956]
     [ 0.0416628 ]]
    wout,bout of Backpropagation:
    [[ 3.34342103]
     [-1.97924327]
     [ 3.90636787]],
    [[-1.71231223]]
    

# 在R中实现NN


```python
# input matrix
X=matrix(c(1,0,1,0,1,0,1,1,0,1,0,1),nrow = 3, ncol=4,byrow = TRUE)

# output matrix
Y=matrix(c(1,1,0),byrow=FALSE)

#sigmoid function
sigmoid<-function(x){
1/(1+exp(-x))
}

# derivative of sigmoid function
derivatives_sigmoid<-function(x){
x*(1-x)
}

# variable initialization
epoch=5000
lr=0.1
inputlayer_neurons=ncol(X)
hiddenlayer_neurons=3
output_neurons=1

#weight and bias initialization
wh=matrix( rnorm(inputlayer_neurons*hiddenlayer_neurons,mean=0,sd=1), inputlayer_neurons, hiddenlayer_neurons)
bias_in=runif(hiddenlayer_neurons)
bias_in_temp=rep(bias_in, nrow(X))
bh=matrix(bias_in_temp, nrow = nrow(X), byrow = FALSE)
wout=matrix( rnorm(hiddenlayer_neurons*output_neurons,mean=0,sd=1), hiddenlayer_neurons, output_neurons)

bias_out=runif(output_neurons)
bias_out_temp=rep(bias_out,nrow(X))
bout=matrix(bias_out_temp,nrow = nrow(X),byrow = FALSE)
# forward propagation
for(i in 1:epoch){

hidden_layer_input1= X%*%wh
hidden_layer_input=hidden_layer_input1+bh
hidden_layer_activations=sigmoid(hidden_layer_input)
output_layer_input1=hidden_layer_activations%*%wout
output_layer_input=output_layer_input1+bout
output= sigmoid(output_layer_input)

# Back Propagation

E=Y-output
slope_output_layer=derivatives_sigmoid(output)
slope_hidden_layer=derivatives_sigmoid(hidden_layer_activations)
d_output=E*slope_output_layer
Error_at_hidden_layer=d_output%*%t(wout)
d_hiddenlayer=Error_at_hidden_layer*slope_hidden_layer
wout= wout + (t(hidden_layer_activations)%*%d_output)*lr
bout= bout+rowSums(d_output)*lr
wh = wh +(t(X)%*%d_hiddenlayer)*lr
bh = bh + rowSums(d_hiddenlayer)*lr

}
output
```

# [可选]反向传播算法的数学理解

设Wi为输入层和隐层之间的权重。 Wh是隐层和输出层之间的权重。

现在，`h =σ（u）=σ（WiX）`，即h是u的函数，u是Wi和X的函数。这里我们将我们的函数表示为σ

`Y =σ（u'）=σ（Whh）`，即Y是u'的函数，u'是Wh和h的函数。

我们将不断参考上述方程来计算偏导数。

我们主要感兴趣的是找到两个项：∂E/∂Wi和∂E/∂Wh即改变输入和隐藏层之间权重的误差变化，改变隐层和输出之间权重的变化 层。

但是为了计算这两个偏导数，我们将需要使用部分微分的链规则，因为E是Y的函数，Y是u'的函数，u'是Wi的函数。

让我们把这个属性很好的用于计算梯度。

`∂E/∂Wh = (∂E/∂Y).( ∂Y/∂u’).( ∂u’/∂Wh), ……..(1)

We know E is of the form E=(Y-t)2/2.

So, (∂E/∂Y)= (Y-t)`

现在，σ是一个S形函数，并具有σ（1-σ）形式的有意义的区分。 我敦促读者在他们身边进行验证。
所以, `(∂Y/∂u’)= ∂( σ(u’)/ ∂u’= σ(u’)(1- σ(u’))`.

但是, `σ(u’)=Y, So`,

`(∂Y/∂u’)=Y(1-Y)`

现在得出, `( ∂u’/∂Wh)= ∂( Whh)/ ∂Wh = h`

取代等式（1）中的值我们得到，

`∂E/∂Wh = (Y-t). Y(1-Y).h`

所以，现在我们已经计算了隐层和输出层之间的梯度。 现在是计算输入层和隐藏层之间的梯度的时候了。
`∂E/∂Wi =(∂ E/∂ h). (∂h/∂u).( ∂u/∂Wi)`
但是，` (∂ E/∂ h) = (∂E/∂Y).( ∂Y/∂u’).( ∂u’/∂h)`. 在上述方程中替换这个值得到：

`∂E/∂Wi =[(∂E/∂Y).( ∂Y/∂u’).( ∂u’/∂h)]. (∂h/∂u).( ∂u/∂Wi)……………(2)`
那么，首先计算隐层和输出层之间的梯度有什么好处？

如等式（2）所示，我们已经计算出∂E/∂Y和∂Y/∂u'节省了空间和计算时间。 我们会在一段时间内知道为什么这个算法称为反向传播算法。

让我们计算公式（2）中的未知导数。

`∂u’/∂h = ∂(Whh)/ ∂h = Wh`

`∂h/∂u = ∂( σ(u)/ ∂u= σ(u)(1- σ(u))`

但是, `σ(u)=h, So,`

`(∂Y/∂u)=h(1-h)`

得出, `∂u/∂Wi = ∂(WiX)/ ∂Wi = X`

取代等式（2）中的所有这些值，我们得到：</br>
`∂E/∂Wi = [(Y-t). Y(1-Y).Wh].h(1-h).X`

所以现在，由于我们已经计算了两个梯度，所以权重可以更新为:

`Wh = Wh + η . ∂E/∂Wh`

 `Wi = Wi + η . ∂E/∂Wi`

其中η是学习率。

所以回到这个问题：为什么这个算法叫做反向传播算法？

原因是：如果您注意到∂E/∂Wh和∂E/∂Wi的最终形式，您将看到术语（Yt）即输出错误，这是我们开始的，然后将其传播回输入 层重量更新。

那么，这个数学在哪里适合代码？

`hiddenlayer_activations= H`

`E = Y-t`

`Slope_output_layer = Y（1-Y）`

`lr =η`

`slope_hidden_layer = h（1-h）`

`wout = Wh`

现在，您可以轻松地将代码与数学联系起来。

# 结束语

本文主要从头开始构建神经网络，并了解其基本概念。 我希望你现在可以理解神经网络的工作，如前向和后向传播的工作，优化算法（全批次和随机梯度下降），如何更新权重和偏差，Excel中每个步骤的可视化以及建立在python和R的代码.

因此，在即将到来的文章中，我将解释在Python中使用神经网络的应用，并解决与以下问题相关的现实生活中的挑战：

1. 计算机视觉
2. 言语
3. 自然语言处理

我在写这篇文章的时候感到很愉快，并希望从你的反馈中学习。 你觉得这篇文章有用吗？ 感谢您的建议/意见。 请随时通过以下意见提出您的问题。


```python
（转载请注明来源）
```
