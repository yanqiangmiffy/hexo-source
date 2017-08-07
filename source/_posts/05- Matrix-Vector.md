

title: 通过递归的矩阵向量空间预测组合语义
date: 2017/08/06 16:31:25
---

<Excerpt in index | 首页摘要> 
关于论文的总结
<!-- more -->

[Semantic Compositionality Through Recursive Matrix-Vector Spaces](http://www.socher.org/index.php/Main/SemanticCompositionalityThroughRecursiveMatrix-VectorSpaces)

# 摘要
单字矢量空间模型已经在学习词汇信息方面非常成功。但是，它们无法捕捉到更长的短语的位置意义，这样就阻碍了它们对语言的深入理解。我们介绍一种递归神经网络（RNN）模型，该模型学习任意句法类型和长度的短语和句子的组合向量表示。我们的模型为解析树中的每个节点分配向量和矩阵：向量捕获组成部分的固有含义，而矩阵捕获它如何改变相邻单词或短语的含义。这种矩阵向量RNN可以学习命题逻辑的运算符和自然语言的含义。该模型在三个不同的实验中获得最显著的表现：预测副词形容词对的细粒度情感分布;对电影评论的情感标签进行分类，并使用他们之间的句法路径对名词之间的因果关系或主题信息进行分类。
# 简介
语义词向量空间是许多有用的自然语言应用的核心，例如搜索查询扩展（Jones et al。2006），信息检索的事实提取（Pas¸caet al。2006）和消歧的文本自动注释带有的维基百科链接（Ratinov et al。2011）等等（Turney和Pantel。2010）。在这些模型中，单词的含义被编码为从单词及其相邻单词的共现统计中计算出的向量。这些向量已经表明它们与人类对词相似性的判断有很好的相关性（Griffiths et al。2007）。
# 方法
![方法.png](http://upload-images.jianshu.io/upload_images/1531909-4e65add68e8839c0.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
## 二分法解析树
![二分法解析树.png](http://upload-images.jianshu.io/upload_images/1531909-2cabd92402d45add.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
>The song was composed by as famous Indian musician

## 递归矩阵向量模型
![递归矩阵向量模型.png](http://upload-images.jianshu.io/upload_images/1531909-b198ecca3d36e95f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
### 初始化
- 用预先训练的50维词向量初始化所有的单词向量
- 将矩阵初始化为X=I+ε，其中I�是实体矩阵
### 组合
![组合.png](http://upload-images.jianshu.io/upload_images/1531909-3e8bfbfc08f12ad6.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
## 训练
我们通过在每个父节点顶部添加一个softmax分类器来训练向量表示，以一种情感分类或一些关系分类
![softmax.png](http://upload-images.jianshu.io/upload_images/1531909-b414b509852fda04.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
>其中W label∈R K×n是权重矩阵。如果有K个标签，则d∈RK是K维多项式分布

我们将t（x）∈RK×1表示为节点x处的目标分布向量,t（x）具有0-1编码：t（x）处的条目为1，其余条目为0.后计算d（x）和t（x）之间的交叉熵误差。
![交叉熵.png](http://upload-images.jianshu.io/upload_images/1531909-978a597c1be47c4b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
并将目标函数定义为所有训练数据上的E（x）之和：
![QQ截图20170807151929.png](http://upload-images.jianshu.io/upload_images/1531909-dbbcadf4712bd6db.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
>其中θ=（W，W M，W label，L，L M）是我们应该学习的模型参数的集合。 λ是正则化参数的向量.L和L M分别是字矢量和字矩阵的集合。

## 语义关系分类
- 我们首先在解析树之间找到我们要分类的关系的两个单词之间的路径。
- 然后，我们选择路径的最高节点，并使用该节点的向量作为特征对关系进行分类。
- 最后，我们将MV-RNN模型应用于由两个单词所跨越的子树。

![语义关系分类.png](http://upload-images.jianshu.io/upload_images/1531909-1d0a4d2d89815601.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

# 结果
我们对以下数据集进行了实验：
- SemEval 2010 Task 8
有9个有序的关系（有两个方向）和一个无向的其他类，所以一共有19个类。 这些关系有：信息主题，因果关系，工具代理。 如果关系中的单词的顺序正确，则对将其计为正确。
![SemEval 2010 Task8.png](http://upload-images.jianshu.io/upload_images/1531909-55c1e5bd2ff97e51.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
    1. Accuracy (calculated for the above confusion matrix)= 2094/2717 =77.07％
    2. F1_score= 82.51％
    3. 我们还使用根据“SemEval 2007 Task 4”的代码要求修改的不同数据集来执行测试并使用以前的培训模型
    4. 该实验的F1得分为40.08％，忽略方向性。

## 与其他办法的对比
![对比.png](http://upload-images.jianshu.io/upload_images/1531909-9a1608415535eec8.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
结果的改善也是由于其他方法的一些常见缺点。 例如：
•许多方法用无序的单词列表来表示文本，而情绪不仅取决于单词的含义，而且还取决于它们的顺序。
•使用的功能是手动开发的，不一定会捕获该单词的所有功能。

# 结论
- 我们的模型建立在语法上合理的解析树上，可以处理组合现象。
- 我们的模型的主要新颖性是矩阵向量表示与递归神经网络的组合。
- 它可以学习一个单词的意义向量，以及该单词如何修改其邻居（通过其矩阵）。
- MV-RNN将有吸引力的理论性能与大型噪声数据集的良好性能相结合。