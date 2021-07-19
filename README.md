知乎文章链接：[FocalLoss详解：从目标检测到FocalLoss，你看过的解释和代码可能一直是错的 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/391186824)
本仓库有将讲解用的notebook，在pytorch环境下可直接运行（不需要GPU）

文章正文：

Focal Loss 很多人应该都挺熟的，用来解决分类问题中 样本不均衡 + 难易样本的问题。但是网上很多关于FocalLoss的理解都是错误的，并且给了**错误的代码实现**。这两天正好再看检测相关论文，也解决了一致困扰着我的问题，所以写篇文章详细解读一下FocalLoss

先贴一个论文公式 + 普遍的错误代码：

![FL(p_t)=-\alpha_t(1-p_t)^\gamma log(p_t)](https://www.zhihu.com/equation?tex=FL(p_t)%3D-%5Calpha_t(1-p_t)%5E%5Cgamma%20log(p_t))FL(p_t)=-\alpha_t(1-p_t)^\gamma log(p_t) 

```
def forward(self, logits, labels):
    log_p = F.log_softmax(logits)
    pt = label_onehot * log_p
    sub_pt = 1 - pt
    fl = -self.alpha * (sub_pt)**self.gamma * log_p
    return fl.mean()
```

乍一看上去和公式完全一样，这里也引出一开始一直困扰我的问题：α没有起到作用。原本α应该是用来平衡类间样本数的，现在直接乘到损失前面，相当于对损失乘了一个常数。那么作用就变成平衡FocalLoss和正则化损失（如果有），或者调整学习率（如果没有正则化）。这个常数跟类间一点关系没有。

网上最常出现的错误就是把从二分类到多分类。搜索”FocalLoss实现“等关键词，会发现很多先给一个二分类案例，再引出所谓多分类的代码。

那么什么是正确的？为了更好说明，我下面会详解从目标检测到FocalLoss，最后会给出正确的版本以及一些理解。

## 一、目标检测中的分类任务与常规的分类任务的区别

FocalLoss的出现，主要是为了解决 anchor-based (one-stage) **目标检测网络的分类**问题。

注意，这里是目标检测网络的分类问题，而不是分类问题，这两者是不一样的。**区别在于，对于分配问题，一个图片一定是属于某一确定的类的；而检测任务中的分类，是有大量的anchor无目标的（可以称为负样本）。**

那么问题来了，负样本的标签是什么呢？

正常的K类分类任务的标签，是用一个K长度的向量作为标签，用one-hot（或者+smooth，这里先不考虑）来进行编码，最终的标签是一个形如[0,..., 1, ..., 0]这样的。那么如果想要检测背景，自然可以想到增加一个1维，如果目标检测任务有K类，这里只要用K+1维来表示分类，其中1维代表无目标即可。

但是实际任务中不是这么设计的（我没具体看过再之前的论文，不太了解原因）。

它用了另一个方案。

我们知道分类任务中，最后一般使用softmax来归一，使得所有类别的输出加和为1。但是检测任务中，对于无目标的anchor，我们并不希望最终结果加和为1，而是所有的概率输出都是0。

那么可以这样，我们将一个多分类任务看做多个二分类任务，针对每一个类别，我输出一个概率，如果接近0则代表非该类别，如果接近1，则代表这个anchor是该类别。

所以网络输出不需要用softmax来归一，而是对K长度向量的每一个分量进行sigmoid激活，让其输出值代表二分类的概率。对于无目标的anchor，gt中所有的分量都是0，代表属于每一类的概率是0。

**总结一下，FocalLoss解决的问题不是多分类问题，而是多个二分类问题。**

## **二、FocalLoss解决了什么**

简单说，大家都知道FocalLoss解决了样本不均+难易样本问题。

两者其实在解决一个问题：easy negative

现在假设我们一张图有100k个anchors（论文里说的大概会有这个数量anchors），对于猫（某一类别）来讲，可能只有100个anchor是属于猫的。

我们前面说过，这个分类器不是一个多类分类器，而是多个二分类器。

所以对于猫来说，相当于正负样本比例为1:1000，其中大部分负样本是无目标anchors，可能会有少部分是其他类别，但主要是那些只有背景的anchor。

所谓的正负样本不均衡，其实就是对于这个二分类任务来讲。

**很多解释FocalLoss的老是说，先拿二分类交叉熵来举例，再引出多分类交叉熵，再结合FocalLoss。这实际上是不对的，自始至终FocalLoss只用了二分类的交叉熵。**

> We introduce the focal loss starting from the cross entropy (CE) loss for binary classiﬁcation （脚注1）. 脚注1： Extending the focal loss to the multi-class case is straightforward and works well; for simplicity **we focus on the binary loss in this work**.

在目标检测的分类任务中，它只用了binary loss，虽然它是一个多分类任务。在文章最后，我也会给出一个正确的根据我的理解写的常规分类任务中，多分类任务+FocalLoss的思路。

## 三、FocalLoss怎么解决的

公式： ![FL(p_t)=-\alpha_t(1-p_t)^\gamma log(p_t)](https://www.zhihu.com/equation?tex=FL(p_t)%3D-%5Calpha_t(1-p_t)%5E%5Cgamma%20log(p_t))FL(p_t)=-\alpha_t(1-p_t)^\gamma log(p_t) 

这里我们看一下官方的代码，代码一般比论文清楚

fvcore/focal_loss.py at master · facebookresearch/fvcore (github.com)github.com

Facebook团队（也是focalloss作者）开源的，focalloss的代码是在detectron库里，但是代码中的loss是直接import fvcore这个库中的代码，所以我这里直接贴出了focalloss的源头。

重要的部分我贴一下（我删减了对α的判断，默认α存在）：

```
p = torch.sigmoid(inputs)
ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
p_t = p * targets + (1 - p) * (1 - targets)
loss = ce_loss * ((1 - p_t) ** gamma)
alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
loss = alpha_t * loss
```

这里我们结合我写的测试代码一起来看：

![img](https://pic1.zhimg.com/v2-a535f13cbd4adafc1eed03c06fdff0b0_b.png)

先随便初始化一下，inputs就是网络输出的结果，对于K分类，就有K维；targets是标签，我们假设正确分类是第二类。

### 第一步，先分别用sigmoid来获得每一个二分类的概率

![img](https://pic3.zhimg.com/v2-6a8007d3d64db9edf1332c62b6a2ef1a_b.png)

很明显，这一步求得的概率和并不为1。每一个分量代表anchor属于该类的概率，直觉上应该是求和小于1（加上无目标的概率=1），但是这里我们不作这个约束。

### 第二步，计算**二分类交叉熵**

![img](https://pic2.zhimg.com/v2-2ec480f62b01868a25bba848bace04c1_b.png)

这里很重要，二分类交叉熵是与多分类交叉熵不一样的

我们知道如果把这个任务直接看成多分类的话，交叉熵损失实际上是**正确的概率的负对数**，即 ![-log(p_t), y_t=1](https://www.zhihu.com/equation?tex=-log(p_t)%2C%20y_t%3D1)-log(p_t), y_t=1 。并且交叉熵的结果是1个值，而非一个长度仍为3的向量。

那么这个长度为3的向量是怎么计算出来的呢？实际上是分别进行了3次二分类交叉熵的运算，也就是 ![BCE=-ylog(p)-(1-y)log(1-p)](https://www.zhihu.com/equation?tex=BCE%3D-ylog(p)-(1-y)log(1-p))BCE=-ylog(p)-(1-y)log(1-p) 

以第一维来看，y=0, p=0.1192（p为sigmoid后的概率，y是target的第一维），所以对于这个二分类来看， ![BCE=-(1-y)log(1-p)](https://www.zhihu.com/equation?tex=BCE%3D-(1-y)log(1-p))BCE=-(1-y)log(1-p) ；第三维同理

第二维y=1，所以 ![BCE=-ylog(p)](https://www.zhihu.com/equation?tex=BCE%3D-ylog(p))BCE=-ylog(p) 

### 第三步，计算p_t，也就是模型分类正确的概率

![img](https://pic3.zhimg.com/v2-074724c46ed40936702baec8bf490152_b.png)

对于第一维，target=0，p=0.1269，所以模型分类正确的概率是1-p=0.8808。以此类推

### 第四步，计算损失

![img](https://pic3.zhimg.com/v2-f62bc631e809fc3215702d62c151be76_b.png)

按照公式，上一步的 ![p_t](https://www.zhihu.com/equation?tex=p_t)p_t 是模型分对的概率，所以为了降低简单样本的权重，分对的概率越高， ![1-p_t](https://www.zhihu.com/equation?tex=1-p_t)1-p_t 就越小。

注意一下，公式里的 ![p_t](https://www.zhihu.com/equation?tex=p_t)p_t 一直都是分类对的概率，而不是正样本概率。在二分类中，这些公式应该都是两项求和，只不过分类错误的那项会乘0，所以在公式中会被忽略。最终的形式是只有分类正确的概率。

### 第五步，乘 weighting factor

![img](https://pic2.zhimg.com/v2-9088cf963328c173d1669253ce0c7a35_b.png)

结合这部分，我想再重申一下正负样本问题

根据target，我们可以看出这个anchor是有类别的，属于第二类。但是在目标检测中，由于我们把任务拆成多个二分类任务，所以对于其他类别来讲，这个anchor也是负样本。不是只有背景（无目标）的anchor是负样本，而是对于每个类别来讲，非这个类别的anchor都是负样本。

所以可以看出三个α是不一样的，在取值0.75下，只有第二类是0.75，另外两个都是0.25

## 四、网上的FocalLoss错在哪？

对于概念我上文已经说过，网上很多都是以二分类为例，而实际上FocalLoss就是用于二分类的。只不过用多个二分类任务来实现多分类的目标检测。

那么对于源码，其实有一个非常大且简单的bug，我上文也提到了。

我这里再贴一个错误源码：

```
log_p = F.log_softmax(logits)
pt = label_onehot * log_p
sub_pt = 1 - pt
fl = -self.alpha * (sub_pt)**self.gamma * log_p
return fl.mean()
```

注意看这个alpha，这一步相当于对模型整体乘以一个常数，我们知道这实际上是没有什么意义的（在平衡样本层面）。这个alpha仅能用于平衡正则损失或者影响学习率。

那么对于多分类任务应该怎么处理呢

根据FocalLoss的原理，我认为有三种方案：

- 还是按照目标检测中的分类器一样，将任务设置为多个二分类任务
- 对每个类别设置不同的alpha
- 放弃alpha
