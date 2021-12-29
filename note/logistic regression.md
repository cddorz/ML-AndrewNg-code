---
title: machine-learning（二）
date: 2021-12-29 22:47:39
tags: logistic regression
categories: AI
top_img: https://img0.baidu.com/it/u=728045523,3008166041&fm=26&fmt=auto
cover: https://img0.baidu.com/it/u=728045523,3008166041&fm=26&fmt=auto

---



## 一、逻辑回归（logistic regression）

#### 1.1 假说表示

根据线性回归模型我们只能预测连续的值，然而对于分类问题，我们需要输出0或1，我们可以预测：

当${h_\theta}\left( x \right)>=0.5$时，预测 $y=1$。

当${h_\theta}\left( x \right)<0.5$时，预测 $y=0$ 。

我们引入一个新的模型，逻辑回归，该模型的输出变量范围始终在0和1之间。

逻辑回归模型的假设是： $h_\theta \left( x \right)=g\left(\theta^{T}X \right)$

其中：

$X$ 代表特征向量

$g$ 代表逻辑函数（**logistic function**)是一个常用的逻辑函数为**S**形函数（**Sigmoid function**），公式为： $g\left( z \right)=\frac{1}{1+{{e}^{-z}}}$。

```python
import numpy as np
def sigmod(z):
    return 1 / (1 + np.exp(-z))
```

函数图像为：

![sigmod function](https://s2.loli.net/2021/12/29/oIvTPd5AyBONuKz.jpg)

$h_\theta \left( x \right)$的作用是，对于给定的输入变量，根据选择的参数计算输出变量=1的可能性（**estimated probablity**）即$h_\theta \left( x \right)=P\left( y=1|x;\theta \right)$

例如，如果对于给定的$x$，通过已经确定的参数计算得出$h_\theta \left( x \right)=0.7$，则表示有70%的几率$y$为正向类，相应地$y$为负向类的几率为1-0.7=0.3。

### 1.2 判定边界（decision boundary）

在逻辑回归中，我们预测：

当${h_\theta}\left( x \right)>=0.5$时，预测 $y=1$。

当${h_\theta}\left( x \right)<0.5$时，预测 $y=0$ 。

根据上面绘制出的 **S** 形函数图像，我们知道当

$z=0$ 时 $g(z)=0.5$

$z>0$ 时 $g(z)>0.5$

$z<0$ 时 $g(z)<0.5$

又 $z={\theta^{T}}x$ ，即：

${\theta^{T}}x>=0$  时，预测 $y=1$

${\theta^{T}}x<0$  时，预测 $y=0$

现在假设我们有一个模型：

![模型1](https://s2.loli.net/2021/12/29/v6rI3mhX5eqKQj2.png)

并且参数$\theta$ 是向量[-3 1 1]。 则当$-3+{x_1}+{x_2} \geq 0$，即${x_1}+{x_2} \geq 3$时，模型将预测 $y=1$。

我们可以绘制直线${x_1}+{x_2} = 3$，这条线便是我们模型的分界线，将预测为1的区域和预测为 0的区域分隔开。

![判定边界.jpg](https://s2.loli.net/2021/12/29/ZgQa7plCsOIR69h.jpg)

> 紫色线即是判定边界

### 1.3 代价函数

$J\left( \theta  \right)=\frac{1}{m}\sum\limits_{i=1}^{m}{[-{{y}^{(i)}}\log \left( {h_\theta}\left( {{x}^{(i)}} \right) \right)-\left( 1-{{y}^{(i)}} \right)\log \left( 1-{h_\theta}\left( {{x}^{(i)}} \right) \right)]}$

即：$J\left( \theta  \right)=-\frac{1}{m}\sum\limits_{i=1}^{m}{[{{y}^{(i)}}\log \left( {h_\theta}\left( {{x}^{(i)}} \right) \right)+\left( 1-{{y}^{(i)}} \right)\log \left( 1-{h_\theta}\left( {{x}^{(i)}} \right) \right)]}$

```python
import numpy as np
def cost(theta, X, y):
    theta = np.matrix(theta)
  	X = np.matrix(X)
  	y = np.matrix(y)
    first_term = np.multiply(-y, np.log(sigmod(X @ theta.T)))
    second_term = np.multiply((1 - y), np.log(1 - sigmod(X @ theta.T)))
    return np.sum(first_term - second_term) / len(X)
```

在得到这样一个代价函数以后，我们便可以用梯度下降算法来求得能使代价函数最小的参数了。算法为：

**Repeat** {

$\theta_j := \theta_j - \alpha \frac{\partial}{\partial\theta_j} J(\theta)$

(**simultaneously update all** )

}

求导后得到：

**Repeat** {

$\theta_j := \theta_j - \alpha \frac{1}{m}\sum\limits_{i=1}^{m}{{\left( {h_\theta}\left( \mathop{x}^{\left( i \right)} \right)-\mathop{y}^{\left( i \right)} \right)}}\mathop{x}_{j}^{(i)}$ 

**(simultaneously update all** )

}

现在，如果你把这个更新规则和我们之前用在线性回归上的进行比较的话，你会惊讶地发现，这个式子正是我们用来做线性回归梯度下降的。

那么，线性回归和逻辑回归是同一个算法吗？要回答这个问题，我们要观察逻辑回归看看发生了哪些变化。实际上，假设的定义发生了变化。

对于线性回归假设函数：

${h_\theta}\left( x \right)={\theta^T}X={\theta_{0}}{x_{0}}+{\theta_{1}}{x_{1}}+{\theta_{2}}{x_{2}}+...+{\theta_{n}}{x_{n}}$

而现在逻辑函数假设函数：

${h_\theta}\left( x \right)=\frac{1}{1+{{e}^{-{\theta^T}X}}}$

因此，即使更新参数的规则看起来基本相同，但由于假设的定义发生了变化，所以逻辑函数的梯度下降，跟线性回归的梯度下降实际上是两个完全不同的东西。