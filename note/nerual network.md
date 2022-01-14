## 神经网络

### 1.1模型表示

神经网络模型建立在很多神经元之上，每一个神经元又是一个个学习模型。这些神经元（也叫激活单元，**activation unit**）采纳一些特征作为输出，并且根据本身的模型提供一个输出。下图是一个以逻辑回归模型作为自身学习模型的神经元示例，在神经网络中，参数又可被成为权重（**weight**）。

![示例](https://s2.loli.net/2022/01/14/AMe5Swam6h9kWf1.jpg)

> x_0 是偏置单元，一般每一层都会加入一个偏置单元（也称作偏置项）

（以下解释参见：https://www.zhihu.com/question/305340182/answer/721739423）

在神经网络中，以[sigmoid函数](https://www.zhihu.com/search?q=sigmoid函数&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A721739423})为例，加入偏置后也是增加了函数的灵活性，提高了神经元的拟合能力。

![img](https://pic2.zhimg.com/50/v2-56eb6c184c6c51bee461a714a0f2e7d9_720w.jpg?source=1940ef5c)

在神经元中，output  =  sum (weights * inputs) + bias。**偏置实际上是对神经元激活状态的控制**。比如在上图中，当偏置为20时，x较小时y的值就可以很大，就可以较快的将该神经元激活。

以此设计出类似神经元的神经网络：

![模型.jpg](https://s2.loli.net/2022/01/14/XyOij9No5qgG7RY.jpg)

神经网络模型是许多逻辑单元按照不同层级组织起来的网络，每一层的输出变量都是下一层的输入变量。下图为一个3层的神经网络，第一层成为输入层（**Input Layer**），最后一层称为输出层（**Output Layer**），中间一层成为隐藏层（**Hidden Layers**）。我们为每一层都增加一个偏差单位（**bias unit**）：

下面引入一些标记法来帮助描述模型：

$a_{i}^{\left( j \right)}$ 代表第$j$ 层的第 $i$ 个激活单元。${{\theta }^{\left( j \right)}}$代表从第 $j$ 层映射到第$ j+1$ 层时的权重的矩阵，例如${{\theta }^{\left( 1 \right)}}$代表从第一层映射到第二层的权重的矩阵。其尺寸为：以第 $j+1$层的激活单元数量为行数，以第 $j$ 层的激活单元数加一为列数的矩阵。例如：上图所示的神经网络中${{\theta }^{\left( 1 \right)}}$的尺寸为 3*4 --> （3，4） @ （4，1） = （3，1）

>前向传播算法：（神经网络从左向右）
>
>$a_{1}^{(2)}=g(\Theta _{10}^{(1)}{{x}_{0}}+\Theta _{11}^{(1)}{{x}_{1}}+\Theta _{12}^{(1)}{{x}_{2}}+\Theta _{13}^{(1)}{{x}_{3}})$
>
>$a_{2}^{(2)}=g(\Theta _{20}^{(1)}{{x}_{0}}+\Theta _{21}^{(1)}{{x}_{1}}+\Theta _{22}^{(1)}{{x}_{2}}+\Theta _{23}^{(1)}{{x}_{3}})$
>
>$a_{3}^{(2)}=g(\Theta _{30}^{(1)}{{x}_{0}}+\Theta _{31}^{(1)}{{x}_{1}}+\Theta _{32}^{(1)}{{x}_{2}}+\Theta _{33}^{(1)}{{x}_{3}})$
>
>${{h}_{\Theta }}(x)=g(\Theta _{10}^{(2)}a_{0}^{(2)}+\Theta _{11}^{(2)}a_{1}^{(2)}+\Theta _{12}^{(2)}a_{2}^{(2)}+\Theta _{13}^{(2)}a_{3}^{(2)})$
>
>矩阵表示：
>
>$\theta \cdot X=a$ 

![公式](https://s2.loli.net/2022/01/14/x85VCDQj3rpgILo.png)

令 ${{z}^{\left( 2 \right)}}={{\theta }^{\left( 1 \right)}}x$，则 ${{a}^{\left( 2 \right)}}=g({{z}^{\left( 2 \right)}})$ ，计算后添加 $a_{0}^{\left( 2 \right)}=1$

${{z}^{\left( 3 \right)}}={{\theta }^{\left( 2 \right)}}{{a}^{\left( 2 \right)}}$，则 $h_\theta(x)={{a}^{\left( 3 \right)}}=g({{z}^{\left( 3 \right)}})$

然后这只是针对训练集中的一行进行计算，如果针对整个训练集，则需要进行转置，即：

${{z}^{\left( 2 \right)}}={{\Theta }^{\left( 1 \right)}}\times {{X}^{T}}$

 ${{a}^{\left( 2 \right)}}=g({{z}^{\left( 2 \right)}})$

其实神经网络就像是**logistic regression**，只不过我们把**logistic regression**中的输入向量$\left[ x_1\sim {x_3} \right]$ 变成了中间层的$\left[ a_1^{(2)}\sim a_3^{(2)} \right]$, 即:  $h_\theta(x)=g\left( \Theta_0^{\left( 2 \right)}a_0^{\left( 2 \right)}+\Theta_1^{\left( 2 \right)}a_1^{\left( 2 \right)}+\Theta_{2}^{\left( 2 \right)}a_{2}^{\left( 2 \right)}+\Theta_{3}^{\left( 2 \right)}a_{3}^{\left( 2 \right)} \right)$ 

我们可以把$a_0, a_1, a_2, a_3$看成更为高级的特征值，也就是$x_0, x_1, x_2, x_3$的进化体，并且它们是由 $x$与$\theta$决定的，因为是梯度下降的，所以$a$是变化的，并且变得越来越厉害，所以这些更高级的特征值远比仅仅将 $x$次方厉害，也能更好的预测新数据。

这就是神经网络相比于逻辑回归和线性回归的优势。

### 1.2 代价函数

在逻辑回归中，我们只有一个输出变量，又称标量（**scalar**），也只有一个因变量$y$，但是在神经网络中，我们可以有很多输出变量，我们的$h_\theta(x)$是一个维度为$K$的向量，并且我们训练集中的因变量也是同样维度的一个向量，因此我们的代价函数会比逻辑回归更加复杂一些，为：$\newcommand{\subk}[1]{ #1_k }$

$h_\theta\left(x\right)\in \mathbb{R}^{K}$

 ${\left({h_\theta}\left(x\right)\right)}_{i}={i}^{th} \text{output}$

$J(\Theta) = -\frac{1}{m} \left[ \sum\limits_{i=1}^{m} \sum\limits_{k=1}^{k} {y_k}^{(i)} \log {(h_\Theta(x^{(i)}))_k} + \left( 1 - y_k^{(i)} \right) \log \left( 1- {\left( h_\Theta \left( x^{(i)} \right) \right)_k} \right) \right] + \frac{\lambda}{2m} \sum\limits_{l=1}^{L-1} \sum\limits_{i=1}^{s_l} \sum\limits_{j=1}^{s_{l+1}} \left( \Theta_{ji}^{(l)} \right)^2$

### 1.3 反向传播算法

之前我们在计算神经网络预测结果的时候我们采用了一种正向传播方法，我们从第一层开始正向一层一层进行计算，直到最后一层的$h_{\theta}\left(x\right)$。

现在，为了计算代价函数的偏导数$\frac{\partial}{\partial\Theta^{(l)}_{ij}}J\left(\Theta\right)$，我们需要采用一种反向传播算法，也就是首先计算最后一层的误差，然后再一层一层反向求出各层的误差，直到倒数第二层。

以一个例子来说明反向传播算法。

假设我们的训练集只有一个样本$\left({x}^{(1)},{y}^{(1)}\right)$，我们的神经网络是一个四层的神经网络，其中$K=4，S_{L}=4，L=4$，对每一层的输出单元（我们用$\delta$来表示误差）：

+ $\delta^{(4)}=a^{(4)}-y$

+ $\delta^{(3)}=\left({\Theta^{(3)}}\right)^{T}\delta^{(4)}\ast g'\left(z^{(3)}\right)$，其中$g'(z^{(3)})$是 $S$ 形函数的导数，$g'(z^{(3)})=a^{(3)}\ast(1-a^{(3)})$。而$(θ^{(3)})^{T}\delta^{(4)}$则是权重导致的误差的和。

+ $\delta^{(2)}=(\Theta^{(2)})^{T}\delta^{(3)}\ast g'(z^{(2)})$

+ 第一层是输入，无误差

+ 最后：假设$λ=0$，即我们不做任何正则化处理时有：

  $\frac{\partial}{\partial\Theta_{ij}^{(l)}}J(\Theta)=a_{j}^{(l)} \delta_{i}^{l+1}$

  > $l$ 代表目前所计算的是第几层。
  >
  > $j$ 代表目前计算层中的激活单元的下标，也将是下一层的第$j$个输入变量的下标。
  >
  > $i$ 代表下一层中误差单元的下标，是受到权重矩阵中第$i$行影响的下一层中的误差单元的下标。

  code：

  ```python
  def gradient(theta, X, y): # 求 j_theta 对 theta 的偏导数， 对应课件 backpropagation algorithm
      # initialize
      t1, t2 = deserialize(theta)  # t1: (25,401) t2: (10,26)
      m = X.shape[0]
  
      delta1 = np.zeros(t1.shape)  # (25, 401)
      delta2 = np.zeros(t2.shape)  # (10, 26)
  
      a1, z2, a2, z3, h = feed_forward(theta, X)
      
      for i in range(m):
          # 第 i 个样本的相关参数
          a1i = a1[i, :]  # (1, 401)
          z2i = z2[i, :]  # (1, 25)
          a2i = a2[i, :]  # (1, 26)
  
          hi = h[i, :]    # (1, 10)
          yi = y[i, :]    # (1, 10)
  
          d3i = hi - yi  # (1, 10)
          z2i = np.insert(z2i, 0, np.ones(1))  # make it (1, 26) to compute d2i, 添加偏置项
          d2i = np.multiply(t2.T @ d3i, sigmoid_gradient(z2i))
          
          delta2 += np.matrix(d3i).T @ np.matrix(a2i)  # (1, 10).T @ (1, 26) -> (10, 26)
          delta1 += np.matrix(d2i[1:]).T @ np.matrix(a1i) # (1, 25).T @ (1, 401) -> (25, 401)
      
      delta1  = delta1 / m
      delta2 = delta2 / m
      return serialize(delta1, delta2)
  ```

  

