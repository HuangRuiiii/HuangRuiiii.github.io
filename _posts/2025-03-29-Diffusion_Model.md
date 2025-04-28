---
layout: post
title: "Diffusion Model"
date: 2025-03-29
author: Rui
tag: Diffusion
typora-copy-images-to: ./..\images\posts\2025-03-29-Diffusion_Model
typora-root-url: ./..
---
当我们说Diffusion model的时候，一般默认为Jonathan Ho在2020年发表在NIPS的Denoising Diffusion Probabilistic Models这篇文章，简称DDPM。Y Song在2019发表在NIPS上的Generative Modeling by Estimating Gradients of the Data Distribution这篇文章提出了一个基于得分函数的生成模型（Score-based generative modeling），该篇文章与DDPM的原理相似，但角度不同，本质上可以互相转换。随后Y Song在2021年发表在ICLR的Score-based generative modeling with stochastic differential equations这篇文章把两种方法用随机微分方程归纳到了同一个框架下。而经典综述文章Understanding Diffusion Models: A Unified Perspective也将**预测原始数据、预测噪声、预测得分函数**这三个不同的角度总结为了Three equivalent interpretations。

本文内容分为四个部分，第一部分介绍DDPM，第二部分介绍Score-based generative modeling，第三部分介绍Score-based generative modeling with stochastic differential equations，第四部分介绍DDIM。

# 1. DDPM

首先要明确的是DDPM是一种潜变量模型。如下图所示，在以往类似VAE的潜变量生成模型中，模型将原始的观测数据$x$（高维）通过一个Encoder压缩到一个潜变量空间$z$（低维），再从潜变量空间中进行采样，并将采样结果通过Decoder还原回样本空间得到生成结果$\hat{x}$ 。在这类模型中，使用潜变量方法有两个好处：一是将离散的高维样本压缩到连续的低维潜变量，从低维的潜变量空间中进行采样相对来说更加容易；二是通过对低维的潜变量空间施加先验分布，使得潜变量空间有规律，从而具备了一定的语义信息（原本的高维样本空间比较稀疏且混乱，不含语义信息）。

<img src="/images/posts/2025-03-29-Diffusion_Model/image-20250423174430391.png" alt="image-20250423174430391" style="zoom: 67%;" />

但VAE这类压缩潜变量空间的方法也有很多限制，比如为了能从潜变量空间采样同时获得tractable inference，须对潜变量空间赋予scale-location family的先验分布，比如高斯。这样的先验分布一定是合理的吗？当然不一定，如果潜变量真的有一个准确的先验分布的话，那也只有上帝才能知道。所以实际上这些简单的先验分布限制了VAE能够学到的潜变量空间的形状，当那个“真实”的分布远比高斯复杂时，这种简单的先验假设回使模型的生成能力和表达能力受限。为了避免这类局限性，Diffusion model不压缩潜变量空间，也不对潜变量空间做先验假设，使得模型能够拟合任意复杂的潜变量分布结构，从而生成高质量的数据样本。

<img src="/images/posts/2025-03-29-Diffusion_Model/image-20250424095009403.png" alt="image-20250424095009403"  />

简单来说，Diffusion model (DM) 由前向加噪过程（训练过程），和反向去噪过程（生成过程）组成。以图像数据为例，假设$x_0$代表原始观测数据，DM逐步向$x_0$加入高斯噪声，假设加噪过程为$1,2,\ldots,T$步，在每一步加噪后得到的数据$x_1,x_2,\ldots,x_T$即为DM的潜变量，当$T$足够大时，DM的加噪方法可保证最终的$x_T$是一个纯高斯噪声。这个加噪过程的目的是为了训练一个能够预测噪声的的神经网络（一般用Unet），加噪过程就是在构造训练神经网络所需要的data和lalel，实际操作过程中在每一步加噪后，用神经网络来预测加入的噪声，由于真实的噪声我们是知道的，因此在预测之后可以算一个loss从而优化这个神经网络。在$T$步加噪完成后，神经网络也就训练完毕。在反向去噪过程中，首先从一个高斯分布中随机采样出一个$x_T$，然后用训练好的神经网络从$x_T$中预测噪声，$x_T$减掉噪声后得到$x_{T-1}$，这个过程反复进行$T$次最终得到（生成）的就是我们想要的观测样本数据$x_0$。

> 这里需要注意，我们前面说DM可以拟合任意复杂的潜变量分布结构，其中“复杂的潜变量分布结构”体现在反向去噪过程中的每一步的中间变量，即 $x_{T-1},\ldots, x_0$。这些中间变量可以随着神经网络的拟合调整成很复杂的分布，而不被简单的高斯所限制。最终，整个生成数据的路径（从高斯到真实数据）是一条经过高度非线性、高复杂度变换的轨迹。所以，DM的强大之处就在于：整个逐步生成的过程，相当于学会了一个“任意复杂的映射轨迹”，而不是单一分布。

接下来具体看一下DM中的公式推导。前向加噪过程定义如下：


$$
q(x_t \mid x_{t-1}) = \mathcal{N}(x_t; \sqrt\alpha_t x_{t-1}, (1-\alpha_t)\mathbf{I})
$$


注意这里面的$\alpha_t$我们可以把它当作一个超参数，也就是说整个前向加噪过程是一个固定的线性高斯过程，没有参数需要学习。同时，这个过程也是一个马尔可夫过程。确定了前向过程后，再来看反向过程的公式表示。我们用$p_\theta(x_{t-1} \mid x_t)$来表示反向过程中用训练好的神经网络（参数为$\theta$）进行的从$t$步到$t-1$步的单步去噪过程。反向去噪过程的联合分布定义如下：


$$
p(x_{0:T}) := p(x_T) \prod_{t=1}^{T} p_\theta(x_{t-1} | x_t)
$$


其中$p(x_T) =\mathcal{N}(x_T; \mathbf{0}, \mathbf{I})$。根据公式可知，DDPM将反向过程自然的定义为了一个马尔可夫过程（否则反向过程的联合分布不能进行上式分解）。在此基础上可进行如下推导：

<img src="/images/posts/2025-03-29-Diffusion_Model/image-20250424001956814.png" alt="image-20250424001956814"  />

上述推导与我们在VAE的文章中所进行的推导式类似的，这里我们只关注最后一行的consistency term项。由推导结果可知，consistency term项是对$T-1$个时间不上的KL divergence的期望求和，这里每个时间步上的KL divergence计算的是在$t$时间步上，由前向扩散过程得到的分布$q(x_t \mid x_{t-1})$和由反向生成得到的分布$p_\theta(x_t \mid x_{t-1})$这两个分布之间的差异。为了最大化ELBO，我们需要让每一步上的这两个分布之间的差异尽可能小，本质上是想让前向加噪扩散得到的$x_t$与反向生成出的$x_t$尽可能结果一致。

<img src="/images/posts/2025-03-29-Diffusion_Model/image-20250424094918060.png" alt="image-20250424094918060"  />

那么如何来计算这个consistency term项呢？这里需要解决的问题有两个，一是如何计算KL divergence，二是如何求期望。针对第一个问题，我们已知$q(x_t \mid x_{t-1}) = \mathcal{N}(x_t; \sqrt\alpha_t x_{t-1}, (1-\alpha_t)\mathbf{I})$，我们只需让去噪神经网络也输出一个高斯分布，即：


$$
p_\theta(x_{t-1}|x_t):=N(x_{t-1}; \mu_\theta(x_t,t), \Sigma_\theta(x_t, t))
$$


其中$\mu_\theta(x_t, t)$和$\Sigma_\theta (x_t, t)$是神经网络的输出，然后根据公式来计算两个高斯分布之间的KL divergence即可。针对第二个问题，我们很自然的想到用MC采样的方法来近似期望。但实际上，这样得到的结果并不好，主要原因在于，consistency term中的期望是关于$x_{t-1}$和$x_{t+1}$两个随机变量来求的，这样用MC方法估计出来的结果方差较大。并且由于整个consistency term要在$T-1$步上求和，因此当$T$较大的时候，方差也会累计的很大。

解决方案是重新推导ELBO，尝试将consistency term中关于两个随机变量求期望转换为关于一个随机变量求期望。这里一个key sight在于，得益于我们所定义的前向加噪的马尔可夫过程，有$q(x_t \mid x_{t-1}) = q(x_t \mid x_{t-1}, x_0)$，然后根据贝叶斯公式我们可以将其重写为：


$$
q(x_t \mid x_{t-1}, x_0) = \frac{q(x_{t-1} \mid x_t, x_0)q(x_t \mid x_0)}{q(x_{t-1} \mid x_0)} \tag{1}
$$


将上式代入到ELBO的推导过程中

<img src="/images/posts/2025-03-29-Diffusion_Model/image-20250424104216573.png" alt="image-20250424104216573" />

由上述推导可知，原本关于两个随机变量求期望的consistency term累加项，转换为了关于一个随机变量求期望的denoising matching term累加项，这时再用MC采样方法去估计该期望项，便可以得到一个方差更小的结果。还记得我们前面说求consistency term有两个问题，一个是求KL divergence，一个是求期望。现在求期望的问题解决了，但由于KL divergence中原本简单的$q(x_{t} \mid x_{t-1})$（高斯分布）变为了$q(x_{t-1} \mid x_t, x_0)$（前向过程的后验分布），因此现在问题又重新回到如何求KL divergence。还是利用贝叶斯公式：


$$
q(x_{t-1} \mid x_t, x_0) = \frac{q(x_t \mid x_{t-1}, x_0)q(x_{t-1} \mid x_0)}{q(x_t \mid x_0)}
$$


其中$q(x_t \mid x_{t-1}, x_0)=q(x_t \mid x_{t-1})$是我们定义的前向过程（高斯），因此等式右侧需要我们进行推导的就只有$q(x_{t-1} \mid x_0)$和$q(x_t \mid x_0)$。由于DDPM所定义的前向过程是线性高斯的，根据重参数化技巧，对$x_t \sim q(x_t \mid x_{t-1})$采样可以被写为：


$$
x_t=\sqrt{\alpha_t}x_{t-1} + \sqrt{1-\alpha_t}\epsilon, \quad \epsilon \sim \mathcal{N}(\epsilon;\mathbf{0}, \mathbf{I})
$$


类似的对$x_{t-1} \sim q(x_{t-1} \mid x_{t-2})$采样也可以被重参数化为：


$$
x_{t-1}=\sqrt{\alpha_{t-1}}x_{t-2}+\sqrt{1-\alpha_{t-1}}\epsilon, \quad \epsilon \sim \mathcal{N}(\epsilon;\mathbf{0}, \mathbf{I})
$$


这样我们就可以递归的得到对$x_t \sim q(x_t \mid x_0)$的重参数化的采样结果：

<img src="/images/posts/2025-03-29-Diffusion_Model/image-20250424111058805.png" alt="image-20250424111058805" style="zoom:80%;" />

其对应的分布即为：


$$
q(x_t \mid x_0)=\mathcal{N}(x_t ; \sqrt{1-\bar{\alpha}}_t x_0, (1-\bar{\alpha}_t)\mathbf{I})
$$


这个公式告诉我们，基于线性高斯的加噪假设，DDPM的加噪过程并不需要真的逐步从$x_0,x_1,\ldots$一直加噪到$x_t$，而是可以通过公式直接推导出任意时间步上的加噪数据$x_t$。回到前面的问题，有了$q(x_t \mid x_0)$和$q(x_{t-1} \mid x_0)$，我们就可以推导出$q(x_{t-1} \mid x_t, x_0)$的具体分布：

![image-20250424112459522](/images/posts/2025-03-29-Diffusion_Model/image-20250424112459522.png)

通过推理知道前向过程的后验分布$q(x_{t-1} \mid x_t, x_0)$服从高斯，因此可以将反向过程的近似后验$p_\theta(x_{t-1} \mid x_t)$也建模成高斯。具体来说就是让神经网络去model这个高斯分布$p_\theta(x_{t-1} \mid x_t)$的均值$\mu_\theta(x_t, t)$和方差$\Sigma_\theta(t)$。但是注意，根据上面的推导结果来看，在给顶$\alpha_t$的情况下，方差应该是一个固定常数，因此我们只需要对高斯的均值进行参数化建模即可。根据两个高斯分布之间的KL divergence计算公式可得：

<img src="/images/posts/2025-03-29-Diffusion_Model/image-20250424120627845.png" alt="image-20250424120627845" />

> 在我们前面刚刚提到让近似后验的高斯均值与真实后验的高斯均值一致的时候，其实第一反应应该是直接让两者之差最小。但是根据上面的推到结果来看，最终优化的时候目标确是让两者的二范数距离最小，因此这个优化的目标函数实际上是一个可以变动的地方，文中选择用KL divergence来作为目标函数（不是选择用KL divergence，而是推导出了KL divergence），本质上minimize二范数距离， 那当然也可以用一范数距离或者其它距离。

到这里应该可以结束了，训练时只需要预测出高斯分布的均值$\mu_\theta(x_t, t)$，然后计算这个二范数距离进行优化就好了。但从实际效果来看还不够好，还可以再细致一些。$\mu_\theta(x_t, t)$与$\mu_q(x_t, x_0)$之间相差的其实只有$x_0$，真实的后验均值$\mu_q(x_t, x_0)$是在已知$x_t$和$x_0$的条件下推出来的，近似的后验均值只能利用$x_t$，而$x_0$是未知的。所以是不是只需要让模型去预测$x_0$就好了？我们尝试把$\mu_\theta(x_t, t)$和$\mu_q(x_t, x_0)$的具体表达式代入进去，其中：


$$
\mu_q(x_t,x_0)=\frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})x_t+\sqrt{\bar{\alpha}_{t-1}}(1-\alpha_t)x_0}{1-\bar{\alpha}_t}
$$


$\mu_\theta(x_t,t)$还没有具体形式（因为前面说的是让模型直接输出均值，所以没有具体形式），那不妨仿照$\mu_q(x_t,x_0)$的形式来写一个，其中未知的$x_0$的未知就用$\hat{x}_\theta(x_t, t)$来替代：


$$
\mu_\theta(x_t, t) = \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})x_t + \sqrt{\bar{\alpha}_{t-1}}(1-\alpha_t)\hat{x}_\theta(x_t,t)}{1-\bar{\alpha}_t}
$$


代入到上面的优化公式中：

<img src="/images/posts/2025-03-29-Diffusion_Model/image-20250424122524206.png" alt="image-20250424122524206" />

也就是说只需让神经网络来预测观测样本数据$x_0$就可以了。做具体优化时，denoising  matching term需要对$T-1$步上的期望求和，这个优化目标等价于下式

<img src="/images/posts/2025-03-29-Diffusion_Model/image-20250424123221750.png" alt="image-20250424123221750" />

DDPM中进一步对优化目标进行分析，用噪声$\epsilon_0$替换掉了$x_0$（不难推），因此DDPM只需要预测噪声即可得到更好的生成结果。

<img src="/images/posts/2025-03-29-Diffusion_Model/image-20250424123759512.png" alt="image-20250424123759512" />

算法写起来并不复杂。前向加噪的过程就是训练的过程，输入一张样本$x_0$，采样一个时间步$t\sim \mathcal{U}(1,\ldots,T)$，根据公式向样本中加入对应时间步的噪声$\epsilon$得到$x_t$，然后SGD训练神经网络，预测加入的噪声，反复进行直至收敛。反向采样过程，就是按照时间步从后往前，用训练好的模型逐步从$x_T$中去除噪声。

<img src="/images/posts/2025-03-29-Diffusion_Model/image-20250424124232525.png" alt="image-20250424124232525" />

# 2. Score based generative model

## 2.1 Score matching

在介绍Score based generative model之前首先来看一下Score matching方法。Score matching方法提出来最初是为了解决Non-Normalized Statistical Models的参数估计问题。这类估计问题比较特殊，我们只知道密度函数$p_\theta$中的一部分$q_\theta$，而未知的另一常数部分$Z_\theta$与分布的参数$\theta$有关，因此不能直接用最大似然的方法进行估计。

<img src="/images/posts/2025-03-29-Diffusion_Model/image-20250424124657285.png" alt="image-20250424124657285" style="zoom: 50%;" />

如何拿掉这个未知的常数部分呢？最直接的想法就是求导。作者定义了一个Score function，关于$x$对$\log p_\theta(x)$求导，求导后的结果$S_\theta (x)$即为Score function，不再依赖于常数$Z_\theta$(注意这里所说的常数是指其包含**参数变量**$\theta$，但不包含**观测变量**$x$, 而求导是关于观测变量求导，因此$Z_\theta$可以视为常数)。

<img src="/images/posts/2025-03-29-Diffusion_Model/image-20250424152102700.png" alt="image-20250424152102700" />

如上图所示，$p_\theta(x)$为我们要建模的概率密度函数，$p_{data}(x)$为样本数据真实的概率密度函数，那么相应的就应该有两个Score function，一个是Model score function $S_\theta(x)$，一个是Data score function $S_{data}(x)$，其中$S_{data}(x)$还是未知的。我们希望用matching两个Score function的方法来估计模型参数$\theta$，也就是让$p_\theta(x)$和$p_{data}(x)$关于观测变量的导数处处相等。然而$S_{data}(x)$是未知的，如何才能计算出$S_\theta(x)$和$S_{data}(x)$两者之间的差异，从而来优化参数$\theta$呢？

<img src="/images/posts/2025-03-29-Diffusion_Model/image-20250424152205640.png" alt="image-20250424152205640" />

如果还要估计出$S_{data}(x)$的话，那问题就变得很麻烦了。好在原文中提供了一个定理，对上图中的目标函数$J(\theta)$可以进行等价的转换（具体推导参考原文）。转换之后$S_{data}(x)$被放在了与参数$\theta$无关的常数项里。因此最终的目标函数就可以被写为下图中的形式，minimize的时候可以把const去掉。

<img src="/images/posts/2025-03-29-Diffusion_Model/image-20250424153420845.png" alt="image-20250424153420845" />

总结一下，Score matching方法的最大优势，是我们不需要知道完整的模型的概率密度函数$p_\theta(x)$，只需知道得分函数$S_\theta(x)$即可完成近似。

## 2.2 Score-based generative modeling

在了解了Score matching方法之后，我们再来看一下这篇发表于2019的文章*Score-based generative modeling*。这里我们主要关注如何把Score matching的方法用在生成模型上。模型假设我们手中有来自$p_{data}(x)$的观测样本数据$x_1, x_2, \ldots x_N$。模型的目标是用$p_\theta(x)$来model真实分布$p_{data}(x)$，从而我们可以在$p_\theta(x)$中采样生成新的数据。想要实现这个目标最直接的办法是对概率密度函数直接建模，然后使用最大似然。然而这个方法显然是不行的，因为我们不知道的这个概率密度函数具体应该是什么形式，所以基于最大似然的很多模型做了假设，比如VAE就假设潜变量空间里是一个高斯，这样就可以做最大似然了。但是这些假设是否合理呢？并不一定，大多数时候只是为了方便计算而已。

<img src="/images/posts/2025-03-29-Diffusion_Model/image-20250424154031135.png" alt="image-20250424154031135" />

但是如果我们去model分布的Score function而不是pdf的话，就可以让这个问题变简单一些。注意得分函数$S_\theta(x)$并不依赖于归一化常数$Z_\theta$，这意味着对于模型的输出结果没有归一化的约束，因此会显著增强模型的灵活性，我们可以使用一个神经网络来model得分函数$S_\theta(x)$。

<img src="/images/posts/2025-03-29-Diffusion_Model/image-20250424155401493.png" alt="image-20250424155401493" />

这时候我们的训练目标就是：

<img src="/images/posts/2025-03-29-Diffusion_Model/image-20250424155454975.png" alt="image-20250424155454975" />

根据前面2.1小节部分对Score matching的介绍我们已经知道了，这个期望中我们未知的那一部分$p_{data}(x)$是可以作为与$\theta$无关的常数项放在一边的，因此这个问题可以优化求解。也就是说我们可以训练出一个神经网络，让它在每个$x$的位置输出正确的Score function。

<img src="/images/posts/2025-03-29-Diffusion_Model/image-20250424160516915.png" alt="image-20250424160516915" />

如上图所示，训练出的神经网络输出的得分函数是一个向量场（因为是关于$x$的梯度），这个$S_\theta(x)$所对应的向量场和$S_{data}(x)$所对应的向量场应该是相近的。到这里，我们稍微梳理一下这一小节的内容，我们本来希望对数据的概率密度函数$p_\theta(x)$进行建模，让$p_\theta(x)$近似$p_{data}(x)$，从而可以从$p_\theta(x)$中采样出样本数据，但是对$p_\theta(x)$建模难度较大，因为我们不知道这个分布的具体形式应该是什么样的，所以退而求其次，我们对数据的得分函数$S_\theta(x)$进行建模，让$S_\theta(x)$近似$S_{data}(x)$，那么现在的问题就是，我们该如何根据$S_\theta(x)$来进行采样生成呢？

<img src="/images/posts/2025-03-29-Diffusion_Model/image-20250424162631317.png" alt="image-20250424162631317" />

如上图所示，文章中使用了Langevin动力系统进行MCMC采样。上图的采样公式中，由于噪声项$z_i$的加入使得从同一个起始点出发，我们能够通过Langevin动力学采样采到来自不同模态的样本；如果没有这个噪声，从一个固定点采样时，每次都会沿着得分函数确定性地走向同一个模态。

<img src="/images/posts/2025-03-29-Diffusion_Model/image-20250424170650386.png" alt="image-20250424170650386" />

<img src="/images/posts/2025-03-29-Diffusion_Model/image-20250424170847279.png" alt="image-20250424170847279" />

到目前为止，我们已经讨论了如何通过Score matching来训练一个基于得分的模型，并通过Langevin动力学采样生成样本。然而，这种简单的方法在实际中操作中效果并不好，主要是存在以下两个问题：

1. 在低密度区域，由于可用于计算Score matching目标函数的数据点很少，估算得到的得分函数往往不准确；
2. 得分函数估计不准确会导致Langevin动力学采样过程一开始就发生偏离，无法采样生成高质量样本。

<img src="/images/posts/2025-03-29-Diffusion_Model/image-20250424171138064.png" alt="image-20250424171138064" />

这两个问题本质上是一个问题，都是在说模型在低密度区域的Score function估计不准确的问题（图片中的示例好像没有准确的展现出困难）。**解决问题的办法就是向数据中添加噪声，当噪声的幅度足够大时，它就可以覆盖数据的低密度区域，从而提高Score function估计的准确性**。

> 以下是我个人对这句话的理解。想象有两座山峰被浸在海水中，海平面上只露出两个山尖（高密度区域），而山凹（低密度区域）则被淹没在海平面下，这时候我从外面来看（观测样本）就只能看到两个山尖和海平面。假设我从上往下扔一个小球，这个小球会沿着坡度向上滚（Langevin采样），那么如果这个小球落在了露在海平面上的那一部分山体上（高密度区域），那么它自然会滚到正确的位置（采样准确），但如果小球落在了海平面上，这个区域没有坡度，它就没有办法滚到正确的位置（采样不准确）。解决办法就是，向这两座山上同时填土（加噪声），这样两座山的整体海拔就会一起升高，尽管可能会模糊掉一些细节，但当升高到一定程度时，山凹也会露在海平面之上。这个时候整个小球落下的区域都是有坡度的，从而使得在任意位置落下的小球都能滚到正确的位置去。

添加噪声的目的是为了填充原本的低密度数据区域，添加噪声后的数据虽然可能会覆盖掉一些原本低密度区域的结构特征，但仍然会保留高密度区域的结构特征，这使得Langevin采样能够有效采样出高密度区域的数据（这些高密度区域的数据就是我们可能会观测到的样本数据）。

<img src="/images/posts/2025-03-29-Diffusion_Model/image-20250424173432416.png" alt="image-20250424173432416" />

有了这个解决方案之后，下一个问题是，该如何确定加入噪声的程度呢？加的太轻了没有用，加的太重了就全都覆盖成高斯噪声了。既然无法确定应该加入的噪声大小，那就在不同尺度上都去加噪。

<img src="/images/posts/2025-03-29-Diffusion_Model/image-20250427183837290.png" alt="image-20250427183837290" />

上图对应的就是DDPM中的前向加噪过程（注意这一页中的分布都是加噪过程中的真实分布，还没有用到参数）。有了这个方案之后可以使用重参数化技巧进行采样。再接下来我们就可以训练一个神经网络来估计这个加噪后的数据的score function。

<img src="/images/posts/2025-03-29-Diffusion_Model/image-20250427183918003.png" alt="image-20250427183918003" />

确定了加噪方法之后，采样方法也需要重新考虑。经过训练的模型会从预测一个纯高斯的向量场$S_\theta(x_L,L)$开始，然后使用Langevin动力系统进行采样，采样结果作为$S_\theta(x_{L-1},L-1)$的初始状态，然后再进行采样（注意每一步的采样都基于前一步的采样结果），这样逐步向前，最终采样结果就会收敛到真实的样本分布。以上的采样过程被称为Annealed Langevin采样。

<img src="/images/posts/2025-03-29-Diffusion_Model/ald.gif" alt="ald" />

到这里SGM就结束了，这个思路和DDPM不太一样，但是如果反映在数据和操作上，所做的事情是类似的。因此Diffusion model有了第三个等价的形式。

<img src="/images/posts/2025-03-29-Diffusion_Model/image-20250424175806794.png" alt="image-20250424175806794" />

# 3. Score-based generative modeling with stochastic differential equations (SDEs)

## 3.1 SDE

这篇文章为使用随机微分方程（SDE）为前面的所有内容提供了一统一的框架。文章中指出，我们可以将加噪过程看作一个连续时间的随机过程。

<img src="/images/posts/2025-03-29-Diffusion_Model/perturb_vp.gif" alt="perturb_vp" />

随机过程是随机微分方程的解，因此我们可以用随机微分方程（SDE）来描述这些随机过程。具体形式如下图所示：

<img src="/images/posts/2025-03-29-Diffusion_Model/image-20250424180308255.png" alt="image-20250424180308255" />

原本SGM和DDPM的加噪过程可以看作是这个连续随机过程的离散版本。上图中SDE里面的这几个系数（drift coefficient, diffusion coefficient）是可以人为设计的。所以加噪的过程，就是设计SDE的过程。回想一下前面的内容，无论是DDPM还是SGM，我们都是在用一个反向去噪过程来生成样本，那么相应地，对于SDE来说我们也可以通过解一个反向随机微分方程（Reverse SDE）来实现样本生成 （关于Reverse SDE，有一篇1982年的参考文献证明对于任意SDE，存在Reverse SDE）。

<img src="/images/posts/2025-03-29-Diffusion_Model/denoise_vp.gif" alt="denoise_vp" />

如下图所示Reverse SDE会包含一个得分函数，而根据第2节所讲的内容，我们可以训练出一个Score-based model，将模型预测的结果插入到这个方程中，之后要做的就是解这个Reverse SDE。

<img src="/images/posts/2025-03-29-Diffusion_Model/image-20250424181329665.png" alt="image-20250424181329665" />

解SDE的方法有很多种，这里我们简单看一下其中的一种数值方法

<img src="/images/posts/2025-03-29-Diffusion_Model/image-20250424185332642.png" alt="image-20250424185332642" />

图中所示的这种数值解法，对应的就是2.2小节所讲的Annealed Langevin采样，两者都是用Score function来不断的更新采样结果。

## 3.2 ODE

文章中证明了可以将任何随机微分方程（SDE）转换为常微分方程（ODE），同时保持其边缘分布 $p_t(x) , t\in[0,T]$不变。因此，通过求解这个ODE得到的结果，一定程度上等价于求解Reverse SDE的结果（采样结果）。这个与SDE对应的ODE被称为**概率流ODE（probability flow ODE）**，如下图所示。

<img src="/images/posts/2025-03-29-Diffusion_Model/image-20250426145744746.png" alt="image-20250426145744746" />

注意虽然看起来ODE就是把SDE的随机项去掉，但是实际上好像没有这么简单，在论文附录中有详细的推导可以参考。更本质的对应关系可能要查看SDE相关书籍了。ODE是确定的没有随机噪声没有随机性，因此生成的样本轨迹更加平滑且具有可重复性，可插值，生成样本的质量也更高，同时还可以提高采样效率。但相应的也牺牲了一定的采样结果的多样性。

# 4. DDIM

