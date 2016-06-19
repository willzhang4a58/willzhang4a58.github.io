---
layout: post
title:  "Bias-Variance Decomposition"
date: 2016-06-18 17:16:24
categories: Machine&nbsp;Learning
comments: true
---

`本文大部分参考周志华著《机器学习》`

同一个算法，在不同数据集上的训练结果会有差异

对测试样本$$x$$，令$$y_D$$为其在数据集D中的Label，令$$y$$为其真实的Label，令$$f(x;D)$$为在数据集$$D$$上训练的模型对$$x$$的预测结果

为了方便，设定为回归任务，则期望预测为

\begin{align}
\bar{f}(x)=E_D[f(x;D)]
\end{align}

其与真实Label的差别称为Bias

\begin{align}
bias^2(x) = (\bar{f}(x) - y)^2
\end{align}

使用样本数相同的不同训练集产生的方差为

\begin{align}
var(x) = E_D\left[\left(f(x;D) - \bar{f}(x)\right)^2\right]
\end{align}

Noise为

\begin{align}
\varepsilon^2=E_D\left[ (y_D - y)^2  \right]
\end{align}

然后，我们开始对泛化误差的分解

\begin{align}
E(f;D) &= E_D\left[ \left(f(x;D) - y_D\right)^2 \right] \\\
&= E_D\left[ \left(f(x;D) -\bar{f}(x) + \bar{f}(x)- y_D\right)^2 \right] \\\
&= E_D \left[\left(f(x;D) -\bar{f}(x) \right)^2\right] + E_D\left[ \left(\bar{f}(x)- y_D\right)^2\right] + E_D\left[2\left(f(x;D) -\bar{f}(x)\right)\left(\bar{f}(x)- y_D\right)\right]\\\
&= E_D \left[\left(f(x;D) -\bar{f}(x) \right)^2\right] + E_D\left[ \left(\bar{f}(x)- y_D\right)^2\right] \\\
&= E_D \left[\left(f(x;D) -\bar{f}(x) \right)^2\right] + E_D\left[ \left(\bar{f}(x)-y+y- y_D\right)^2\right] \\\
&= E_D \left[\left(f(x;D) -\bar{f}(x) \right)^2\right] + E_D\left[ \left(\bar{f}(x)-y\right)^2\right]+E_D\left[ \left(y- y_D\right)^2\right]+2E_D\left[ \left(\bar{f}(x)-y)(y- y_D\right)^2\right] \\\
&= E_D \left[\left(f(x;D) -\bar{f}(x) \right)^2\right] + \left[ \left(\bar{f}(x)-y\right)^2\right]+E_D\left[ \left(y- y_D\right)^2\right] \\\
&= var(x) + bias^2(x) + \varepsilon^2
\end{align}

泛化误差可分解为Bias，Variance，Noise之和

* Bias度量了学习算法的期望预测与真实结果的偏离程度，反映了学习算法本身的拟合能力
* Variance度量同等大小的训练集的变动导致的差异，反映了数据扰动的影响
* Noise表达了当前任务任何学习算法的期望泛化误差的下界

一般来说，Bias和Variance存在冲突，一方变好的同时会引起另一方的变差




