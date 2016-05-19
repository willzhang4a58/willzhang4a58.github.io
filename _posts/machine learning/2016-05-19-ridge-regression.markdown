---
layout: post
title:  "Bayesian Linear Regression"
date: 2016-05-18 12:52:47
categories: Machine&nbsp;Learning
comments: true
---

## 1. 线性回归

在一个标准的线性回归问题中，我们有$$n$$个数据点$$\left\{x_1,x_2,...,x_n\right\}$$，以及对应的$$\left\{y_1,y_2,...,y_n\right\}$$

且$$x_i \in R^{k*1}$$，$$y_i \in R$$

而线性回归中，给定$$x_i$$和参数$$\beta \in R^{k*1}$$，$$y_i$$的条件分布设定为

\begin{align}
y_i = x_i^T\beta + \epsilon_i \\\
\end{align}

其中，$$\epsilon_i \sim N(0, \sigma^2)$$

因此，$$y_i \sim N(x_i^T\beta, \sigma^2)$$

使用符号$$X$$表示一个$$n$$行$$k$$列的矩阵，第$$i$$个行向量代表$$x_i^T$$

使用符号$$y$$表示大小为$$n$$的向量，第$$i$$个元素代表$$y_i$$

频率方法可直接得到一个解

\begin{align}
\hat{\beta} = (X^TX)^{-1}X^Ty
\end{align}

## 2. 共轭先验

在贝叶斯方法中，我们还需要补充先验，在那之前先整理似然函数

\begin{align}
P(y|X,\beta,\sigma^2) &= \prod_{i=1}^nN(x_i^T\beta, \sigma^2) \\\
&= \left\(\frac{1}{\sigma\sqrt{2\pi}}\right\)^nexp\left\(-\frac{\sum_{i=1}^n(y_i - x^T_i\beta)^2}{2\sigma^2}\right\)\\\
&= \left\(\frac{1}{\sigma\sqrt{2\pi}}\right\)^nexp\left\(-\frac{(y-X\beta)^T(y-X\beta)}{2\sigma^2}\right\)
\end{align}

去掉常数，有

\begin{align}
P(y|X,\beta,\sigma^2) \propto \sigma^{-n}exp\left\(-\frac{(y-X\beta)^T(y-X\beta)}{2\sigma^2}\right\)
\end{align}

将$$(y-X\beta)^T(y-X\beta)$$改写形式

\begin{align}
(y-X\beta)^T(y-X\beta) = (y-X\hat{\beta})^T(y-X\hat{\beta}) + (\beta - \hat{\beta})^T(X^TX)(\beta - \hat{\beta})
\end{align}

令

\begin{align}
v &= n - k \\\
vs^2 &= (y-X\hat{\beta})^T(y-X\hat{\beta}) 
\end{align}

于是我们得到一个新形式的似然函数

\begin{align}
P(y|X,\beta,\sigma^2) & \propto \sigma^{-n}exp\left\(-\frac{vs^2 + (\beta-\hat{\beta})^T(X^TX)(\beta-\hat{\beta})}{2\sigma^2}\right\) \\\
& \propto \sigma^{-n}exp\left\(-\frac{vs^2 }{2\sigma^2}\right\)exp\left\(-\frac{(\beta-\hat{\beta})^T(X^TX)(\beta-\hat{\beta})}{2\sigma^2}\right\) \\\
& \propto \left\[(\sigma^2)^{-v/2}exp\left\(-\frac{vs^2 }{2\sigma^2}\right\)\right\] \left\[(\sigma^2)^{-k/2}exp\left\(-\frac{(\beta-\hat{\beta})^T(X^TX)(\beta-\hat{\beta})}{2\sigma^2}\right\)\right\]
\end{align}


令$$v_0,s_0$$为$$v,s$$的先验值，随后为了使先验共轭，令先验的形式与似然函数对应

\begin{align}
P(\beta,\sigma^2) = P(\sigma^2)P(\beta|\sigma^2)
\end{align}

其中，令

\begin{align}
P(\sigma^2) =Inv-Gamma(\frac{v_0}{2}, \frac{v_0s_0^2}{2}) \propto \sigma^{-(v_0+2)}exp\left\( -\frac{v_0s_0^2}{2\sigma^2} \right\)
\end{align}

设定$$a_0=\frac{v_0}{2},b_0=\frac{v_0s_0^2}{2}$$，于是

\begin{align}
P(\sigma^2) \propto \sigma^{-(2a_0+2)}exp\left\( -\frac{b_0}{\sigma^2} \right\)
\end{align}

而另一个条件概率设定为高斯分布，式中的$$\mu_0,\Lambda_0$$为先验值	

\begin{align}
P(\beta |\sigma^2) \propto \sigma^{-k}exp\left\( -\frac{(\beta-\mu_0)^T\Lambda_0(\beta - \mu_0)}{2\sigma^2} \right\)
\end{align}

## 3. 后验

有了先验和似然，可以开始计算后验了，根据贝叶斯公式

\begin{align}
P(\beta,\sigma^2|y,X) &\propto P(y|X,\beta,\sigma^2)P(\beta|\sigma^2)P(\sigma^2) \\\
& \propto \left\[\sigma^{-n}exp\left\(-\frac{(y-X\beta)^T(y-X\beta)}{2\sigma^2}\right\)\right\]\left\[\sigma^{-k}exp\left\( -\frac{(\beta-\mu_0)^T\Lambda_0(\beta - \mu_0)}{2\sigma^2} \right\)\right\] \\\
&* \left\[\sigma^{-(2a_0+2)}exp\left\( -\frac{b_0}{\sigma^2} \right\)\right\]
\end{align}

由于

\begin{align}
(y-X\beta)^T(y-X\beta) + (\beta-\mu_0)^T\Lambda_0(\beta - \mu_0) = (\beta-\mu_n)^T(X^TX+\Lambda_0)(\beta - \mu_n) + y^Ty - \mu_n^T(X^TX+\Lambda_0)\mu_n + \mu_0^T\Lambda_0\mu_0
\end{align}

式中$$\mu_n = (X^TX + \Lambda_0)^{-1}(X^TX\hat{\beta} + \Lambda_0\mu_0)$$，由上式也可以看出$$\mu_n$$就是$$\beta$$的后验均值

使用$$\mu_n$$，可以整理后验概率如下

\begin{align}
P(\beta,\sigma^2|y,X) \propto \left\[\sigma^{-k}exp\left\( -\frac{(\beta-\mu_n)^T(X^TX + \Lambda_0)(\beta-\mu_n)}{2\sigma^2} \right\)\right\] \left\[\sigma^{-n-2a_0-2} exp \left\( -\frac{2b_0 + y^Ty - \mu_n^T(X^TX + \Lambda_0)\mu_n+\mu_0^T\Lambda_0\mu_0}{2\sigma^2} \right\)\right\]
\end{align}

可以看到后验概率整理成一个高斯分布$$N(\mu_n,\sigma^2\Lambda_n^{-1})$$乘上一个逆伽马分布$$Inv-Gamma(a_n,b_n)$$，与先验概率的形式保持一致

由整理后的式子容易得到

\begin{align}
\mu_n &= (X^TX + \Lambda_0)^{-1}(\Lambda_0\mu_0 + X^Ty) \\\
\Lambda_n &= X^TX + \Lambda_0 \\\
a_n &= a_0 + \frac{n}{2} \\\
b_n &= b_0 + \frac{y^Ty+\mu_0^T\Lambda_0\mu_0-\mu_n^T\Lambda_n\mu_n}{2}
\end{align}

也就是

\begin{align}
P(\beta,\sigma^2|y,X) \propto \left\[\sigma^{-k}exp\left\( -\frac{(\beta-\mu_n)^T\Lambda_n(\beta-\mu_n)}{2\sigma^2} \right\)\right\] \left\[(\sigma^2)^{-a_n-1} exp \left\( -\frac{b_n}{\sigma^2} \right\)\right\]
\end{align}

由上式可以看出后验概率$$\beta \sim N(\mu_n, \Lambda_n^{-1})$$

## 4. Ridge Regression

当$$\mu_0=0,\Lambda_0=cI$$时称为Ridge Regression

式中的$$c \in R$$
