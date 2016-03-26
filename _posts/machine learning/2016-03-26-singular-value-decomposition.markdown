---
layout: post
title:  "Singular Value Decomposition"
date: 2016-03-25 13:56:53
categories: Machine&nbsp;Learning
comments: true
---

* content
{:toc}

## 1. 对矩阵基本的理解

一个m行n列的矩阵，符号标记为M，其可以被视为线性变换的便利表达法

比如，有一个n维向量x，则通过矩阵乘法$$Mx=y$$便可以得到一个m维向量y

矩阵M记录了一个从n维空间到m维空间的一个线性变换

为了方便，以下考虑的均是理想情况

## 2. 特征值与特征向量

M是一个n行n列的方阵

若有$$\lambda,v$$满足$$Mv=\lambda v$$，其中，$$\lambda$$为数值，$$v$$为n维向量

则称$$v$$为矩阵M的特征向量，且对应一个特征值$$\lambda$$

特征向量经过线性变换后不改变方向，可以将特征向量视为这个线性变换的基

特征值可视为这个线性变换在这个基上的权重

令矩阵M有n个线性无关的特征向量$$v_1,v_2,...,v_n$$，对应的特征值为$$\lambda_1,\lambda_2,...,\lambda_n$$

且满足
\begin{align}
\lambda_1 \geq \lambda_2 \geq ... \geq \lambda_n
\end{align}

则有

\begin{align}
M * \begin{bmatrix} v_1 & v_2 & ... & v_n \end{bmatrix} = 
\begin{bmatrix}\lambda_1 v_1 & \lambda_2 v_2 & ... & \lambda_n v_n \end{bmatrix}=
\begin{bmatrix} v_1 & v_2 & ... & v_n \end{bmatrix}
\begin{bmatrix}
\lambda_1 & 0 & ... & 0 \\\
0 & \lambda_2 & ... & 0 \\\
... & ... & ... & ... \\\
0 & 0 & 0 & \lambda_n
\end{bmatrix}
\end{align}

令
\begin{align}
V &= \begin{bmatrix} v_1 & v_2 & ... & v_n \end{bmatrix} \\\
\Lambda &=
\begin{bmatrix}
\lambda_1 & 0 & ... & 0 \\\
0 & \lambda_2 & ... & 0 \\\
... & ... & ... & ... \\\
0 & 0 & 0 & \lambda_n
\end{bmatrix}
\end{align}

则有

\begin{align}
M = V \Lambda V^{-1}
\end{align}

上式称为特征值分解

## 3. 奇异值分解

特征值分解只适用于方阵，对于普通的m行n列矩阵，则使用奇异值分解

令$$u_1,u_2,...,u_m$$为矩阵$$M*M^T$$的特征向量, 称为左奇异向量

令$$v_1,v_2,...,v_n$$为矩阵$$M^T*M$$的特征向量，称为右奇异向量

令
\begin{align}
U &= \begin{bmatrix}
u_1 & u_2 & ... & u_m
\end{bmatrix} \\\
V &= \begin{bmatrix}
v_1 & v_2 & ... & v_n
\end{bmatrix}
\end{align}

直接上定义
\begin{align}
M = 
\begin{bmatrix}
u_1 & u_2 & ... & u_m
\end{bmatrix}
\begin{bmatrix}
\sigma_1 & 0 & ... \\\
0 & \sigma_2 & ... \\\
... & ... & ...
\end{bmatrix}
\begin{bmatrix}
v_1^T \\\
v_2^T \\\
... \\\
v_n^T
\end{bmatrix}
\end{align}
其中
\begin{align}
\Sigma= 
\begin{bmatrix}
\sigma_1 & 0 & ... \\\
0 & \sigma_2 & ... \\\
... & ... & ...
\end{bmatrix}
\end{align}

为m行n列的对角矩阵，对角线上的值称为奇异值

设有n维向量，则其可表示为
\begin{align}
x = a_1 v_1 + a_2 v_2 + ... + a_n v_n
\end{align}

接下来一步步模拟线性变换的过程，首先左乘$$V^T$$

\begin{align}
\begin{bmatrix}
v_1^T \\\
v_2^T \\\
... \\\
v_n^T
\end{bmatrix}x =
\begin{bmatrix}
a_1v_1^Tv_1 \\\
a_2v_2^Tv_2 \\\
... \\\
a_nv_n^Tv_n \\\
\end{bmatrix}
\end{align}

接着左乘$$\Lambda$$

如果$$m > n$$
\begin{align}
\begin{bmatrix}
\sigma_1 & 0 & ... \\\
0 & \sigma_2 & ... \\\
... & ... & ...
\end{bmatrix}
\begin{bmatrix}
a_1v_1^Tv_1 \\\
a_2v_2^Tv_2 \\\
... \\\
a_nv_n^Tv_n \\\
\end{bmatrix} = 
\begin{bmatrix}
a_1 \sigma_1 v_1^Tv_1 \\\
a_2 \sigma_2 v_2^Tv_2 \\\
... \\\
a_n \sigma_n v_n^Tv_n \\\
... \\\
0 \\\
0
\end{bmatrix}
\end{align}

则\begin{align}x = a_1\sigma_1v_1^Tv_1u_1 + a_2\sigma_2v_2^Tv_2u_2 + ... + a_n\sigma_nv_n^Tv_nu_n\end{align}

如果 $$m < n$$
\begin{align}
\begin{bmatrix}
\sigma_1 & 0 & ... \\\
0 & \sigma_2 & ... \\\
... & ... & ...
\end{bmatrix}
\begin{bmatrix}
a_1v_1^Tv_1 \\\
a_2v_2^Tv_2 \\\
... \\\
a_nv_n^Tv_n \\\
\end{bmatrix} = 
\begin{bmatrix}
a_1 \sigma_1 v_1^Tv_1 \\\
a_2 \sigma_2 v_2^Tv_2 \\\
... \\\
a_m \sigma_m v_m^Tv_m \\\
\end{bmatrix}
\end{align}
则\begin{align}x = a_1\sigma_1v_1^Tv_1u_1 + a_2\sigma_2v_2^Tv_2u_2 + ... + a_m\sigma_mv_m^Tv_mu_m\end{align}

由上式，可以这么理解线性变换

首先将n维向量使用矩阵的右奇异向量作为基表示，再将每个维度映射到一个左奇异向量基

奇异值则可以认为是输入与输出间进行的标量的膨胀控制。

# 4. Reference

* [Wiki-特征分解]
* [Wiki——奇异值分解]

[Wiki-特征分解]: https://zh.wikipedia.org/wiki/%E7%89%B9%E5%BE%81%E5%88%86%E8%A7%A3
[Wiki——奇异值分解]: https://zh.wikipedia.org/wiki/%E5%A5%87%E5%BC%82%E5%80%BC%E5%88%86%E8%A7%A3
