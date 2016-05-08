---
layout: post
title:  "Principal Components Analysis"
date: 2016-05-07 19:13:09
categories: Machine&nbsp;Learning
comments: true
---

* content
{:toc}

假设我们有$$m$$个$$n$$维数据点，$$\{x^{(1)},x^{(2)},...,x^{(m)}\}$$

因为一些原因，现在需要对其进行有损压缩。

对每个$$n$$维数据点$$x^{(i)}$$，在$$l$$维空间$$(l \leq n)$$中找到一个数据点$$c^{(i)}$$作为其压缩结果

将$$x$$压缩为$$c$$的过程称为编码，从$$c$$重新构造出$$x$$的过程称为解码

因为两个过程实际上都能用矩阵乘法表示，我们将解码过程表示为

\begin{align}
x=Dc
\end{align}

上式中，$$D$$是一个$$n$$行$$l$$列的矩阵

* 由于放大$$D$$的值，缩小$$c$$的值，上式仍然成立，为了使结果唯一，令$$D$$的所有列向量均为单位向量  
* 为了使解码空间（$$D$$的列空间）能够覆盖整个$$n$$维空间，令$$D$$的$$n$$个列向量互相正交

上两个约束的公式化表述是$$D^TD=I_l$$

由解码过程不难推出编码过程就是

\begin{align}
c=D^Tx
\end{align}

经过编码再解码的结果$$r(x)=DD^Tx$$

我们的优化目标就是最小化$$r(x)$$与$$x$$的差距，最优值$$D^*$$的公式化表述如下

\begin{align}
& D^* = \mathop{argmin}\_D\sum\_{i}||x^{(i)}-DD^Tx^{(i)}||^2_2 \\\
& s.t. \quad D^TD=I_l
\end{align}

$$X$$为一个$$m$$行$$n$$列的矩阵，第$$i$$个行向量表示$$(x^{(i)})^T$$

使用$$X$$，式子可化简为

\begin{align}
& D^* = \mathop{argmin}\_D||X-XDD^T||^2_F \\\
& s.t. \quad D^TD=I_l
\end{align}

F表示Frobenius范数  

继续化简

\begin{align}
D^* &= \mathop{argmin}\_D||X-XDD^T||^2_F \\\
&= \mathop{argmin}\_DTr\left\((X-XDD^T)^T(X-XDD^T)\right\) \\\
&= \mathop{argmin}\_DTr\left\((X^T-DD^TX^T)(X-XDD^T)\right\) \\\
&= \mathop{argmin}\_DTr\left\(  X^TX-X^TXDD^T-DD^TX^TX+DD^TX^TXDD^T \right\) \\\
&= \mathop{argmin}\_DTr\left\(  -X^TXDD^T-DD^TX^TX+DD^TX^TXDD^T \right\) \\\
&= \mathop{argmin}\_D  -Tr(X^TXDD^T)-Tr(DD^TX^TX)+Tr(DD^TX^TXDD^T)  \\\
&= \mathop{argmin}\_D  -2Tr(X^TXDD^T)+Tr(X^TXDD^TDD^T)  \\\
&= \mathop{argmin}\_D  -2Tr(X^TXDD^T)+Tr(X^TXDD^T)  \\\
&= \mathop{argmin}\_D  -Tr(X^TXDD^T)  \\\
&= \mathop{argmax}\_D  Tr(D^TX^TXD)  \\\
\end{align}

于是，优化目标变为

\begin{align}
& D^* = \mathop{argmax}\_D  Tr(D^TX^TXD) \\\
& s.t. \quad D^TD=I_l
\end{align}

将$$X^TX$$特征分解，有

\begin{align}
X^TX= V \Lambda V^{-1}
\end{align}

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

$$v_i$$表示特征向量，且保证是单位向量，此外$$\lambda_1 \geq \lambda_2 \geq ... \geq \lambda_n $$

优化目标变为

\begin{align}
& D^* = \mathop{argmax}\_D  Tr(D^TV\Lambda V^{-1}D) \\\
& s.t. \quad D^TD=I_l
\end{align}

最优解可直接得到

\begin{align}
D = \begin{bmatrix} v_1 & v_2 & ... & v_l \end{bmatrix}
\end{align}