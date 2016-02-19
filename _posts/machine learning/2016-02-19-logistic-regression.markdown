---
layout: post
title:  "Logistic Regression"
date: "2016-02-19 21:47:24"
categories: Machine&nbsp;Learning
comments: true
---

* content
{:toc}

## 1. 前言

Logistic Regression是业界最常见的分类算法之一。因其原理简单，且效果不错，被大量地应用于实际业务中。

## 2. 符号定义

设有如下m个数据点
\begin{align}
{(x^{(1)}, y^{(1)}), (x^{(2)}, y^{(2)}), ..., (x^{(m)}, y^{(m)})}
\end{align}

每个数据点包含特征x和类别y，其中
\begin{align}
x^{(i)}&=[x^{(i)}_1, x^{(i)}_2, ..., x^{(i)}_n]^T \\\
y^{(i)}& \in \\{0, 1\\}
\end{align}

## 3. 优化目标

Logistic Regression的目标是寻找一个函数完成从x到y的映射，如下
\begin{align}
h(x) = sigmoid(w^Tx+b)
\end{align}

w和b为Logistic Regression的参数，通过调整这两项参数使得预测的类别h(x)更贴近真实的类别的y<br>
通常，对h(x)的概率解释是x对应的y为1的概率，因此1 - h(x)也就是对应的y为0的概率<br>
因此，其似然函数为

\begin{align}
L(w,b)=\prod_{i=1}^m{h(x^{(i)})}^{y^{(i)}}(1-h(x^{(i)}))^{1-y^{(i)}}
\end{align}

Logistic Regression的优化目标就是似然函数最大化

实际应用中通常对似然函数取对数

\begin{align}
\ln L(w,b)&=\sum_{i=1}^my^{(i)}\ln {h(x^{(i)})} + (1-y^{(i)})ln {(1-h(x^{(i)}))}
\end{align}

再乘-1则为最后的Cost Function

\begin{align}
Cost&=\sum_{i=1}^m-y^{(i)}\ln {h(x^{(i)})} - (1-y^{(i)})ln {(1-h(x^{(i)}))}
\end{align}


## 4. 梯度

无论是梯度下降，还是牛顿法，都需要计算梯度，因此本文也求解一下<br>


求梯度

\begin{align}
\frac{\partial Cost}{\partial h(x^{(i)})} &= -\frac{y^{(i)}}{h(x^{(i)})} - \frac{(1-y^{(i)})}{h(x^{(i)}) - 1} \\\
\frac{\partial h(x^{(i)})}{\partial w} &= sigmoid'(w^Tx^{(i)}+b)x^{(i)} = h(x^{(i)})(1-h(x^{(i)}))x^{(i)}\\\
\frac{\partial h(x^{(i)})}{\partial b} &= sigmoid'(w^Tx^{(i)}+b) = h(x^{(i)})(1-h(x^{(i)})) \\\
\frac{\partial Cost}{\partial w} &= \sum_{i=1}^m\frac{\partial Cost}{\partial h(x^{(i)})}\frac{\partial h(x^{(i)})}{\partial w} = \sum_{i=1}^m  y^{(i)}(h(x^{(i)}) -1)x^{(i)} + (1-y^{(i)})  h(x^{(i)})x^{(i)} = \sum_{i=1}^m (h(x^{(i)})-y^{(i)})x^{(i)}  \\\
\frac{\partial Cost}{\partial b} &= \sum_{i=1}^m\frac{\partial Cost}{\partial h(x^{(i)})}\frac{\partial h(x^{(i)})}{\partial b} = \sum_{i=1}^m  y^{(i)}(h(x^{(i)}) -1) + (1-y^{(i)})  h(x^{(i)}) = \sum_{i=1}^m h(x^{(i)})-y^{(i)}
\end{align} 


## 5. Reference

* [Wiki——Logistic Regression]

[Wiki——Logistic Regression]: https://en.wikipedia.org/wiki/Logistic_regression

