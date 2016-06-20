---
layout: post
title:  "Naive Bayes Classifier"
date: 2016-06-19 15:11:28 
categories: Machine&nbsp;Learning
comments: true
---

面临一个分类问题，单个样本表示为$$x = (x_1,x_2,...,x_n)$$，意思是n个特征

而其类别$$y$$有$$K$$个可取值

根据贝叶斯定理

\begin{align}
p(y|x) = \frac{p(y)p(x|y)}{p(x)}
\end{align}

由于分母和$$y$$无关，因此

\begin{align}
p(y|x) &\propto p(y)p(x|y) \\\
&= p(y)p(x_n|y)p(x_{n-1}|y,x_n)...p(x_1|y,x_2,x_3,...,x_n)
\end{align}

在朴素贝叶斯中，认为各个特征互相独立，因此

\begin{align}
p(y|x) &\propto p(y)\prod_{i=1}^np(x_i|y)
\end{align}

也就是说

\begin{align}
p(y|x) = \frac{1}{Z}p(y)\prod_{i=1}^np(x_i|y)
\end{align}

$$Z$$用于保证

\begin{align}
\sum_y p(y | x)=1
\end{align}

在$$x$$确定的时候是个常量

对于$$x$$，其最终预测的类别$$\hat{y}$$为

\begin{align}
\hat{y} = \mathop{argmax}\_y p(y)\prod\_{i=1}^np(x\_i|y)
\end{align}

其实就是MAP
