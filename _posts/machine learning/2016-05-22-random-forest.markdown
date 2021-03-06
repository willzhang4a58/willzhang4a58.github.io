---
layout: post
title:  "Bagging and Random Forest"
date: 2016-05-21 21:16:01 
categories: Machine&nbsp;Learning
comments: true
---

* content
{:toc}

## 1. Bagging

首先我们有m条训练数据集

对这m条数据进行放回式采样，采样m次得到的m条数据用来训练一个模型

重复t次（模型可能使用不同算法产生），就能得到t个模型

对于单个样本，将t个模型对其的预测结果通过某种策略合并产生最终的预测结果

需要注意到，放回式采样，会导致对于每个模型，其实都会有约36.8%的数据（称为“包外样本”）并没有参与训练，我们可以用这部分数据作为验证数据集

## 2. Random Forest

随机森林是Bagging的一个扩展

其t个模型都使用决策树，并且对决策树的训练过程做了一些修改

普通的决策树模型在节点分裂是会选择一个最优的特征进行分裂

假设当前分裂节点有n个特征可以选择

随机森林首先会随机出大小为k（$$k\leq n$$）的特征子集，随后再从中选择最优的特征进行分裂

经验值$$k=\log_2n$$

Bagging策略中的包外样本还可用于对决策树进行剪枝