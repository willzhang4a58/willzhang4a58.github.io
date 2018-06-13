---
layout: post
title:  "Newton Method"
date: 2016-05-14 21:15:29
categories: Machine&nbsp;Learning
comments: true
---

* content
{:toc}

## 1. 牛顿法

考虑问题$$f(x) = 0$$

迭代法，选择一个接近$$f(x)$$零点的$$x^0$$，计算相应的$$f(x^0)$$和切线斜率$$f^{'}(x^0)$$

计算穿过$$(x^0,f(x^0))$$的切线和x轴的交点的x坐标，记为$$x^1$$

\begin{align}
x^1 = x^0 - \frac{f(x^0)}{f^{'}(x^0)}
\end{align}

通常$$x^1$$会比$$x^0$$更接近解，使用$$x^1$$进行下一轮迭代，迭代公式为

\begin{align}
x^{m+1} = x^{m} - \frac{f(x^m)}{f^{'}(x^m)}
\end{align}

设某个机器学习问题所求解的cost function为f(x)，那么其最优解一般满足$$f^{'}(x) = 0$$

使用牛顿法，迭代式子为

\begin{align}x^{m+1} = x^{m} - \frac{f^{'}(x^m)}{f^{''}(x^m)}\end{align}

将上式推广，有

\begin{align}
x^{m+1} = x^{m} - {[Hf(x^m)]}^{-1}\nabla f(x^m)
\end{align}

H为Hessian矩阵，下面谈一下上式的收敛条件

\begin{align}
- {[Hf(x^m)]}^{-1}\nabla f(x^m) = x^{m+1} - x^{m}  \Longrightarrow \nabla f(x^m) = -\[Hf(x^m)\](x^{m+1} - x^{m})
\end{align}

且

\begin{align}
(x^{m+1} - x^{m})^T\nabla f(x^m) = -(x^{m+1} - x^{m})^T\[Hf(x^m)\](x^{m+1} - x^{m})
\end{align}

当满足$$(x^{m+1} - x^{m})^T\nabla f(x^m) < 0$$时，可保证cost function在下降

因此必须使得$$ (x^{m+1} - x^{m})^T[Hf(x^m)](x^{m+1} - x^{m}) > 0$$

所以可以得到结论，当Hessian矩阵正定时，可保证收敛

步骤总结

1. 给定初值$$x^0$$，令$$m = 0$$
2. 计算$$\nabla f(x^m)$$和$$Hf(x^m)$$
3. 确定下降方向$$d^m = - {[Hf(x^m)]}^{-1}\nabla f(x^m)$$
4. 计算新的x值，$$x^{m+1}=x^m+d^m$$
5. 令$$m =m+ 1$$，回到第2步

## 2. 阻尼牛顿法

1. 给定初值$$x^0$$，令$$m = 0$$
2. 计算$$\nabla f(x^m)$$和$$Hf(x^m)$$
3. 确定下降方向$$d^m = - {[Hf(x^m)]}^{-1}\nabla f(x^m)$$
4. 计算步长$$\lambda^m = {argmin_\lambda} f(x^m + \lambda d^m)$$，计算新的x值，$$x^{m+1}=x^m+\lambda^m d^m$$
5. 令$$m =m+ 1$$，回到第2步


## 3. 拟牛顿条件

回忆牛顿法的迭代式子

\begin{align}
x^{m+1} = x^{m} - {[Hf(x^m)]}^{-1}\nabla f(x^m)
\end{align}

拟牛顿法的基本思想是构造一个可以近似Hessian矩阵或者其逆矩阵的正定矩阵

将$$f(x)$$在$$x^{m+1}$$处做泰勒展开，得到

\begin{align}
f(x) \approx f(x^{m+1}) + \nabla f(x^{m+1})(x-x^{m+1}) + \frac{1}{2}\nabla^2 f(x^{m+1})(x-x^{m+1})^2
\end{align}

计算其梯度

\begin{align}
\nabla f(x) \approx \nabla f(x^{m+1}) + \[Hf(x^{m+1})\](x - x^{m+1})
\end{align}

为了方便，用符号$$H^{m+1}$$表示$$[Hf(x^{m+1})]$$

于是

\begin{align}
\nabla f(x) \approx \nabla f(x^{m+1}) + H^{m+1}(x - x^{m+1})
\end{align}

取$$x=x^m$$，有
\begin{align}
\nabla f(x^m) \approx \nabla f(x^{m+1}) + H^{m+1}(x^m - x^{m+1})
\end{align}

整理，得

\begin{align}
\nabla f(x^{m+1}) -\nabla f(x^{m})\approx  H^{m+1}(x^{m+1} - x^{m})
\end{align}

迭代过程中的Hessian矩阵受上式约束，上式也称为拟牛顿条件

定义符号B为拟牛顿法对Hessian矩阵的近似矩阵
定义符号D为拟牛顿法对Hessian矩阵的逆的近似矩阵

## 4. DFP算法

DFP算法的基本思想是通过迭代的方法，近似Hessian矩阵的逆，迭代式为

\begin{align}
D^{m+1} = D^m + \Delta D^m
\end{align}

其中，$$D_0$$一般取单位矩阵

将$$\Delta D^m$$定义为

\begin{align}
\Delta D^m = \alpha uu^T + \beta vv^T
\end{align}

式中$$\alpha$$和$$\beta$$为待定一维向量，$$u$$和$$v$$为待定n维向量，有

\begin{align}
D^{m+1} = D^m + \alpha uu^T + \beta vv^T
\end{align}
上式乘$$y^m = \nabla f(x^{m+1}) -\nabla f(x^{m})$$，结合拟牛顿条件有

\begin{align}
x^{m+1} - x^{m} = D^my^m + \alpha uu^Ty^m + \beta vv^Ty^m
\end{align}

变换形式，有

\begin{align}
x^{m+1} - x^{m} = D^my^m + (\alpha u^Ty^m)u + (\beta v^Ty^m)v
\end{align}

令$$\alpha = \frac{1}{u^Ty^m}$$，$$\beta = \frac{-1}{v^Ty^m}$$，得到

\begin{align}
x^{m+1} - x^{m} -D^my^m=   u - v
\end{align}

为使上式成立，令$$u=x^{m+1} - x^{m}$$，$$v=D^my^m$$，因此

\begin{align}
& \alpha = \frac{1}{u^Ty^m}=\frac{1}{(x^{m+1} - x^{m})^Ty^m} \\\
& \beta = \frac{-1}{v^Ty^m} = \frac{-1}{(y^m)^TD^my^m}
\end{align}

最终，得到

\begin{align}
\Delta D^m &= \alpha uu^T + \beta vv^T \\\
&= \frac{(x^{m+1} - x^{m})(x^{m+1} - x^{m})^T}{(x^{m+1} - x^{m})^Ty^m} - \frac{D^my^m(y^m)^TD^m}{(y^m)^TD^my^m}
\end{align}

步骤总结

1.  给定初值$$x^0$$，令$$D^0 = I$$，$$m=0$$
2. 计算下降方向$$d^m=- {[Hf(x^m)]}^{-1}\nabla f(x^m)$$
3. 计算步长$$\lambda^m = {argmin_\lambda} f(x^m + \lambda d^m)$$
4. 计算新的x，$$x^{m+1} = x^m + \lambda^md^m$$
5. 计算$$\nabla f(x^{m+1})$$
6. 计算$$y^m = \nabla f(x^{m+1}) -\nabla f(x^{m})$$
7. 计算$$D^{m+1} = D^m + \Delta D^m$$
8. 令$$m=m+1$$，回到第2步

## 5. BFGS算法

BFGS算法的基本思想是通过迭代的方法，近似Hessian矩阵，迭代式为
\begin{align}
B^{m+1} = B^m + \Delta B^m
\end{align}

$$B^0$$一般取单位矩阵，与DFP类似
\begin{align}\Delta B^m = \alpha uu^T + \beta vv^T\end{align}

因此

\begin{align}
B^{m+1} & = B^m + \Delta B^m \\\
& = B^m + \alpha uu^T + \beta vv^T \\\
\end{align}

上式乘$$s^m = (x^{m+1} - x^{m})$$，结合拟牛顿条件有

\begin{align}
\nabla f(x^{m+1}) -\nabla f(x^{m}) & = B^ms^m + \alpha uu^Ts^m + \beta vv^Ts^m \\
\end{align}

变换形式
\begin{align}
\nabla f(x^{m+1}) -\nabla f(x^{m}) & = B^ms^m + (\alpha u^Ts^m)u + (\beta v^Ts^m)v \\
\end{align}

令$$\alpha = \frac{1}{u^Ts^m}$$，$$\beta = \frac{-1}{v^Ts^m}$$，有

\begin{align}
\nabla f(x^{m+1}) -\nabla f(x^{m}) -B^ms^m =  u - v
\end{align}

为使上式成立，令

\begin{align}
u &= \nabla f(x^{m+1}) -\nabla f(x^{m}) \\\
v &= B^ms^m
\end{align}

综上
\begin{align}
\Delta B^m &= \alpha uu^T + \beta vv^T \\\
&=\frac{uu^T}{u^Ts^m} - \frac{vv^T}{v^Ts^m} \\\
&=\frac{uu^T}{u^T(x^{m+1} - x^{m})} - \frac{B^ms^m(B^ms^m)^T}{(B^ms^m)^T(x^{m+1} - x^{m})} \\\
&=\frac{uu^T}{u^T(x^{m+1} - x^{m})} - \frac{B^m(x^{m+1} - x^{m})(x^{m+1} - x^{m})^TB^m}{(x^{m+1} - x^{m})^TB^m(x^{m+1} - x^{m})}
\end{align}

令$$y^m = \nabla f(x^{m+1}) -\nabla f(x^{m})$$，且有$$s^m = (x^{m+1} - x^{m})$$，上式简化为

\begin{align}
\Delta B^m = \frac{y^m(y^m)^T}{(y^m)^Ts^m} - \frac{B^ms^m(s^m)^TB^m}{(s^m)^TB^ms^m}
\end{align}

因此

\begin{align}
B^{m+1} = B^m +  \frac{y^m(y^m)^T}{(y^m)^Ts^m} - \frac{B^ms^m(s^m)^TB^m}{(s^m)^TB^ms^m}
\end{align}

对上式求逆，得到

\begin{align}
D^{m+1} = (I - \frac{s^m(y^m)^T}{(y^m)^Ts^m})D^m(I - \frac{y^m(s^m)^T}{(y^m)^Ts^m})+\frac{s^m(s^m)^T}{(y^m)^Ts^m}
\end{align}

步骤总结

1. 给定初值$$x^0$$，令$$D^0=I$$，$$m=0$$
2. 计算下降方向$$d^m = - D^m\nabla f(x^m)$$
3. 计算步长$$\lambda^m = {argmin_\lambda} f(x^m + \lambda d^m)$$
4. 计算新的x，$$x^{m+1} = x^m + \lambda^md^m$$
5. 计算$$\nabla f(x^{m+1})$$
6. 计算$$y^m = \nabla f(x^{m+1}) -\nabla f(x^{m})$$
7. 计算$$D^{m+1} = (I - \frac{s^m(y^m)^T}{(y^m)^Ts^m})D^m(I - \frac{y^m(s^m)^T}{(y^m)^Ts^m})+\frac{s^m(s^m)^T}{(y^m)^Ts^m}$$
8. 令$$m=m+1$$，回到第2步

## 6. L-BFGS

BFGS的Hessian矩阵在数据维度较高时会使用极大的内存

L-BFGS在BFGS的基础上做了近似，解决了这个问题

首先回忆BFGS的Hessian矩阵迭代式

\begin{align}
D^{m+1} = (I - \frac{s^m(y^m)^T}{(y^m)^Ts^m})D^m(I - \frac{y^m(s^m)^T}{(y^m)^Ts^m})+\frac{s^m(s^m)^T}{(y^m)^Ts^m}
\end{align}

式中
\begin{align}
& s^m = (x^{m+1} - x^{m}) \\\
& y^m = \nabla f(x^{m+1}) -\nabla f(x^{m})
\end{align}

定义符号$$\rho^m=\frac{1}{(y^m)^Ts^m}$$，$$V^m=I - \rho^my^m(s^m)^T$$
Hessian矩阵迭代式为

\begin{align}
D^{m+1} = (V^m)^TD^mV^m+\rho^ms^m(s^m)^T
\end{align}

由于$$D^0$$通常为单位矩阵，因此有

\begin{align}
D^1 &= (V^0)^TD^0V^0+\rho^0s^0(s^0)^T \\\
D^2 &= (V^1)^TD^1V^1+\rho^1s^1(s^1)^T \\\
&= (V^1)^T[(V^0)^TD^0V^0+\rho^0s^0(s^0)^T]V^1+\rho^1s^1(s^1)^T\\\
&= (V^1)^T(V^0)^TD^0V^0V^1+(V^1)^T\rho^0s^0(s^0)^TV^1+\rho^1s^1(s^1)^T\\\
D^3 &= (V^2)^TD^2V^2+\rho^2s^2(s^2)^T \\\
&= (V^2)^T(V^1)^T(V^0)^TD^0V^0V^1V^2+(V^2)^T(V^1)^T\rho^0s^0(s^0)^TV^1V^2+(V^2)^T\rho^1s^1(s^1)^TV^2+\rho^2s^2(s^2)^T
\end{align}

一般式

\begin{align}
D^{m+1}=& [(V^m)^T(V^{m-1})^T...(V^0)^T]D^0(V^0V^1...V^m) \\\
+&[(V^m)^T(V^{m-1})^T...(V^1)^T]\rho^0s^0(s^0)^T(V^1V^2...V^m) \\\
+&[(V^m)^T(V^{m-1})^T...(V^2)^T]\rho^1s^1(s^1)^T(V^2V^3...V^m) \\\
\+ &... \\\
\+ &(V^m)^T\rho^{m-1}s^{m-1}(s^{m-1})^TV^m \\\
\+ &\rho^{m}s^{m}(s^{m})^T
\end{align}

由上式，计算$$D^{m+1}$$需要用到$$\{s^i,y^i\}_{i=0}^m$$

因此，在计算过程中不存储Hessian矩阵，而是连续地存储k组$$\{s^i, y^i\}$$，就能够精确地计算得到$$D^1,D^2,...,D^{k}$$

对于$$D^{k+1}$$开始的计算就需要近似了，若要精确地计算$$D^{k+1}$$，我们需要的信息是$$\{s^i,y^i\}_{i=0}^k$$，一共k+1组数据，但只能保存k组数据，因此在进行$$D^{k+1}$$的计算之前需要抛弃一组，因此舍弃最早的一组向量$$\{s^0, y^0\}$$，也就是说，在计算$$D^{k+1}$$时使用的信息为$$\{s^i,y^i\}_{i=1}^k$$，以此类推，计算$$D^{k+2}$$时使用的信息为$$\{s^i,y^i\}_{i=2}^{k+1}$$

得到近似计算式

\begin{align}
D^{m+1}=& [(V^m)^T(V^{m-1})^T...(V^{m-k+1})^T]D^0(V^{m-k+1}V^{m-k+2}...V^m) \\\
+&[(V^m)^T(V^{m-1})^T...(V^{m-k+2})^T]\rho^0s^0(s^0)^T(V^{m-k+2}V^{m-k+3}...V^m) \\\
+&[(V^m)^T(V^{m-1})^T...(V^{m-k+3})^T]\rho^1s^1(s^1)^T(V^{m-k+3}V^{m-k+4}...V^m) \\\
\+ &... \\\
\+ &(V^m)^T\rho^{m-1}s^{m-1}(s^{m-1})^TV^m \\\
\+ &\rho^{m}s^{m}(s^{m})^T
\end{align}

由于Hessian矩阵的作用是计算下降方向$$d^m = - D^m\nabla f(x^m)$$

\begin{align}
D^{m}=& [(V^{m-1})^T(V^{m-2})^T...(V^{m-k})^T]D^0(V^{m-k}V^{m-k+1}...V^{m-1}) \\\
+&[(V^{m-1})^T(V^{m-2})^T...(V^{m-k+1})^T]\rho^0s^0(s^0)^T(V^{m-k+1}V^{m-k+2}...V^{m-1}) \\\
+&[(V^{m-1})^T(V^{m-2})^T...(V^{m-k+2})^T]\rho^1s^1(s^1)^T(V^{m-k+2}V^{m-k+3}...V^{m-1}) \\\
\+ &... \\\
\+ &(V^{m-1})^T\rho^{m-2}s^{m-2}(s^{m-2})^TV^{m-1} \\\
\+ &\rho^{m-1}s^{m-1}(s^{m-1})^T
\end{align}

而近似计算式中的V为n*n矩阵，直接使用近似计算式并不能达到减少内存的目的，因此有一个计算的trick，在不直接计算V的情况下得到下降方向
计算下降方向的伪代码为

\begin{align}
& q = \nabla f(x^m) \\\
& for \quad i=m-1,m-2,...,m-k \\\
& \quad\quad \alpha[i]=\rho^is^iq \\\
& \quad\quad q=q-\alpha[i]y^i \\\
& z=\frac{(y^{m-1})^Ts^{m-1}}{(y^{m-1})^Ty^{m-1}}q \\\
& for \quad i = m-k,m-k+1,...,m-1 \\\
& \quad\quad \beta[i]=\rho^i(y^i)^Tz \\\
& \quad\quad z = z + s^i(\alpha[i]-\beta[i]) \\\
& d^m=-z
\end{align}

下文称k组数据为信息队列
步骤总结

1. 给定初值$$x^0$$，令$$D^0=I$$，$$m=0$$，信息队列初始化为空
2. 根据信息队列计算下降方向$$d^m$$
3. 计算步长$$\lambda^m=argmin_\lambda f(x^m+\lambda d^m)$$
4. 计算$$x^{m+1}=x^m+\lambda^m d^m$$
5. 计算$$s^m$$，$$y^m$$
6. 更新信息队列
7. 令$$m=m+1$$，回到第2步


## 7. OWLQN 

由于L1正则在0处不可微，因此在使用L1正则的情况下不能直接使用L-BFGS算法。

OWLQN便是解决这个问题的一种方法。

定义sign函数$$\sigma$$

\begin{align}
\sigma(x)=\left\\{
\begin{aligned}
-1 &\quad if \;  x < 0\\\
0 &\quad if \; x=0\\\
1 & \quad if \; x > 0
\end{aligned}
\right.
\end{align}

定义$$\pi$$函数，该函数有一个n维向量y作为参数

\begin{align}
\pi_i(x;y)=\left\\{\begin{aligned}
& x_i \quad if \; \sigma(x_i) = \sigma(y_i) \\\
& 0 \quad otherwise
\end{aligned}\right.
\end{align}

回到优化目标

\begin{align}
f(x) = l(x) + C||x||_1
\end{align}

$$l(x)$$为损失函数，C为$$L1$$系数

对于向量$$\xi\in\{-1,0,1\}^n$$，定义

\begin{align}
\Omega_\xi=\{x\in R^n|\pi(x;\xi)=x\}
\end{align}

对于任意$$x\in \Omega_\xi$$，有

\begin{align}
f(x)=l(x) + C\xi^Tx
\end{align}

这是一个可使用L-BFGS求解的函数

因此，OWLQN的基本思想就是每一次迭代均不跨越象限，这样就能够使用L-BFGS

定义伪梯度$$\Gamma f(x)$$

\begin{align}
\Gamma_if(x)=\left\\{\begin{aligned}
& {\partial_i}^{-}f(x) & if \; {\partial_i}^-f(x)>0\\\
& {\partial_i}^+f(x)  & if \; {\partial_i}^+f(x)<0\\\
& 0 & otherwise
\end{aligned}\right.
\end{align}

式中

\begin{align}
 {\partial_i}^{\pm}f(x) = \frac{\partial l(x)}{\partial x_i} + \left\\{\begin{aligned}
 & C\sigma(x_i)  \quad & if \; x_i \neq 0\\\
 & \pm C \quad &if \; x_i = 0
 \end{aligned}\right.
\end{align}

使用伪梯度作为梯度进行计算

伪梯度的合理性参考下链接的图5，6，7
http://www.cnblogs.com/vivounicorn/archive/2012/06/25/2561071.html 

为了保证一轮迭代后x不跨越象限，还需要

\begin{align}
x^{k+1}=\pi(x^{k+1};\xi^k)
\end{align}

步骤总结

1. 给定初值$$x^0$$，令$$D^0=I$$，$$k=0$$，信息队列初始化为空
2. 计算梯度下降方向$$v^k=-\Gamma f(x^k)$$
3. 通过信息队列，计算L-BFGS的方向$$d^k=Dv^k$$
4. 这一步是个trick，$$d^k=\pi(d^k;v^k)$$
5. 计算步长$$\lambda^k=argmin_\lambda f(x^k+\lambda d^k)$$
6. 计算$$x^{k+1}=\pi(x^k+\lambda^k d^k;x^k)$$
7. 计算$$s^k$$，$$y^k$$
8. 更新信息队列
9. 令$$k=k+1$$，回到第2步
