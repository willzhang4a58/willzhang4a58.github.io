---
layout: post
title:  "Memory Order and Operations on Atomic Types"
date: 2018-06-11 23:15:00 
categories: Multithread&nbsp;Programming
comments: true
---

* content
{:toc}

## 1. Atomic Operation

c++11定义了原子类型，可以支持多个线程的同时读写。

原子类型支持的几个主要原子操作如下

Name| Explanation
:--|:--
store| 设置值
load|获取值
exchange|设置新值，获取旧值
compare_exchange_strong |如果与期望值相等（bitwise），设置新值，否则，获取旧值
compare_exchange_weak|相比strong，可能会fail spuriously（即便与期望值相等，却认为不相等），但是在某些平台上可能会更快

这里的原子操作是指，在软件层面看来，这个操作，要么已经执行完毕，要么没有执行，不可能执行到一半。

## 2. Memory Order

在标准里定义的原子操作，基本都会有一个memory order的参数，为什么需要这个参数

因为在没有数据依赖的情况下，编译器和CPU都会对指令重新排序，以获得更好的性能。

也就是说在下面这么简单的代码里（我们假设每一行代码都是原子操作）

```cpp
a = 100;
b = 0;
```

实际执行却可能先对b赋值，再对a赋值

在多线程的场景下，如果仅靠原子操作，这种重排会导致bug，比如

```cpp
std::atomic<int32_t> a(0);
std::atomic<int32_t> b(0);
auto set_a100_and_b1 = std::thread([&]() {
  a.store(100, std::memory_order_relaxed); // 1
  b.store(1, std::memory_order_relaxed); // 2
});
auto if_b1_then_read_a = std::thread([&]() {
  while (b.load(std::memory_order_relaxed) == 0) {
  }
  assert(a.load(std::memory_order_relaxed) == 100); // 3
});
```

memory_order_relaxed表示对重排不做限制，因此实际运行时只保证了原子操作

假如没有重排，可以认为，执行顺序一定会是`1->2->3`，不会有问题

考虑到重排后，执行顺序可能会变为`2->3->1`，从而触发断言错误

标准里定义了如下的memory_order

Name| Explanation
:--|:--
memory_order_relaxed|无限制
memory_order_consume|当前线程依赖此原子变量的读写操作不可以重排到本次操作前面
memory_order_acquire|当前线程的读写操作不可以重排到本次操作前面
memory_order_release|当前线程的读写操作不可以重排到本次操作后面
memory_order_acq_rel|acquire + release
memory_order_seq_cst|acq_rel, 且所有seq_cst操作有一个全序关系
