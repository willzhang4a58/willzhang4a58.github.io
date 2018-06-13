---
layout: post
title:  "Spin Mutex implemented with C++11"
date: 2018-06-12 13:32:00 
categories: Multithread&nbsp;Programming
comments: true
---

```cpp
class SpinMutex final {
 public:
  SpinMutex(const SpinMutex&) = delete;
  SpinMutex(SpinMutex&&) = delete;
  SpinMutex& operator = (const SpinMutex&) = delete;
  SpinMutex& operator = (SpinMutex&&) = delete;

  SpinMutex() : flag_(ATOMIC_FLAG_INIT) {}
  ~SpinMutex() {
    assert(flag_.test_and_set() == false);
  }

  void lock() {
    while (flag_.test_and_set(std::memory_order_acquire)) {
    }
  }

  void unlock() {
    flag_.clear(std::memory_order_release);
  }

 private:
  std::atomic_flag flag_;
};
```
