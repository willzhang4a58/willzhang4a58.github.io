---
layout: post
title:  "Lock-Free Stack implemented with C++11"
date: 2018-06-17 12:14:00 
categories: Multithread&nbsp;Programming
comments: true
---

```cpp
template<typename T>
class LockFreeStack {
 public:
  LockFreeStack(const LockFreeStack&) = delete;
  LockFreeStack(LockFreeStack&&) = delete;
  LockFreeStack& operator = (const LockFreeStack&) = delete;
  LockFreeStack& operator = (LockFreeStack&&) = delete;

  LockFreeStack() = default;
  ~LockFreeStack() = default;

  void Push(const T& val) {
    auto new_head = std::make_shared<Node>();
    new_head->val = val;
    new_head->next = nullptr;
    while (std::atomic_compare_exchange_weak(&head_, &(new_head->next), new_head) == false) {
      // do nothing
    }
  }

  void Pop(T* val) {
    std::shared_ptr<Node> popped;
    while (!std::atomic_compare_exchange_weak(&head_, &popped, popped ? popped->next : nullptr)
        || popped == nullptr) {
      // do nothing
    }
    *val = popped->val;
  }

 private:
  struct Node {
    T val;
    std::shared_ptr<Node> next;
  };

  std::shared_ptr<Node> head_;

};
```
