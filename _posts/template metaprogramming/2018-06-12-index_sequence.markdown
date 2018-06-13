---
layout: post
title:  "make_integer_sequence implemented with C++11"
date: 2018-06-12 14:48:00 
categories: Template&nbsp;Metaprogramming
comments: true
---

```cpp
#include <assert.h>
#include <atomic>
#include <thread>
#include <iostream>
#include <type_traits>

namespace std {

#if __cplusplus < 201402L

template<typename T, T... Ints>
struct integer_sequence {
  static_assert(is_integral<T>::value, "");
  using value_type = T;
  static constexpr size_t size() { return sizeof...(Ints); }
};

template<typename T, T index, typename U, T... Ints>
struct make_integer_sequence_impl;

template<typename T, T index, T... Ints>
struct make_integer_sequence_impl<T, index, typename enable_if<index != 0>::type, Ints...>
    : make_integer_sequence_impl<T, index - 1, void, index, Ints...> {};

template<typename T, T index, T... Ints>
struct make_integer_sequence_impl<T, index, typename enable_if<index == 0>::type, Ints...> {
  using type = integer_sequence<T, index, Ints...>;
};

template<typename T, T n>
using make_integer_sequence = typename make_integer_sequence_impl<T, n - 1, void>::type;

template<size_t... Ints>
using index_sequence = integer_sequence<size_t, Ints...>;

template<size_t n>
using make_index_sequence = make_integer_sequence<size_t, n>;

template<typename... T>
using index_sequence_for = make_index_sequence<sizeof...(T)>;

#endif

}

template<size_t head, size_t... tail_indices>
typename std::enable_if<sizeof... (tail_indices) == 0>::type PrintIndexSequenceImpl() {
  std::cout << head << std::endl;
}

template<size_t head, size_t... tail_indices>
typename std::enable_if<sizeof... (tail_indices) != 0>::type PrintIndexSequenceImpl() {
  std::cout << head << std::endl;
  PrintIndexSequenceImpl<tail_indices...>();
}

template<size_t... indices>
void PrintIndexSequence(std::index_sequence<indices...>) {
  PrintIndexSequenceImpl<indices...>();
}

int main() {
  PrintIndexSequence(std::make_index_sequence<100>());
  return 0;
}
```
