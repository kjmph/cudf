/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <exception>
#include <functional>
#include <future>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace cudf::io::detail {

/**
 * @brief Owns asynchronous tasks and drains every valid future before destruction.
 *
 * This guard is intended for asynchronous operations that access caller-owned storage. A failure
 * while scheduling or consuming one task must not allow that storage to unwind while previously
 * issued tasks are still running.
 *
 * Explicit consumption preserves the first exception in task order, drains the remaining tasks,
 * then rethrows that exception. Destruction performs the same drain without throwing, which
 * preserves an exception that is already unwinding the stack.
 *
 * @tparam T Value returned by each task
 */
template <typename T>
class future_drain_guard {
 public:
  explicit future_drain_guard(std::size_t capacity) { tasks_.reserve(capacity); }

  ~future_drain_guard() noexcept { drain_noexcept(); }

  future_drain_guard(future_drain_guard const&)            = delete;
  future_drain_guard& operator=(future_drain_guard const&) = delete;

  future_drain_guard(future_drain_guard&& other) noexcept : tasks_{std::move(other.tasks_)} {}
  future_drain_guard& operator=(future_drain_guard&&) = delete;

  /**
   * @brief Adds a task to this guard.
   *
   * Callers should reserve the maximum task count in the constructor before issuing any task. A
   * `std::future` does not generally wait in its destructor, so allowing this operation to allocate
   * after a task has been issued would leave no way to drain that task if allocation failed.
   */
  void push(std::future<T>&& task)
  {
    // All current users reserve their maximum task count before issuing work. Keep this check here
    // so a future caller cannot silently weaken the lifetime guarantee.
    if (tasks_.size() == tasks_.capacity()) {
      try {
        task.get();
      } catch (...) {
        // The capacity error remains primary, but the untracked task must finish first.
      }
      throw std::logic_error("future_drain_guard capacity exhausted");
    }
    tasks_.push_back(std::move(task));
  }

  [[nodiscard]] bool empty() const noexcept { return tasks_.empty(); }
  [[nodiscard]] std::size_t size() const noexcept { return tasks_.size(); }

  /**
   * @brief Gets every task result and passes successful results to `consumer`.
   *
   * Exceptions from either a task or the consumer are remembered in task order. Every remaining
   * task is drained before the first exception is rethrown.
   */
  template <typename Consumer>
  void consume_all(Consumer&& consumer)
  {
    consume_all_indexed([&consumer](std::size_t, auto&&... values) {
      std::invoke(consumer, std::forward<decltype(values)>(values)...);
    });
  }

  /**
   * @brief Gets every task result and passes each successful result and task index to `consumer`.
   *
   * Unlike a successful-result counter, the index identifies the task's original position even when
   * an earlier task failed. Every remaining task is drained before the first exception is rethrown.
   */
  template <typename Consumer>
  void consume_all_indexed(Consumer&& consumer)
  {
    std::exception_ptr first_exception;

    for (std::size_t task_index = 0; task_index < tasks_.size(); ++task_index) {
      auto& task = tasks_[task_index];
      try {
        if constexpr (std::is_void_v<T>) {
          task.get();
          std::invoke(consumer, task_index);
        } else {
          std::invoke(consumer, task_index, task.get());
        }
      } catch (...) {
        if (first_exception == nullptr) { first_exception = std::current_exception(); }
      }
    }

    tasks_.clear();
    if (first_exception != nullptr) { std::rethrow_exception(first_exception); }
  }

  /**
   * @brief Gets every task result, discarding successful values.
   *
   * Every task is drained before the first exception in task order is rethrown.
   */
  void wait_all()
  {
    if constexpr (std::is_void_v<T>) {
      consume_all([] {});
    } else {
      consume_all([](T) {});
    }
  }

 private:
  void drain_noexcept() noexcept
  {
    try {
      wait_all();
    } catch (...) {
      // Destruction must preserve any primary exception that is already unwinding the stack.
    }
  }

  std::vector<std::future<T>> tasks_;
};

}  // namespace cudf::io::detail
