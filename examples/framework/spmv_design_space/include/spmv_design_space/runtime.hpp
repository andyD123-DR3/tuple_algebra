#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace ctdp::spmv_dsl {

// Small fixed-width value block modelled on numerics_base::part1::block.
// It is deliberately just std::array plus alignment: the demonstrator can
// express a SIMD-friendly lane policy without taking a dependency on ISA
// intrinsics.  The compiler is then free to auto-vectorise the lane loops.
template<class T, std::size_t W>
inline constexpr std::size_t fixed_block_alignment() noexcept {
    std::size_t a = sizeof(T);
    while (a < W * sizeof(T) && a < 64) {
        a <<= 1;
    }
    return a;
}

template<class T, std::size_t W>
struct alignas(fixed_block_alignment<T, W>()) fixed_block {
    std::array<T, W> v{};

    using element_type = T;
    static constexpr std::size_t width = W;

    constexpr T& operator[](std::size_t i) noexcept { return v[i]; }
    constexpr const T& operator[](std::size_t i) const noexcept { return v[i]; }
    constexpr T* data() noexcept { return v.data(); }
    constexpr const T* data() const noexcept { return v.data(); }
};

template<class T, std::size_t W>
constexpr fixed_block<T, W> fixed_broadcast(T x) noexcept {
    fixed_block<T, W> r{};
    for (std::size_t i = 0; i < W; ++i) {
        r[i] = x;
    }
    return r;
}

template<class T, std::size_t W, class Op>
inline fixed_block<T, W> fixed_binary(const fixed_block<T, W>& a,
                                      const fixed_block<T, W>& b,
                                      Op op) noexcept(noexcept(op(a[0], b[0]))) {
    fixed_block<T, W> r{};
    for (std::size_t i = 0; i < W; ++i) {
        r[i] = op(a[i], b[i]);
    }
    return r;
}

struct row_task {
    std::size_t first = 0; // index into a row-order vector, not necessarily a matrix row id
    std::size_t last = 0;
};

inline std::vector<row_task> make_row_tasks(std::size_t n, std::size_t grain) {
    if (grain == 0) {
        grain = 1;
    }
    std::vector<row_task> tasks;
    for (std::size_t first = 0; first < n; first += grain) {
        tasks.push_back({first, std::min(n, first + grain)});
    }
    return tasks;
}

// Persistent row-task pool following the shape used by the deterministic GEMM
// packed-panel examples: long-lived workers, epoch wake-up, atomic task cursor.
// The pool does not split a row. Each task owns a disjoint range of row-order
// entries, so writes to residual/z/x_next are non-overlapping.
class persistent_row_pool {
    std::vector<std::thread> workers_;
    std::mutex m_;
    std::condition_variable cv_start_;
    std::condition_variable cv_done_;
    bool stop_ = false;
    std::uint64_t epoch_ = 0;
    std::size_t remaining_ = 0;
    std::atomic<std::size_t> next_{0};
    const std::vector<row_task>* tasks_ = nullptr;
    std::function<void(std::size_t, std::size_t)> job_;

    void worker_loop() {
        std::uint64_t seen = 0;
        for (;;) {
            const std::vector<row_task>* local_tasks = nullptr;
            std::function<void(std::size_t, std::size_t)> local_job;
            {
                std::unique_lock<std::mutex> lock(m_);
                cv_start_.wait(lock, [&] { return stop_ || epoch_ != seen; });
                if (stop_) {
                    return;
                }
                seen = epoch_;
                local_tasks = tasks_;
                local_job = job_;
            }

            for (;;) {
                const std::size_t t = next_.fetch_add(1, std::memory_order_relaxed);
                if (t >= local_tasks->size()) {
                    break;
                }
                const auto task = (*local_tasks)[t];
                local_job(task.first, task.last);
            }

            {
                std::lock_guard<std::mutex> lock(m_);
                if (--remaining_ == 0) {
                    cv_done_.notify_one();
                }
            }
        }
    }

public:
    explicit persistent_row_pool(std::size_t n) {
        if (n == 0) {
            n = 1;
        }
        workers_.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
            workers_.emplace_back([this] { worker_loop(); });
        }
    }

    ~persistent_row_pool() {
        {
            std::lock_guard<std::mutex> lock(m_);
            stop_ = true;
            ++epoch_;
        }
        cv_start_.notify_all();
        for (auto& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    persistent_row_pool(const persistent_row_pool&) = delete;
    persistent_row_pool& operator=(const persistent_row_pool&) = delete;

    [[nodiscard]] std::size_t size() const noexcept { return workers_.size(); }

    template<class Fn>
    void run(const std::vector<row_task>& tasks, Fn&& fn) {
        if (tasks.empty()) {
            return;
        }
        if (workers_.size() == 1 || tasks.size() == 1) {
            for (const auto task : tasks) {
                fn(task.first, task.last);
            }
            return;
        }
        {
            std::lock_guard<std::mutex> lock(m_);
            tasks_ = &tasks;
            job_ = std::forward<Fn>(fn);
            remaining_ = workers_.size();
            next_.store(0, std::memory_order_relaxed);
            ++epoch_;
        }
        cv_start_.notify_all();
        {
            std::unique_lock<std::mutex> lock(m_);
            cv_done_.wait(lock, [&] { return remaining_ == 0; });
        }
    }
};

struct execution_context {
    persistent_row_pool* pool = nullptr;
    std::size_t task_grain = 64;
};

} // namespace ctdp::spmv_dsl
