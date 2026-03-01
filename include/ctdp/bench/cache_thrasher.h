#ifndef CTDP_BENCH_CACHE_THRASHER_H
#define CTDP_BENCH_CACHE_THRASHER_H

// ctdp::bench::cache_thrasher — Standalone LLC eviction primitive
//
// Allocates a buffer 1.5× LLC size, walks it with a prime stride,
// and uses DoNotOptimize to prevent elision. This forces all cache
// lines to be evicted before the next measurement iteration.

#include "compiler_barrier.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace ctdp::bench {

/// LLC cache eviction primitive.
/// Construct with the LLC size in bytes (or 0 for a sensible default).
class cache_thrasher {
public:
    /// @param llc_bytes  LLC size in bytes (0 = use 8 MiB default)
    explicit cache_thrasher(std::size_t llc_bytes = 0) {
        if (llc_bytes == 0) llc_bytes = 8u * 1024u * 1024u; // 8 MiB default
        buffer_size_ = llc_bytes * 3 / 2; // 1.5× LLC
        buffer_.resize(buffer_size_ / sizeof(std::uint64_t), 0);
    }

    /// Walk the buffer with a prime stride, forcing LLC eviction.
    /// The result is consumed via DoNotOptimize to prevent elision.
    void thrash() noexcept {
        constexpr std::size_t kPrimeStride = 127; // cache lines
        constexpr std::size_t kWordsPerLine = 8;   // 64B / 8B
        std::size_t stride = kPrimeStride * kWordsPerLine;

        std::uint64_t acc = 0;
        std::size_t n = buffer_.size();
        for (std::size_t i = 0; i < n; i += stride) {
            acc += buffer_[i];
            buffer_[i] = acc; // write to dirty the line
        }
        // Second pass with offset for better coverage
        for (std::size_t i = stride / 2; i < n; i += stride) {
            acc ^= buffer_[i];
            buffer_[i] = acc;
        }
        DoNotOptimize(acc);
        ClobberMemory();
    }

    /// Buffer size in bytes
    [[nodiscard]] std::size_t buffer_bytes() const noexcept {
        return buffer_.size() * sizeof(std::uint64_t);
    }

private:
    std::size_t buffer_size_ = 0;
    std::vector<std::uint64_t> buffer_;
};

} // namespace ctdp::bench

#endif // CTDP_BENCH_CACHE_THRASHER_H
