#pragma once
// Experimental typed index geometry for executable Fourier factorisations.
// No arithmetic semantics are inferred from a coordinate map. The caller
// must still prove its index transformation and verify the full algorithm.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <vector>
#include <array>
#include <functional>
#include <ctdp/space/permutation.h>

namespace ctdp::space {

// Scatter/gather are non-in-place operations. A transpose or permuted
// factor index map may overwrite unread source elements, so even partial
// aliasing is refused; callers use explicit scratch and retain full
// numerical-observation qualification in their physical lowerers.
namespace physical_geometry_detail {
template<class T>
[[nodiscard]] inline bool disjoint(std::span<const T> a,
                                   std::span<T> b) noexcept {
    return a.empty() || b.empty() ||
        std::less_equal<const T*>{}(a.data()+a.size(),b.data()) ||
        std::less_equal<const T*>{}(b.data()+b.size(),a.data());
}
} // namespace physical_geometry_detail

struct shape2 {
    std::size_t outer{},inner{};
    [[nodiscard]] std::size_t elements() const {
        if(!outer || !inner || outer>std::numeric_limits<std::size_t>::max()/inner)
            throw std::invalid_argument("FFT geometry: invalid shape");
        return outer*inner;
    }
};
struct index2 { std::size_t outer{},inner{}; };
struct view2 {
    shape2 shape{};
    std::size_t outer_stride{},inner_stride{},base{},capacity{};

    static view2 make(shape2 s,std::size_t stride0,std::size_t stride1,
                      std::size_t storage,std::size_t offset=0) {
        const auto count=s.elements();
        if(count>1048576 || !stride0 || !stride1 || offset>=storage)
            throw std::invalid_argument("FFT geometry: invalid view");
        const auto limit=std::numeric_limits<std::size_t>::max();
        if(s.outer-1>(limit-offset)/stride0)
            throw std::invalid_argument("FFT geometry: outer stride overflow");
        const auto tail=offset+(s.outer-1)*stride0;
        if(s.inner-1>(limit-tail)/stride1 ||
           tail+(s.inner-1)*stride1>=storage)
            throw std::invalid_argument("FFT geometry: insufficient storage");
        // A nonzero collision obeys delta_outer*stride0 =
        // delta_inner*stride1. The smallest positive solution uses the gcd.
        const auto g=std::gcd(stride0,stride1);
        if(s.outer>stride1/g && s.inner>stride0/g)
            throw std::invalid_argument("FFT geometry: colliding affine view");
        return {s,stride0,stride1,offset,storage};
    }
    static view2 row_major(shape2 s) {
        const auto n=s.elements();
        return make(s,s.inner,1,n);
    }
    static view2 column_major(shape2 s) {
        const auto n=s.elements();
        return make(s,1,s.outer,n);
    }
    [[nodiscard]] std::size_t address(index2 index) const {
        if(index.outer>=shape.outer || index.inner>=shape.inner)
            throw std::out_of_range("FFT geometry: coordinate outside domain");
        return base+index.outer*outer_stride+index.inner*inner_stride;
    }
    [[nodiscard]] std::size_t address_linear(std::size_t linear) const {
        if(linear>=shape.elements())
            throw std::out_of_range("FFT geometry: linear index outside domain");
        return address({linear/shape.inner,linear%shape.inner});
    }
    template<class T>
    void scatter(std::span<const T> logical,std::span<T> storage) const {
        if(logical.size()!=shape.elements() || storage.size()<capacity ||
           !physical_geometry_detail::disjoint(logical,storage))
            throw std::invalid_argument("FFT geometry: scatter shape");
        for(std::size_t i=0;i<logical.size();++i)
            storage[address_linear(i)]=logical[i];
    }
    template<class T>
    void gather(std::span<const T> storage,std::span<T> logical) const {
        if(logical.size()!=shape.elements() || storage.size()<capacity ||
           !physical_geometry_detail::disjoint(storage,logical))
            throw std::invalid_argument("FFT geometry: gather shape");
        for(std::size_t i=0;i<logical.size();++i)
            logical[i]=storage[address_linear(i)];
    }
};

// A typed, bijective tensor coordinate space. The radices declare the
// logical mixed-radix digits in least-significant-axis-first order.
// physical_fastest is a genuine axis permutation, NOT an arbitrary
// reassociation of a transform: the caller owns the algorithmic proof.
struct factorized_space {
    std::vector<std::size_t> radices;
    std::vector<std::size_t> physical_fastest;
    std::vector<std::size_t> strides;
    std::size_t elements{};

    [[nodiscard]] static factorized_space make(
        std::vector<std::size_t> factors,
        std::vector<std::size_t> fastest) {
        if(factors.empty() || factors.size()>16 ||
           fastest.size()!=factors.size())
            throw std::invalid_argument("FFT geometry: invalid Cartesian rank");
        std::size_t n=1;
        for(auto f:factors) {
            if(f<2 || n>1048576/f)
                throw std::invalid_argument("FFT geometry: unsafe factorisation");
            n*=f;
        }
        std::vector<bool> seen(factors.size());
        std::vector<std::size_t> strides(factors.size());
        std::size_t stride=1;
        for(auto axis:fastest) {
            if(axis>=factors.size() || seen[axis])
                throw std::invalid_argument("FFT geometry: invalid axis permutation");
            seen[axis]=true;
            strides[axis]=stride;
            stride*=factors[axis]; // n bounded to <= 1M.
        }
        if(stride!=n)
            throw std::logic_error("FFT geometry: incomplete axis mapping");
        return {std::move(factors),std::move(fastest),
                std::move(strides),n};
    }
    [[nodiscard]] std::size_t offset(std::size_t logical) const {
        if(logical>=elements)
            throw std::out_of_range("FFT geometry: logical digit out of range");
        std::size_t physical=0;
        for(std::size_t axis=0;axis<radices.size();++axis) {
            const auto digit=logical%radices[axis];
            logical/=radices[axis];
            physical+=digit*strides[axis];
        }
        return physical;
    }
    [[nodiscard]] std::size_t inverse(std::size_t physical) const {
        if(physical>=elements)
            throw std::out_of_range("FFT geometry: physical digit out of range");
        std::vector<std::size_t> digits(radices.size());
        for(auto axis:physical_fastest) {
            digits[axis]=physical%radices[axis];
            physical/=radices[axis];
        }
        std::size_t logical=0,weight=1;
        for(std::size_t axis=0;axis<radices.size();++axis) {
            logical+=weight*digits[axis];
            weight*=radices[axis];
        }
        return logical;
    }
    template<class T>
    void scatter(std::span<const T> logical,std::span<T> physical) const {
        if(logical.size()!=elements || physical.size()!=elements ||
           !physical_geometry_detail::disjoint(logical,physical))
            throw std::invalid_argument("FFT geometry: factor scatter shape");
        for(std::size_t i=0;i<elements;++i)
            physical[offset(i)]=logical[i];
    }
    template<class T>
    void gather(std::span<const T> physical,std::span<T> logical) const {
        if(logical.size()!=elements || physical.size()!=elements ||
           !physical_geometry_detail::disjoint(physical,logical))
            throw std::invalid_argument("FFT geometry: factor gather shape");
        for(std::size_t i=0;i<elements;++i)
            logical[i]=physical[offset(i)];
    }
};

// Compile-time rank adapter reuses the existing CT-DP permutation_value<N>
// rather than introducing a second independent axis-ordinal system.
template<std::size_t N>
struct typed_factorized_space {
    static_assert(N>=1 && N<=16);
    factorized_space view;

    [[nodiscard]] static typed_factorized_space make(
        std::array<std::size_t,N> factors,
        permutation_value<N> fastest) {
        if(!fastest.is_valid())
            throw std::invalid_argument("FFT geometry: invalid CT-DP axis permutation");
        return {factorized_space::make(
            std::vector<std::size_t>(factors.begin(),factors.end()),
            std::vector<std::size_t>(fastest.order.begin(),fastest.order.end()))};
    }
    [[nodiscard]] std::size_t offset(std::size_t logical) const {
        return view.offset(logical);
    }
    [[nodiscard]] std::size_t inverse(std::size_t physical) const {
        return view.inverse(physical);
    }
    template<class T>
    void scatter(std::span<const T> src,std::span<T> dst) const {
        view.scatter(src,dst);
    }
    template<class T>
    void gather(std::span<const T> src,std::span<T> dst) const {
        view.gather(src,dst);
    }
};
} // namespace ctdp::space
