// Standalone qualification for the shared, domain-neutral recursive plan API.
// These tests use actual physical leaf descriptors but deliberately grant no
// numerical/effect or hardware admission from structural shape alone.
#include <ctdp/solver/plans/physical_recursive.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

namespace shared=ctdp::solver::physical;
enum class mode : std::uint8_t { dense, scalar };
struct leaf {
    std::size_t first{},last{},depth{};
    mode selected{};
};
static void check(bool good,char const* reason) {
    if(!good)throw std::runtime_error(reason);
}
template<class Fn>
static void rejects(Fn&& call,char const* reason) {
    bool rejected=false;
    try {call();} catch(std::exception const&) {rejected=true;}
    check(rejected,reason);
}
static auto splitter() {
    return [](shared::interval part,std::size_t)
           ->std::optional<std::size_t> {
        if(part.size()<32)return std::nullopt;
        return part.first+part.size()/2;
    };
}
static auto lowerer() {
    return [](shared::interval p,std::size_t depth)->leaf {
        return {p.first,p.last,depth,
                p.first<32?mode::dense:mode::scalar};
    };
}
static auto witness() {
    return [](leaf const& a,leaf const& b)->shared::fusion_witness {
        const bool equal=a.selected==b.selected;
        // In this small test contract, identical leaf modes have identical
        // state interfaces and every leaf preserves source observations.
        return {equal,equal,equal};
    };
}
static auto join() {
    return [](leaf const& a,leaf const& b)->leaf {
        auto out=a;
        out.last=b.last;
        return out;
    };
}
static void test_bounded_tree_and_fusion() {
    std::vector<leaf> physical;
    shared::append_subtree(physical,{0,64},0,{8,2,4},
                           splitter(),lowerer());
    check(physical.size()==4,"recursive plan: incorrect leaf count");
    shared::require_partition(std::span<const leaf>(physical),{0,64});
    for(std::size_t i=0;i<4;++i) {
        check(physical[i].first==i*16 &&
              physical[i].last==(i+1)*16 &&
              physical[i].depth==2,
              "recursive plan: invalid leaf depth/order");
    }
    const auto fused=shared::fuse_adjacent(
        std::span<const leaf>(physical),shared::interval{0,64},
        witness(),join());
    check(fused.size()==2 &&
          fused[0].first==0 && fused[0].last==32 &&
          fused[1].first==32 && fused[1].last==64 &&
          fused[0].selected==mode::dense &&
          fused[1].selected==mode::scalar,
          "recursive plan: incompatible interfaces fused or same modes not fused");

    // Even structurally identical modes cannot fuse unless every field of
    // their domain-specific boundary witness is explicitly granted.
    const auto denied=shared::fuse_adjacent(
        std::span<const leaf>(physical),{0,64},
        [](leaf const&,leaf const&) {
            return shared::fusion_witness{true,false,true};
        },join());
    check(denied.size()==4,"recursive plan: missing interface proof admitted");

    // Root offset need not be zero; source order must remain exact.
    std::vector<leaf> offset;
    shared::append_subtree(offset,{128,160},0,{8,1,4},
        splitter(),
        [](shared::interval r,std::size_t depth) {
            return leaf{r.first,r.last,depth,mode::scalar};
        });
    shared::require_partition(std::span<const leaf>(offset),{128,160});
    check(offset.size()==2 && offset[0].first==128 &&
          offset[1].last==160,"recursive plan: absolute interval drift");
}
static void test_fail_closed() {
    rejects([&] {
        std::vector<leaf> out;
        shared::append_subtree(out,{4,4},0,{8,2,8},splitter(),lowerer());
    },"recursive plan: empty root admitted");
    rejects([&] {
        std::vector<leaf> out;
        shared::append_subtree(out,{0,64},0,{0,2,8},splitter(),lowerer());
    },"recursive plan: zero minimum leaf admitted");
    rejects([&] {
        std::vector<leaf> out;
        shared::append_subtree(out,{0,64},0,{8,40,8},splitter(),lowerer());
    },"recursive plan: unsafe recursion depth admitted");
    rejects([&] {
        std::vector<leaf> out;
        shared::append_subtree(out,{0,64},0,{8,2,2},splitter(),lowerer());
    },"recursive plan: excessive leaves accepted");
    rejects([&] {
        std::vector<leaf> out;
        shared::append_subtree(out,{0,64},0,{8,2,8},
            [](shared::interval,std::size_t)->std::optional<std::size_t> {
                return 0;
            },lowerer());
    },"recursive plan: zero-sized child accepted");
    rejects([&] {
        std::vector<leaf> out;
        shared::append_subtree(out,{0,64},0,{8,2,8},
            [](shared::interval r,std::size_t)->std::optional<std::size_t> {
                return r.first+1;
            },lowerer());
    },"recursive plan: subminimum child accepted");
    rejects([&] {
        std::vector<leaf> out;
        shared::append_subtree(out,{0,64},0,{8,2,8},splitter(),
            [](shared::interval p,std::size_t depth) {
                return leaf{p.first,p.last+1,depth,mode::dense};
            });
    },"recursive plan: lowerer changed its source observation range");
    const std::array<leaf,2> gap{{
        {0,8,0,mode::dense},{9,16,0,mode::dense}}};
    rejects([&] {
        shared::require_partition(std::span<const leaf>(gap),{0,16});
    },"recursive plan: missing row/event admitted");
    const std::array<leaf,2> overlap{{
        {0,9,0,mode::dense},{8,16,0,mode::dense}}};
    rejects([&] {
        shared::require_partition(std::span<const leaf>(overlap),{0,16});
    },"recursive plan: duplicate row/event admitted");
    const std::array<leaf,2> adjacent{{
        {0,8,0,mode::dense},{8,16,0,mode::dense}}};
    rejects([&] {
        (void)shared::fuse_adjacent(
            std::span<const leaf>(adjacent),{0,16},witness(),
            [](leaf const& a,leaf const& b) {
                return leaf{a.first,b.last+1,0,mode::dense};
            });
    },"recursive plan: invalid merged physical range admitted");
}
static void test_search_economics() {
    check(shared::affordable({1000,0,1000.0,1000.0,0.0,0.25,1.e6}),
          "recursive plan: feasible search was rejected");
    check(!shared::affordable({1,0,1000.0,1000.0,0.0,0.25,1.e6}),
          "recursive plan: expensive short-horizon search admitted");
    check(shared::affordable({1000,2,1000.0,1000.0,100000.0,0.25,1.e6}),
          "recursive plan: measured probe history not admitted");
    check(!shared::affordable({1000,2,1000.0,1000.0,200000.0,0.25,1.e6}),
          "recursive plan: overbudget measured probes admitted");
    check(!shared::affordable({1000,0,1.0,1.0,0.0,1.5,1.e6}),
          "recursive plan: invalid exploration fraction admitted");
    check(!shared::affordable({1000,0,
        std::numeric_limits<double>::infinity(),1.,0.,.25,1.e6}),
          "recursive plan: unbounded cost admitted");

    const shared::adoption_context good{
        1000,100.,10.,70.,10.,1.,1000.,.01,.10};
    const auto answer=shared::should_adopt(good);
    check(answer.adopt && answer.future_savings_ns==30000.,
          "recursive plan: profitable qualified candidate refused");
    auto bad=good;bad.remaining_calls=10;
    check(!shared::should_adopt(bad).adopt,
          "recursive plan: short-horizon candidate incorrectly adopted");
    bad=good;bad.trial_median_ns=105.;
    check(!shared::should_adopt(bad).adopt,
          "recursive plan: regressing candidate adopted");
    bad=good;bad.mad_penalty=-1.;
    check(!shared::should_adopt(bad).adopt,
          "recursive plan: invalid dispersion penalty admitted");
    bad=good;bad.action_wall_ns=
        std::numeric_limits<double>::quiet_NaN();
    check(!shared::should_adopt(bad).adopt,
          "recursive plan: unobserved search cost admitted");
}

int main() {
    test_bounded_tree_and_fusion();
    test_fail_closed();
    test_search_economics();
    std::puts("CT-DP shared physical recursion: PASS");
}
