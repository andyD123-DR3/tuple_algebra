// Independent exhaustive red-team oracle for the chain-fusion reference DP.
// Assertion checks use exceptions rather than assert() so they remain active
// under Release/NDEBUG, including GCC, Clang and MSVC.
#include <ctdp/solver/algorithms/fusion_boundary_witness.h>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <limits>
#include <stdexcept>

namespace fp=ctdp::solver::fusion_probe;
using fp::cost_type;
using fp::unavailable;

void require(bool ok,const char* message) {
    if(!ok)throw std::runtime_error(message);
}
template<class Action>
void must_reject(Action&& act,const char* reason) {
    bool threw=false;
    try {act();}catch(std::exception const&) {threw=true;}
    require(threw,reason);
}
struct oracle_solution {
    bool feasible{};
    cost_type cost{unavailable};
    std::uint64_t bits{};
};

// Deliberately independent: enumerate all 2^(n-1) cut masks, reconstruct
// segments from each mask and sum their stored costs. This does NOT call
// solve_chain, reuse a DP prefix, or import its tie-breaking implementation.
template<std::size_t N>
oracle_solution brute_chain(fp::chain_table<N> const& input) {
    const auto n=input.nodes;
    if(n==0)return {true,0,0};
    oracle_solution best{};
    const auto possible=std::uint64_t{1}<<(n-1);
    for(std::uint64_t cuts=0;cuts<possible;++cuts) {
        cost_type total=0;
        std::size_t begin=0;
        bool valid=true;
        for(std::size_t k=0;k<n;++k) {
            if(k+1<n && !(cuts&(std::uint64_t{1}<<k)))continue;
            const auto segment=input.get(begin,k+1);
            if(segment==unavailable || total>=unavailable-segment) {
                valid=false;break;
            }
            total+=segment;
            begin=k+1;
        }
        if(valid && (!best.feasible || total<best.cost ||
           (total==best.cost && cuts<best.bits)))
            best={true,total,cuts};
    }
    return best;
}

template<std::size_t N>
void compare_chain(fp::chain_table<N> const& t) {
    const auto actual=fp::solve_chain(t);
    const auto expected=brute_chain(t);
    require(actual.feasible==expected.feasible,
            "exact chain DP disagrees with exhaustive partition feasibility");
    if(!expected.feasible)return;
    require(actual.cost==expected.cost && actual.cuts==expected.bits,
            "exact chain DP disagrees with exhaustive optimal cost or tie");
    require(actual.segments==(t.nodes ? 1+std::popcount(actual.cuts):0),
            "exact chain DP reported an invalid cut reconstruction");
}

// Enumerate ALL 2^n layout sequences, including transitions and infeasibility.
template<std::size_t N>
oracle_solution brute_layout(fp::layout_chain<N> const& input) {
    if(input.nodes==0)return {true,0,0};
    oracle_solution best{};
    const auto count=std::uint64_t{1}<<input.nodes;
    for(std::uint64_t choices=0;choices<count;++choices) {
        cost_type sum=0;
        bool admissible=true;
        for(std::size_t i=0;i<input.nodes;++i) {
            const auto current=static_cast<std::size_t>((choices>>i)&1u);
            const auto cost=input.execution[i][current];
            if(cost==unavailable || sum>=unavailable-cost) {
                admissible=false;break;
            }
            sum+=cost;
            if(i>0) {
                const auto previous=static_cast<std::size_t>((choices>>(i-1))&1u);
                const auto transfer=input.conversion[previous][current];
                if(transfer==unavailable || sum>=unavailable-transfer) {
                    admissible=false;break;
                }
                sum+=transfer;
            }
        }
        if(admissible && (!best.feasible || sum<best.cost ||
            (sum==best.cost && choices<best.bits)))
            best={true,sum,choices};
    }
    return best;
}
template<std::size_t N>
void compare_layout(fp::layout_chain<N> const& t) {
    const auto actual=fp::solve_layout(t);
    const auto expected=brute_layout(t);
    require(actual.feasible==expected.feasible,
            "boundary-state DP disagrees with exhaustive layout feasibility");
    if(actual.feasible)
        require(actual.cost==expected.cost && actual.layouts==expected.bits,
                "boundary-state DP disagrees with exhaustive optimal layout");
}

std::uint32_t seed=UINT32_C(0x918a75ef);
std::uint32_t next_rand() {
    seed=seed*UINT32_C(1664525)+UINT32_C(1013904223);
    return seed;
}

void test_chain() {
    constexpr auto static_case=[] {
        fp::chain_table<4> table{3};
        table.set(0,1,4);table.set(1,2,4);table.set(2,3,4);
        table.set(0,2,3);table.set(1,3,6);table.set(0,3,9);
        return fp::solve_chain(table);
    }();
    static_assert(static_case.feasible && static_case.cost==7 &&
                  static_case.cuts==2u && static_case.segments==2);
    fp::chain_table<4> empty{};
    compare_chain(empty);
    fp::chain_table<4> unavailable_case{4};
    compare_chain(unavailable_case);
    auto equal=fp::chain_table<4>{3};
    for(std::size_t i=0;i<3;++i)
        for(std::size_t j=i+1;j<=3;++j)equal.set(i,j,0);
    compare_chain(equal);
    require(fp::solve_chain(equal).cuts==0,
            "equal-cost chain prefers noncanonical unnecessary cuts");

    // All 3^10 assignments of {zero, positive, forbidden} to the ten
    // intervals of a 4-node chain. This is exhaustive for that declared
    // finite cost alphabet, not a sample of hand-selected examples.
    std::size_t explored=0;
    for(std::uint32_t code=0;code<59049u;++code) {
        fp::chain_table<4> table{4};
        auto digits=code;
        for(std::size_t i=0;i<4;++i)
            for(std::size_t j=i+1;j<=4;++j) {
                const auto digit=digits%3u;digits/=3u;
                if(digit!=2u)table.set(i,j,digit);
            }
        compare_chain(table);
        ++explored;
    }
    require(explored==59049u,"exhaustive chain alphabet sweep incomplete");

    // Larger deterministic, irregular chains exercise missing segments,
    // zero costs, ties, sparse feasibility and non-additive-looking tables
    // which remain valid because the TOTAL cost of each segment is explicit.
    for(unsigned rep=0;rep<1200;++rep) {
        fp::chain_table<10> table{1u+next_rand()%9u};
        for(std::size_t i=0;i<table.nodes;++i)
            for(std::size_t j=i+1;j<=table.nodes;++j) {
                const auto draw=next_rand()%7u;
                if(draw!=0)table.set(i,j,static_cast<cost_type>(next_rand()%31u));
            }
        compare_chain(table);
    }
    fp::chain_table<3> overflow{2};
    overflow.set(0,1,unavailable-2);
    overflow.set(1,2,3);
    overflow.set(0,2,17);
    compare_chain(overflow);
    require(fp::solve_chain(overflow).cost==17,
            "overflow should invalidate an impossible partition");
    must_reject([&] {fp::chain_table<2> invalid{3};(void)invalid;},
                "chain capacity exceeded without rejection");
    must_reject([&] {overflow.set(2,2,1);},
                "empty fusion segment admitted");
    must_reject([&] {overflow.set(0,1,unavailable);},
                "unavailable sentinel accepted as finite cost");
}

void test_layout() {
    constexpr auto example=[] {
        fp::layout_chain<2> x{2};
        x.set(0,0,1);x.set(0,1,2);
        x.set(1,0,3);x.set(1,1,1);
        return fp::solve_layout(x);
    }();
    static_assert(example.feasible && example.cost==3 && example.layouts==3);
    fp::layout_chain<5> empty{};
    compare_layout(empty);
    fp::layout_chain<5> impossible{3};
    compare_layout(impossible);
    for(unsigned iteration=0;iteration<1500;++iteration) {
        fp::layout_chain<9> t{1u+next_rand()%8u};
        for(std::size_t node=0;node<t.nodes;++node)
            for(std::size_t state=0;state<2;++state)
                if(next_rand()%5u!=0u)
                    t.set(node,state,next_rand()%22u);
        for(std::size_t before=0;before<2;++before)
            for(std::size_t after=0;after<2;++after) {
                if(next_rand()%4u==0u)
                    t.set_conversion(before,after,unavailable);
                else
                    t.set_conversion(before,after,next_rand()%13u);
            }
        compare_layout(t);
    }
    fp::layout_chain<2> over{2};
    over.set(0,0,unavailable-2);
    over.set(1,0,4);
    over.set(0,1,2);
    over.set(1,1,3);
    compare_layout(over);
    require(fp::solve_layout(over).cost==5,
            "boundary-layout state overflow was accepted");

    const auto witness=fp::find_layout_counterexample();
    require(witness.model_weight==7 &&
            witness.producer[0]==1 && witness.producer[1]==2 &&
            witness.consumer[0]==3 && witness.consumer[1]==1 &&
            witness.greedy_prefix==4 && witness.stateful_best==3,
            "layout counterexample was not minimum in its bounded grammar");
    // Independent minimality: enumerate ALL positive 4-tuples of lower
    // aggregate weight and verify none has a more expensive producer layout
    // that wins globally with forbidden cross-layout conversions.
    for(unsigned a=1;a<=6;++a)
        for(unsigned b=1;b<=6;++b)
            for(unsigned c=1;c<=6;++c)
                for(unsigned d=1;d<=6;++d) {
                    if(a+b+c+d>=witness.model_weight || a>=b)continue;
                    require(!(b+d<a+c),
                            "a lower-weight layout-boundary witness was missed");
                }
}

fp::fork_solution exhaustive_fork_cost(fp::fork_cost x) {
    if(!fp::valid_fork(x))return {};
    // Independent physical action counts: evaluate producer once with
    // materialisation or twice when pure; charge each consumer, savings,
    // and any inter-region memory effect explicitly.
    fp::fork_solution best{true,
        static_cast<cost_type>(x.producer)+x.materialization+x.left+x.right,
        fp::fork_strategy::materialize_once};
    if(x.producer_is_pure) {
        cost_type duplicated=0;
        const std::array<unsigned,2> work{
            static_cast<unsigned>(x.left),static_cast<unsigned>(x.right)};
        const std::array<unsigned,2> savings{
            static_cast<unsigned>(x.gain_left),
            static_cast<unsigned>(x.gain_right)};
        for(std::size_t branch=0;branch<2;++branch) {
            duplicated+=x.producer;
            duplicated+=work[branch];
            duplicated-=savings[branch];
        }
        if(duplicated<best.total)
            best={true,duplicated,fp::fork_strategy::duplicate_producer};
    }
    return best;
}

void test_fork() {
    // Complete small-domain enumeration for both pure and impure producers,
    // zero/positive fusion discounts and shared materialisation overheads.
    for(unsigned pure=0;pure<2;++pure)
        for(unsigned p=1;p<=6;++p)
            for(unsigned l=1;l<=3;++l)
                for(unsigned r=1;r<=3;++r)
                    for(unsigned memory=0;memory<=3;++memory)
                        for(unsigned gl=0;gl<=2;++gl)
                            for(unsigned gr=0;gr<=2;++gr) {
                                const fp::fork_cost x{
                                    static_cast<std::uint16_t>(p),
                                    static_cast<std::uint16_t>(l),
                                    static_cast<std::uint16_t>(r),
                                    static_cast<std::uint16_t>(memory),
                                    static_cast<std::uint16_t>(gl),
                                    static_cast<std::uint16_t>(gr),pure!=0};
                                const auto model=fp::solve_fork_model(x);
                                const auto brute=exhaustive_fork_cost(x);
                                require(model.feasible==brute.feasible &&
                                        model.total==brute.total &&
                                        model.strategy==brute.strategy,
                                        "fork optimiser disagrees with independent physical oracle");
                                const auto lift=fp::lift_independent_chains(x);
                                if(!pure)
                                    require(!lift.feasible,
                                            "effectful producer was illegally duplicated by auxiliary lift");
                                else
                                    require(lift.feasible &&
                                            lift.total==fp::duplication_cost(x),
                                            "independent chain lift did not charge duplicated producer");
                            }
    const auto corpus=fp::find_fork_counterexamples<8>();
    require(corpus.count==8 && corpus.candidates_examined>=8,
            "counterexample corpus did not contain eight witnesses");
    const auto first=corpus.cases[0];
    require(first.problem.producer==4 &&
            first.problem.left==1 && first.problem.right==1 &&
            first.problem.materialization==1 &&
            first.problem.gain_left==1 && first.problem.gain_right==1 &&
            first.model_weight==9 &&
            first.independent_chains==8 && first.global_best==7 &&
            first.regret==1,
            "fork witness not minimal within declared positive-cost grammar");
    for(std::size_t i=0;i<corpus.count;++i) {
        const auto& c=corpus.cases[i];
        require(c.independent_chains>c.global_best &&
                c.regret==c.independent_chains-c.global_best,
                "corpus contains no actual auxiliary-lift regret");
        if(i>0)require(c.model_weight>=corpus.cases[i-1].model_weight,
                       "corpus order violates increasing model complexity");
    }
    // Independent enumeration of every admissible smaller-weight model.
    // The positive gain and memory rules rule out trivial no-op fusion.
    for(unsigned p=1;p<=8;++p)
        for(unsigned l=1;l<=4;++l)
            for(unsigned r=1;r<=4;++r)
                for(unsigned m=1;m<=3;++m)
                    for(unsigned gl=1;gl<=3;++gl)
                        for(unsigned gr=1;gr<=3;++gr) {
                            if(p+l+r+m+gl+gr>=first.model_weight ||
                               l>r || (l==r && gl>gr) ||
                               gl>p+l || gr>p+r)continue;
                            fp::fork_cost x{
                                static_cast<std::uint16_t>(p),
                                static_cast<std::uint16_t>(l),
                                static_cast<std::uint16_t>(r),
                                static_cast<std::uint16_t>(m),
                                static_cast<std::uint16_t>(gl),
                                static_cast<std::uint16_t>(gr),true};
                            const auto projection=exhaustive_fork_cost(x);
                            require(projection.strategy!=fp::fork_strategy::materialize_once ||
                                    fp::duplication_cost(x)<=projection.total,
                                    "independent smaller-weight fork counterexample exists");
                        }
    fp::fork_cost prefer_duplicate{1,3,3,5,1,1,true};
    require(fp::solve_fork_model(prefer_duplicate).strategy==
                fp::fork_strategy::duplicate_producer,
            "materialisation was incorrectly made universally preferable");
    prefer_duplicate.producer_is_pure=false;
    require(fp::solve_fork_model(prefer_duplicate).strategy==
                fp::fork_strategy::materialize_once &&
            !fp::lift_independent_chains(prefer_duplicate).feasible,
            "impure source should require single evaluation in toy model");
}

int main() {
    try {
        test_chain();
        test_layout();
        test_fork();
        std::puts("CT-DP fusion boundary PASS exhaustive_chain=59049 "
                  "random_chain=1200 random_layout=1500 fork_corpus=8");
        return 0;
    }catch(std::exception const& e) {
        std::fprintf(stderr,"CT-DP fusion boundary FAIL: %s\n",e.what());
        return 1;
    }
}
