// Deterministic JSONL corpus for the bounded auxiliary-problem red team.
// Every field is an integer or literal string; there is no clock, RNG, ISA
// fingerprint, hardware benchmark, or floating-point formatting dependence.
#include <ctdp/solver/algorithms/fusion_boundary_witness.h>
#include <cstddef>
#include <cstdio>
#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#endif

namespace fp=ctdp::solver::fusion_probe;

int main() {
#if defined(_WIN32)
    // MSVC text stdout otherwise translates LF to CRLF when redirected.
    // CI compares the exact corpus bytes, not newline-normalised text.
    if(_setmode(_fileno(stdout),_O_BINARY)==-1)return 3;
#endif
    constexpr unsigned grammar_version=1;
    const auto forks=fp::find_fork_counterexamples<8>();
    if(forks.count!=8)return 1;
    std::printf("{\"kind\":\"schema\",\"version\":%u,\"family\":\"ctdp-fusion-boundary\","
                "\"cost_units\":\"abstract_nonnegative_integer\",\"numeric_proof\":false}\n",
                grammar_version);
    for(std::size_t i=0;i<forks.count;++i) {
        const auto& f=forks.cases[i];
        std::printf("{\"kind\":\"fork\",\"rank\":%zu,\"nodes\":3,\"weight\":%u,"
                    "\"producer\":%u,\"left\":%u,\"right\":%u,\"materialization\":%u,"
                    "\"gain_left\":%u,\"gain_right\":%u,\"requires_pure_producer\":true,"
                    "\"lift\":\"duplicate_producer\",\"global\":\"materialize_once\","
                    "\"lift_cost\":%llu,\"optimal_cost\":%llu,\"regret\":%llu}\n",
                    i+1,f.model_weight,
                    static_cast<unsigned>(f.problem.producer),
                    static_cast<unsigned>(f.problem.left),
                    static_cast<unsigned>(f.problem.right),
                    static_cast<unsigned>(f.problem.materialization),
                    static_cast<unsigned>(f.problem.gain_left),
                    static_cast<unsigned>(f.problem.gain_right),
                    static_cast<unsigned long long>(f.independent_chains),
                    static_cast<unsigned long long>(f.global_best),
                    static_cast<unsigned long long>(f.regret));
    }
    const auto layout=fp::find_layout_counterexample();
    if(layout.model_weight!=7)return 2;
    std::printf("{\"kind\":\"layout_boundary\",\"rank\":1,\"nodes\":2,"
                "\"weight\":%u,\"producer_layout0\":%u,\"producer_layout1\":%u,"
                "\"consumer_layout0\":%u,\"consumer_layout1\":%u,"
                "\"conversion\":\"forbidden\",\"greedy_cost\":%llu,"
                "\"stateful_cost\":%llu,\"regret\":%llu}\n",
                layout.model_weight,
                static_cast<unsigned>(layout.producer[0]),
                static_cast<unsigned>(layout.producer[1]),
                static_cast<unsigned>(layout.consumer[0]),
                static_cast<unsigned>(layout.consumer[1]),
                static_cast<unsigned long long>(layout.greedy_prefix),
                static_cast<unsigned long long>(layout.stateful_best),
                static_cast<unsigned long long>(layout.regret));
    return 0;
}
