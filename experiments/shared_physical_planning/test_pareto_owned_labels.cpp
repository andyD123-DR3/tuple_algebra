// Regression: dominance must never observe a moved-from candidate.
#include <ctdp/solver/algorithms/compositional_pareto.h>
#include <vector>
#include <stdexcept>
#include <cstdio>
struct owned {std::vector<double> data;};
int main() {
    for(bool reverse:{false,true}) {
        const auto cost=[](owned const& v)->ctdp::cost_vector<2> {
            return v.data.empty()?ctdp::cost_vector<2>{{0,0}}:
                ctdp::cost_vector<2>{{v.data[0],v.data[1]}};
        };
        auto r=ctdp::solver::physical::compositional_pareto<int,owned,2>(0,{8,100,10},
            [&](int,auto const&,auto const& emit) {
                emit(owned{reverse?std::vector<double>{4,1}:std::vector<double>{1,4}});
                emit(owned{reverse?std::vector<double>{1,4}:std::vector<double>{4,1}});
                emit(owned{{5,5}});
            },cost,[](owned const&a,owned const&b){return a.data<b.data;},
            [](owned const&v){return v.data[0]+v.data[1];});
        if(r.frontier.size()!=2 || r.dominated!=1 || r.approximate ||
           r.frontier[0].data!=std::vector<double>{1,4} ||
           r.frontier[1].data!=std::vector<double>{4,1})
            throw std::runtime_error("Pareto lost an incomparable owned label");
    }
    std::puts("CT-DP owned labels remain intact during dominance: PASS");
}
