#include <ctdp/space/index_geometry.h>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>

namespace g=ctdp::space;
static void check(bool yes,const char* message) {
    if(!yes)throw std::runtime_error(message);
}
template<class Fn>void reject(Fn&& f,const char* message) {
    bool seen=false;
    try{f();}catch(std::exception const&){seen=true;}
    check(seen,message);
}
static void test_2d() {
    const g::shape2 shape{4,7};
    std::vector<std::uint32_t> original(shape.elements());
    for(std::size_t k=0;k<original.size();++k)original[k]=static_cast<std::uint32_t>(100+k);
    std::vector<std::uint32_t> storage(original.size()),restored(original.size());
    for(auto const& view:{g::view2::row_major(shape),
                          g::view2::column_major(shape)}) {
        std::fill(storage.begin(),storage.end(),0);
        view.scatter(std::span<const std::uint32_t>(original),
                     std::span<std::uint32_t>(storage));
        view.gather(std::span<const std::uint32_t>(storage),
                    std::span<std::uint32_t>(restored));
        check(restored==original,"2D geometry: failed true transpose roundtrip");
        for(std::size_t row=0;row<shape.outer;++row)
            for(std::size_t col=0;col<shape.inner;++col)
                check(storage[view.address({row,col})]==original[row*shape.inner+col],
                      "2D geometry: incorrect strided physical address");
    }
    check(g::view2::column_major(shape).address({1,2})==1+2*4,
          "2D geometry: column-major is not physical transpose");
    const auto padded=g::view2::make(shape,1,6,43,3);
    std::vector<std::uint32_t> larger(43,999);
    padded.scatter(std::span<const std::uint32_t>(original),
                   std::span<std::uint32_t>(larger));
    padded.gather(std::span<const std::uint32_t>(larger),
                  std::span<std::uint32_t>(restored));
    check(restored==original,"2D geometry: padded/offset roundtrip");
    check(larger[0]==999 && larger[2]==999,
          "2D geometry: padding values overwritten");
    reject([&]{(void)g::view2::make(shape,1,2,32);},
           "2D geometry: overlapping affine addresses admitted");
    reject([&]{(void)g::view2::make(shape,2,9,25);},
           "2D geometry: out-of-storage addresses admitted");
    reject([&]{(void)g::view2::make(shape,0,9,65);},
           "2D geometry: zero stride admitted");
    reject([&]{(void)g::view2::make(shape,1,
        std::numeric_limits<std::size_t>::max(),99);},
           "2D geometry: stride overflow admitted");
    reject([&]{(void)g::view2::row_major({0,7});},
           "2D geometry: zero dimension admitted");
    reject([&]{(void)g::view2::row_major({
        std::numeric_limits<std::size_t>::max(),3});},
           "2D geometry: extent overflow admitted");
    reject([&]{(void)g::view2::column_major(shape).address({4,0});},
           "2D geometry: out-of-domain coordinate admitted");
    reject([&]{g::view2::column_major(shape).scatter(
        std::span<const std::uint32_t>(original),
        std::span<std::uint32_t>(restored).first(27));},
           "2D geometry: invalid scatter capacity admitted");
}
static void test_multiradix() {
    for(const auto& axes:std::array<std::array<std::size_t,3>,4>{
        {{0,1,2},{2,1,0},{1,0,2},{0,2,1}}}) {
        auto space=g::factorized_space::make({2,4,7},
            std::vector<std::size_t>(axes.begin(),axes.end()));
        check(space.elements==56,"Cartesian geometry: incorrect product");
        std::vector<std::uint32_t> input(space.elements),phys(space.elements),out(space.elements);
        for(std::size_t i=0;i<input.size();++i)
            input[i]=static_cast<std::uint32_t>(37*i+9);
        std::vector<bool> seen(space.elements,false);
        for(std::size_t i=0;i<space.elements;++i) {
            const auto p=space.offset(i);
            check(p<space.elements && !seen[p],
                  "Cartesian geometry: layout not bijective");
            seen[p]=true;
            check(space.inverse(p)==i,
                  "Cartesian geometry: physical inverse incorrect");
        }
        space.scatter(std::span<const std::uint32_t>(input),
                      std::span<std::uint32_t>(phys));
        space.gather(std::span<const std::uint32_t>(phys),
                     std::span<std::uint32_t>(out));
        check(input==out,"Cartesian geometry: mixed radix permutation changed data");
    }
    const auto typed=g::typed_factorized_space<3>::make(
        {2,4,7},g::permutation_value<3>{{2,1,0}});
    for(std::size_t i=0;i<56;++i)
        check(typed.inverse(typed.offset(i))==i,
              "CT-DP typed permutation adapter lost inverse");
    reject([] {
        (void)g::typed_factorized_space<3>::make(
            {2,4,7},g::permutation_value<3>{{0,0,2}});
    },"CT-DP typed permutation adapter admitted duplicate axis");
    const auto identity=g::factorized_space::make({2,4,7},{0,1,2});
    for(std::size_t i=0;i<identity.elements;++i)
        check(identity.offset(i)==i,"Cartesian geometry: identity permutation wrong");
    const auto rev=g::factorized_space::make({2,4,7},{2,1,0});
    check(rev.offset(1)==28 && rev.offset(2)==7 && rev.offset(8)==1,
          "Cartesian geometry: reversed factor stride wrong");

    reject([]{(void)g::factorized_space::make({2,4,7},{0,0,2});},
           "Cartesian geometry: repeated axis accepted");
    reject([]{(void)g::factorized_space::make({2,4,7},{0,1,3});},
           "Cartesian geometry: invalid axis accepted");
    reject([]{(void)g::factorized_space::make({2,4,7},{0,1});},
           "Cartesian geometry: missing axis accepted");
    reject([]{(void)g::factorized_space::make({2,1,7},{0,1,2});},
           "Cartesian geometry: trivial/invalid factor admitted");
    reject([]{(void)g::factorized_space::make({2,65536,16},{0,1,2});},
           "Cartesian geometry: excessively large product admitted");
    reject([&]{(void)identity.inverse(56);},
           "Cartesian geometry: inverse outside physical domain admitted");
}
int main() {
    test_2d();
    test_multiradix();
    std::puts("CT-DP physical index geometry: PASS");
}
