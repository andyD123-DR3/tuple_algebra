// test_binary_cut_desc.cpp - Unit tests for binary_cut_desc
// Sprint 8: Descriptor law tests
// Author: Andrew Drakeford

#include "ct_dp/space/binary_cut_desc.hpp"
#include <gtest/gtest.h>
#include <vector>
#include <numeric>

using namespace ct_dp::space;

// Test: Dimension Protocol
TEST(BinaryCutDesc, DimensionProtocol) {
    binary_cut_desc<10> desc;  // Value-oriented construction
    
    EXPECT_EQ(desc.size(), 9);              // 9 cuts for length 10
    EXPECT_EQ(desc.rank(), 1);              // 1D choice
    EXPECT_EQ(desc.shape()[0], 9);          // Shape consistent with size
    EXPECT_EQ(desc.interval_length(), 10);  // Problem length
    EXPECT_EQ(desc.cut_count(), 9);         // Cut count = size
}

// Test: Inverse Laws (Foundation of coordinate system)
TEST(BinaryCutDesc, InverseLaws) {
    binary_cut_desc<20> desc;
    
    // Law: ordinal → relative → ordinal is identity
    for (size_t ord = 0; ord < desc.size(); ++ord) {
        size_t rel = desc.ordinal_to_relative_cut(ord);
        EXPECT_EQ(desc.relative_cut_to_ordinal(rel), ord)
            << "Failed for ordinal " << ord;
    }
    
    // Law: relative → ordinal → relative is identity
    for (size_t rel = 1; rel < 20; ++rel) {
        size_t ord = desc.relative_cut_to_ordinal(rel);
        EXPECT_EQ(desc.ordinal_to_relative_cut(ord), rel)
            << "Failed for relative cut " << rel;
    }
}

// Test: Boundary Values
TEST(BinaryCutDesc, BoundaryValues) {
    // Len=2: Minimal legal descriptor
    {
        binary_cut_desc<2> desc2;
        EXPECT_EQ(desc2.size(), 1);                      // Only one cut
        EXPECT_EQ(desc2.ordinal_to_relative_cut(0), 1);  // Cut at position 1
        EXPECT_EQ(desc2.relative_cut_to_ordinal(1), 0);  // Inverse
    }
    
    // Len=3: Small case
    {
        binary_cut_desc<3> desc3;
        EXPECT_EQ(desc3.size(), 2);
        EXPECT_EQ(desc3.ordinal_to_relative_cut(0), 1);  // First cut
        EXPECT_EQ(desc3.ordinal_to_relative_cut(1), 2);  // Last cut
    }
    
    // Len=10: First and last cuts
    {
        binary_cut_desc<10> desc10;
        EXPECT_EQ(desc10.ordinal_to_relative_cut(0), 1);   // First cut
        EXPECT_EQ(desc10.ordinal_to_relative_cut(8), 9);   // Last cut
    }
}

// Test: Compile-Time Verification
TEST(BinaryCutDesc, CompileTimeVerification) {
    // Compile-time size queries
    constexpr binary_cut_desc<10> desc;
    static_assert(desc.size() == 9, "Size should be Len-1");
    static_assert(desc.rank() == 1, "Rank should be 1");
    static_assert(desc.interval_length() == 10, "Interval length should be Len");
    
    // Compile-time coordinate conversion
    static_assert(desc.ordinal_to_relative_cut(0) == 1, "First cut at 1");
    static_assert(desc.ordinal_to_relative_cut(8) == 9, "Last cut at 9");
    static_assert(desc.relative_cut_to_ordinal(1) == 0, "Cut 1 → ordinal 0");
    static_assert(desc.relative_cut_to_ordinal(9) == 8, "Cut 9 → ordinal 8");
}

// Test: Ordinal Range Coverage
TEST(BinaryCutDesc, OrdinalRangeCoverage) {
    binary_cut_desc<15> desc;
    
    // All ordinals [0, size-1] map to unique relative cuts
    std::vector<size_t> relative_cuts;
    for (size_t ord = 0; ord < desc.size(); ++ord) {
        relative_cuts.push_back(desc.ordinal_to_relative_cut(ord));
    }
    
    // Verify: cuts are [1, 2, ..., 14]
    EXPECT_EQ(relative_cuts.size(), 14);
    for (size_t i = 0; i < relative_cuts.size(); ++i) {
        EXPECT_EQ(relative_cuts[i], i + 1);
    }
}

// Test: Relative Cut Range Coverage
TEST(BinaryCutDesc, RelativeCutRangeCoverage) {
    binary_cut_desc<15> desc;
    
    // All relative cuts [1, Len-1] map to unique ordinals
    std::vector<size_t> ordinals;
    for (size_t rel = 1; rel < 15; ++rel) {
        ordinals.push_back(desc.relative_cut_to_ordinal(rel));
    }
    
    // Verify: ordinals are [0, 1, ..., 13]
    EXPECT_EQ(ordinals.size(), 14);
    for (size_t i = 0; i < ordinals.size(); ++i) {
        EXPECT_EQ(ordinals[i], i);
    }
}

// Test: Edge Case - Length 2
TEST(BinaryCutDesc, EdgeCaseLength2) {
    binary_cut_desc<2> desc;
    
    EXPECT_EQ(desc.size(), 1);
    EXPECT_EQ(desc.rank(), 1);
    EXPECT_EQ(desc.interval_length(), 2);
    
    // Only one valid ordinal and one valid cut
    EXPECT_EQ(desc.ordinal_to_relative_cut(0), 1);
    EXPECT_EQ(desc.relative_cut_to_ordinal(1), 0);
}

// Test: Shape Consistency
TEST(BinaryCutDesc, ShapeConsistency) {
    binary_cut_desc<7> desc;
    
    auto shape = desc.shape();
    EXPECT_EQ(shape.size(), 1);           // Rank 1 → shape has 1 element
    EXPECT_EQ(shape[0], desc.size());     // Shape[0] == size
    EXPECT_EQ(shape[0], 6);               // Len=7 → size=6
}

// Test: Multiple Descriptor Instances
TEST(BinaryCutDesc, MultipleInstances) {
    binary_cut_desc<5> desc1;
    binary_cut_desc<5> desc2;
    
    // Different instances of same type behave identically
    for (size_t ord = 0; ord < 4; ++ord) {
        EXPECT_EQ(desc1.ordinal_to_relative_cut(ord), 
                  desc2.ordinal_to_relative_cut(ord));
    }
}
