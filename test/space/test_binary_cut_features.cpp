// test_binary_cut_features.cpp - Feature encoding tests
// Sprint 8: Standalone feature encoding helper tests
// Author: Andrew Drakeford

#include "ct_dp/space/binary_cut_desc.hpp"
#include "ct_dp/space/binary_cut_features.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <numeric>

using namespace ct_dp::space;

// Test: One-Hot Encoding
TEST(BinaryCutFeatures, OneHotEncoding) {
    // Fixed-length encoding
    auto features = binary_cut_features<5>::encode_onehot(2);  // rel=2
    
    // Law: Output dimension is Len-1
    EXPECT_EQ(features.size(), 4);
    
    // Law: Exactly one active entry
    EXPECT_EQ(features[1], 1.0);  // Index 1 for rel=2
    size_t count = std::count(features.begin(), features.end(), 1.0);
    EXPECT_EQ(count, 1);
    
    // Law: Sum is 1
    double sum = std::accumulate(features.begin(), features.end(), 0.0);
    EXPECT_DOUBLE_EQ(sum, 1.0);
}

// Test: One-Hot All Positions
TEST(BinaryCutFeatures, OneHotAllPositions) {
    constexpr size_t Len = 7;
    
    for (size_t rel = 1; rel < Len; ++rel) {
        auto features = binary_cut_features<Len>::encode_onehot(rel);
        
        EXPECT_EQ(features.size(), Len - 1);
        EXPECT_EQ(features[rel - 1], 1.0);
        
        // All other entries are 0
        for (size_t i = 0; i < features.size(); ++i) {
            if (i == rel - 1) {
                EXPECT_EQ(features[i], 1.0);
            } else {
                EXPECT_EQ(features[i], 0.0);
            }
        }
    }
}

// Test: Integer Encoding
TEST(BinaryCutFeatures, IntegerEncoding) {
    auto feature = binary_cut_features<10>::encode_integer(5);
    
    // Law: Returns the relative cut directly
    EXPECT_EQ(feature, 5);
}

// Test: Integer Encoding Range
TEST(BinaryCutFeatures, IntegerEncodingRange) {
    constexpr size_t Len = 15;
    
    for (size_t rel = 1; rel < Len; ++rel) {
        auto feature = binary_cut_features<Len>::encode_integer(rel);
        EXPECT_EQ(feature, rel);
    }
}

// Test: Normalized Encoding
TEST(BinaryCutFeatures, NormalizedEncoding) {
    // Variable-length encoding
    auto features = binary_cut_features<10>::encode_normalized<double>(5);
    
    // Law: Fixed 3-element output
    EXPECT_EQ(features.size(), 3);
    
    // Law: Correct values
    EXPECT_DOUBLE_EQ(features[0], 10.0);   // Length
    EXPECT_DOUBLE_EQ(features[1], 5.0);    // Relative cut
    EXPECT_DOUBLE_EQ(features[2], 0.5);    // Normalized ratio
    
    // Law: Ratio in (0,1)
    EXPECT_GT(features[2], 0.0);
    EXPECT_LT(features[2], 1.0);
}

// Test: Normalized Encoding Ratio
TEST(BinaryCutFeatures, NormalizedEncodingRatio) {
    constexpr size_t Len = 20;
    
    for (size_t rel = 1; rel < Len; ++rel) {
        auto features = binary_cut_features<Len>::encode_normalized<double>(rel);
        
        EXPECT_DOUBLE_EQ(features[0], static_cast<double>(Len));
        EXPECT_DOUBLE_EQ(features[1], static_cast<double>(rel));
        EXPECT_DOUBLE_EQ(features[2], static_cast<double>(rel) / Len);
    }
}

// Test: Full Encoding
TEST(BinaryCutFeatures, FullEncoding) {
    // Variable-length with asymmetry
    auto features = binary_cut_features<10>::encode_full<double>(3);
    
    // Law: Fixed 5-element output
    EXPECT_EQ(features.size(), 5);
    
    // Law: Correct values
    EXPECT_DOUBLE_EQ(features[0], 10.0);   // Total length
    EXPECT_DOUBLE_EQ(features[1], 3.0);    // Cut position
    EXPECT_DOUBLE_EQ(features[2], 0.3);    // Normalized ratio
    EXPECT_DOUBLE_EQ(features[3], 3.0);    // Left size
    EXPECT_DOUBLE_EQ(features[4], 7.0);    // Right size
    
    // Law: Left + Right = Total
    EXPECT_DOUBLE_EQ(features[3] + features[4], features[0]);
}

// Test: Full Encoding Size Preservation
TEST(BinaryCutFeatures, FullEncodingSizePreservation) {
    constexpr size_t Len = 15;
    
    for (size_t rel = 1; rel < Len; ++rel) {
        auto features = binary_cut_features<Len>::encode_full<double>(rel);
        
        // Law: Left + Right = Total
        EXPECT_DOUBLE_EQ(features[3] + features[4], features[0]);
        
        // Law: Left size = rel
        EXPECT_DOUBLE_EQ(features[3], static_cast<double>(rel));
        
        // Law: Right size = Len - rel
        EXPECT_DOUBLE_EQ(features[4], static_cast<double>(Len - rel));
    }
}

// Test: Normalized Boundary Values
TEST(BinaryCutFeatures, NormalizedBoundaryValues) {
    constexpr size_t Len = 10;
    
    // First cut (rel=1)
    {
        auto features = binary_cut_features<Len>::encode_normalized<double>(1);
        EXPECT_DOUBLE_EQ(features[2], 0.1);  // 1/10
    }
    
    // Last cut (rel=9)
    {
        auto features = binary_cut_features<Len>::encode_normalized<double>(9);
        EXPECT_DOUBLE_EQ(features[2], 0.9);  // 9/10
    }
    
    // Middle cut (rel=5)
    {
        auto features = binary_cut_features<Len>::encode_normalized<double>(5);
        EXPECT_DOUBLE_EQ(features[2], 0.5);  // 5/10
    }
}

// Test: Compile-Time Encoding
TEST(BinaryCutFeatures, CompileTimeEncoding) {
    constexpr auto onehot = binary_cut_features<5>::encode_onehot(2);
    static_assert(onehot.size() == 4, "One-hot size should be Len-1");
    static_assert(onehot[1] == 1.0, "One-hot should be active at rel-1");
    
    constexpr auto integer = binary_cut_features<10>::encode_integer(5);
    static_assert(integer == 5, "Integer encoding should return rel");
    
    constexpr auto normalized = binary_cut_features<10>::encode_normalized(5);
    static_assert(normalized.size() == 3, "Normalized size should be 3");
    
    constexpr auto full = binary_cut_features<10>::encode_full(5);
    static_assert(full.size() == 5, "Full encoding size should be 5");
}

// Test: Type Flexibility (float vs double)
TEST(BinaryCutFeatures, TypeFlexibility) {
    // Test with float
    {
        auto features = binary_cut_features<10>::encode_onehot<float>(5);
        EXPECT_EQ(features[4], 1.0f);
    }
    
    // Test with double (default)
    {
        auto features = binary_cut_features<10>::encode_onehot<double>(5);
        EXPECT_EQ(features[4], 1.0);
    }
}

// Test: Zero Allocation (Fixed-Size Returns)
TEST(BinaryCutFeatures, ZeroAllocation) {
    // All encodings return stack-allocated containers
    
    auto onehot = binary_cut_features<10>::encode_onehot(5);
    static_assert(std::is_same_v<decltype(onehot), std::array<double, 9>>);
    
    auto integer = binary_cut_features<10>::encode_integer(5);
    static_assert(std::is_same_v<decltype(integer), size_t>);
    
    auto normalized = binary_cut_features<10>::encode_normalized(5);
    static_assert(std::is_same_v<decltype(normalized), std::array<double, 3>>);
    
    auto full = binary_cut_features<10>::encode_full(5);
    static_assert(std::is_same_v<decltype(full), std::array<double, 5>>);
}

// Test: Normalized Ratio Monotonicity
TEST(BinaryCutFeatures, NormalizedRatioMonotonicity) {
    constexpr size_t Len = 20;
    
    double prev_ratio = 0.0;
    for (size_t rel = 1; rel < Len; ++rel) {
        auto features = binary_cut_features<Len>::encode_normalized<double>(rel);
        double ratio = features[2];
        
        // Law: Ratio increases with rel
        EXPECT_GT(ratio, prev_ratio);
        prev_ratio = ratio;
    }
}

// Test: Descriptor Integration
TEST(BinaryCutFeatures, DescriptorIntegration) {
    // Feature encoding works naturally with binary_cut_desc
    binary_cut_desc<10> desc;
    
    for (size_t ord = 0; ord < desc.size(); ++ord) {
        size_t rel = desc.ordinal_to_relative_cut(ord);
        auto features = binary_cut_features<10>::encode_normalized(rel);
        
        // Verify encoding is consistent
        EXPECT_DOUBLE_EQ(features[1], static_cast<double>(rel));
    }
}
