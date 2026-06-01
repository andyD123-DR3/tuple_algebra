#pragma once

#include <cstdlib>
#include <iostream>
#include <string_view>

inline void require(bool condition, std::string_view expression, const char* file, int line) {
    if (!condition) {
        std::cerr << file << ':' << line << ": requirement failed: " << expression << '\n';
        std::exit(1);
    }
}

#define SPMV_REQUIRE(expr) ::require(static_cast<bool>(expr), #expr, __FILE__, __LINE__)
