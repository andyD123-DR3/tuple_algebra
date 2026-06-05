add_executable(spmv_tridiag_demo spmv_tridiag_demo.cpp)
target_include_directories(spmv_tridiag_demo PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/graph_lib
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_compile_features(spmv_tridiag_demo PRIVATE cxx_std_20)

set(CTDP_SPMV_TARGET_X86 FALSE)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64|i[3-6]86)$")
    set(CTDP_SPMV_TARGET_X86 TRUE)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR
   CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    if(CTDP_SPMV_TARGET_X86)
        target_compile_options(spmv_tridiag_demo PRIVATE -O3 -march=native)
    else()
        target_compile_options(spmv_tridiag_demo PRIVATE -O3)
    endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    target_compile_options(spmv_tridiag_demo PRIVATE -O3)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(spmv_tridiag_demo PRIVATE /O2)
endif()
