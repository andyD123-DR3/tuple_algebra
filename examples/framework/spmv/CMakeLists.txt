add_executable(spmv_tridiag_demo spmv_tridiag_demo.cpp)
target_include_directories(spmv_tridiag_demo PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/graph_lib
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_compile_features(spmv_tridiag_demo PRIVATE cxx_std_20)

if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(spmv_tridiag_demo PRIVATE -O3 -march=native)
else()
    target_compile_options(spmv_tridiag_demo PRIVATE /O2)
endif()
