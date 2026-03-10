add_executable(test_spmv_app_types test_spmv_app_types.cpp)
target_link_libraries(test_spmv_app_types PRIVATE GTest::gtest_main)
target_include_directories(test_spmv_app_types PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/graph_lib
    ${PROJECT_SOURCE_DIR}/examples/framework/spmv
)
target_compile_features(test_spmv_app_types PRIVATE cxx_std_20)
add_test(NAME spmv_app_types_tests COMMAND test_spmv_app_types)

add_executable(test_spmv_schema test_spmv_schema.cpp)
target_link_libraries(test_spmv_schema PRIVATE GTest::gtest_main)
target_include_directories(test_spmv_schema PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/graph_lib
    ${PROJECT_SOURCE_DIR}/examples/framework/spmv
)
target_compile_features(test_spmv_schema PRIVATE cxx_std_20)
add_test(NAME spmv_schema_tests COMMAND test_spmv_schema)

add_executable(test_spmv_bench test_spmv_bench.cpp)
target_link_libraries(test_spmv_bench PRIVATE GTest::gtest_main)
target_include_directories(test_spmv_bench PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/graph_lib
    ${PROJECT_SOURCE_DIR}/examples/framework/spmv
)
target_compile_features(test_spmv_bench PRIVATE cxx_std_20)
add_test(NAME spmv_bench_tests COMMAND test_spmv_bench)

