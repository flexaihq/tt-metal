# Common function to set up target properties for TTNN tests
function(setup_ttnn_test_target target_name)
    target_link_libraries(
        ${target_name}
        PUBLIC
            test_common_libs
            ttnn
            Metalium::Metal
            GTest::gmock_main
            Python3::Python
    )
    target_include_directories(
        ${target_name}
        PRIVATE
            ${UMD_HOME}
            ${PROJECT_SOURCE_DIR}
            ${PROJECT_SOURCE_DIR}/tt_metal
            ${PROJECT_SOURCE_DIR}/tests
            ${CMAKE_CURRENT_SOURCE_DIR}
            "$<TARGET_PROPERTY:TT::NN::CPP,INCLUDE_DIRECTORIES>"
    )
    set_target_properties(
        ${target_name}
        PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY
                ${PROJECT_BINARY_DIR}/test/ttnn
    )
endfunction()

set(Python3_FIND_STRATEGY LOCATION)
find_package(Python3 REQUIRED COMPONENTS Development)

add_subdirectory(unit_tests/gtests)
add_subdirectory(benchmark/cpp)
