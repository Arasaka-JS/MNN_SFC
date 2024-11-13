set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx -mfma")


message("SFC Test00")
add_definitions(-DMNN_PIPELINE_DEBUG)
add_definitions(-DLOG_VERBOSE)
message(STATUS "MNN Depends:" ${MNN_DEPS})
add_executable(main.out ${CMAKE_CURRENT_LIST_DIR}/main.cpp)
target_link_libraries(main.out ${MNN_DEPS})
# add_executable(test_avx2_sfc.out ${CMAKE_CURRENT_LIST_DIR}/test_avx2_sfc.cpp)
# target_link_libraries(test_avx2_sfc.out ${MNN_DEPS})

