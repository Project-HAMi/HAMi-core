
find_package(CUDA REQUIRED)

set(TEST_CPP_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(TEST_TARGET_NAMES_LIST) 


file(GLOB_RECURSE TEST_SCRIPTS "${TEST_CPP_SOURCE_DIR}/*.c" "${TEST_CPP_SOURCE_DIR}/*.cu")
foreach(TEST_SCRIPT ${TEST_SCRIPTS}) 
    file(RELATIVE_PATH RELATIVE_TEST_PATH ${TEST_CPP_SOURCE_DIR} ${TEST_SCRIPT})
    get_filename_component(TEST_TARGET_DIR ${RELATIVE_TEST_PATH} DIRECTORY)
    get_filename_component(TEST_TARGET_NAME ${RELATIVE_TEST_PATH} NAME_WE)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${TEST_TARGET_DIR})

    if (TEST_SCRIPT MATCHES "test_shared_region_concurrency.c")
        add_executable(${TEST_TARGET_NAME} ${TEST_SCRIPT}
            ${CMAKE_CURRENT_SOURCE_DIR}/../src/multiprocess/multiprocess_memory_limit.c
            ${CMAKE_CURRENT_SOURCE_DIR}/../src/log_utils.c)
        target_compile_definitions(${TEST_TARGET_NAME} PRIVATE
            SHRREG_SEQUENCE_POINT_CALLBACK=shrreg_test_sequence_point)
        target_compile_options(${TEST_TARGET_NAME} PRIVATE
            -D_GNU_SOURCE -ffunction-sections -fdata-sections)
        set_target_properties(${TEST_TARGET_NAME} PROPERTIES
            LINK_FLAGS "-Wl,--no-export-dynamic,--gc-sections")
    elseif (TEST_TARGET_NAME STREQUAL "test_postinit_owner_death")
        # Build this focused regression test with the production shared-region
        # implementation.  It does not invoke any CUDA/NVML entry point at
        # runtime.  Section garbage collection drops unrelated GPU-facing
        # production functions.
        add_executable(${TEST_TARGET_NAME}
            ${TEST_SCRIPT}
            ${CMAKE_CURRENT_SOURCE_DIR}/../src/multiprocess/multiprocess_memory_limit.c
            ${CMAKE_CURRENT_SOURCE_DIR}/../src/log_utils.c)
        target_compile_definitions(${TEST_TARGET_NAME} PRIVATE
            _GNU_SOURCE)
        target_compile_options(${TEST_TARGET_NAME} PRIVATE
            -ffunction-sections -fdata-sections)
        set_target_properties(${TEST_TARGET_NAME} PROPERTIES
            LINK_FLAGS "-Wl,--no-export-dynamic,--gc-sections")
    elseif (TEST_SCRIPT MATCHES ".*cu")
        cuda_add_executable(${TEST_TARGET_NAME}  ${TEST_SCRIPT})
    else()
        add_executable(${TEST_TARGET_NAME}  ${TEST_SCRIPT})
    endif()

    list(APPEND TEST_TARGET_NAMES_LIST ${TEST_TARGET_NAME})
    if (TEST_SCRIPT MATCHES "test_shared_region_concurrency.c" OR
            TEST_TARGET_NAME STREQUAL "test_postinit_owner_death")
        target_link_libraries(${TEST_TARGET_NAME} -lrt -lpthread)
    else()
        target_link_libraries(${TEST_TARGET_NAME} -lrt -lpthread
            -lnvidia-ml -lcuda -lcudart -L${CUDA_HOME}/lib64)
    endif()
    target_compile_options(${TEST_TARGET_NAME} PUBLIC ${TEST_COMPILE_FLAGS})

    if (DEFINED TEST_DEVICE_ID)
    	target_compile_definitions(${TEST_TARGET_NAME} PUBLIC -DTEST_DEVICE_ID=${TEST_DEVICE_ID})
	endif()
endforeach()
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

add_test(NAME shared_region_concurrency
    COMMAND test_shared_region_concurrency)
# The test waits on the start barrier and on registration in sequence, each
# bounded by SHRREG_TEST_TIMEOUT_MS (20s default), so it can spend 40s before
# printing which stage stalled. Keep this above that budget or ctest kills it
# first and the diagnostic is lost.
set_tests_properties(shared_region_concurrency PROPERTIES TIMEOUT 90)
add_test(NAME postinit_owner_death
    COMMAND test_postinit_owner_death)
set_tests_properties(postinit_owner_death PROPERTIES TIMEOUT 20)


add_custom_target(python_test ALL
    COMMAND cp -r ${CMAKE_CURRENT_SOURCE_DIR}/python ${CMAKE_CURRENT_BINARY_DIR})

# add_custom_target(multiprocess_test_scripts ALL
#    COMMAND cp -r ${CMAKE_CURRENT_SOURCE_DIR}/multiprocess/scripts
#                  ${CMAKE_CURRENT_BINARY_DIR}/multiprocess)
