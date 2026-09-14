
find_package(CUDA REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)

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
    elseif (TEST_TARGET_NAME STREQUAL "test_postinit_owner_death"
            OR TEST_TARGET_NAME STREQUAL "test_fork_child_exit_cleanup"
            OR TEST_TARGET_NAME STREQUAL "test_pid_discovery")
        if (TEST_TARGET_NAME STREQUAL "test_postinit_owner_death"
                OR TEST_TARGET_NAME STREQUAL "test_fork_child_exit_cleanup")
            # Build this focused regression test with the production shared-region
            # implementation.  It does not invoke any CUDA/NVML entry point at
            # runtime.  Section garbage collection drops unrelated GPU-facing
            # production functions.
            add_executable(${TEST_TARGET_NAME}
                ${TEST_SCRIPT}
                ${CMAKE_CURRENT_SOURCE_DIR}/../src/multiprocess/multiprocess_memory_limit.c
                ${CMAKE_CURRENT_SOURCE_DIR}/../src/log_utils.c)
        else()
            add_executable(${TEST_TARGET_NAME}
                ${TEST_SCRIPT}
                ${CMAKE_CURRENT_SOURCE_DIR}/../src/utils.c
                ${CMAKE_CURRENT_SOURCE_DIR}/../src/log_utils.c)
        endif()
        target_compile_definitions(${TEST_TARGET_NAME} PRIVATE
            _GNU_SOURCE)
        target_compile_options(${TEST_TARGET_NAME} PRIVATE
            -ffunction-sections -fdata-sections)
        set_target_properties(${TEST_TARGET_NAME} PROPERTIES
            LINK_FLAGS "-Wl,--no-export-dynamic,--gc-sections")
    elseif (TEST_TARGET_NAME STREQUAL "test_nvml_v2_signatures")
        # Links the production NVML pass-through wrappers and supplies its own
        # entry table, so it needs no NVIDIA driver and no GPU at runtime.
        add_executable(${TEST_TARGET_NAME}
            ${TEST_SCRIPT}
            ${CMAKE_CURRENT_SOURCE_DIR}/../src/nvml/nvml_entry.c
            ${CMAKE_CURRENT_SOURCE_DIR}/../src/log_utils.c)
        target_compile_options(${TEST_TARGET_NAME} PRIVATE
            -ffunction-sections -fdata-sections)
        set_target_properties(${TEST_TARGET_NAME} PROPERTIES
            LINK_FLAGS "-Wl,--gc-sections")
    elseif (TEST_TARGET_NAME STREQUAL "test_dlsym_rtld_next")
        add_executable(${TEST_TARGET_NAME} ${TEST_SCRIPT})
    elseif (TEST_SCRIPT MATCHES ".*cu")
        cuda_add_executable(${TEST_TARGET_NAME}  ${TEST_SCRIPT})
    else()
        add_executable(${TEST_TARGET_NAME}  ${TEST_SCRIPT})
    endif()

    list(APPEND TEST_TARGET_NAMES_LIST ${TEST_TARGET_NAME})
    if (TEST_SCRIPT MATCHES "test_shared_region_concurrency.c"
            OR TEST_TARGET_NAME STREQUAL "test_postinit_owner_death"
            OR TEST_TARGET_NAME STREQUAL "test_fork_child_exit_cleanup"
            OR TEST_TARGET_NAME STREQUAL "test_pid_discovery")
        target_link_libraries(${TEST_TARGET_NAME} -lrt -lpthread)
    elseif (TEST_TARGET_NAME STREQUAL "test_dlsym_rtld_next")
        target_link_libraries(${TEST_TARGET_NAME} -ldl)
    elseif (TEST_TARGET_NAME STREQUAL "test_nvml_v2_signatures")
        target_link_libraries(${TEST_TARGET_NAME} -lpthread)
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

add_test(NAME fork_child_exit_cleanup
    COMMAND test_fork_child_exit_cleanup)
set_tests_properties(fork_child_exit_cleanup PROPERTIES TIMEOUT 20)

add_test(NAME dlsym_rtld_next
    COMMAND test_dlsym_rtld_next)
set_tests_properties(dlsym_rtld_next PROPERTIES TIMEOUT 10)

# Debug level is required: below it the LOG_DEBUG branch in NVML_OVERRIDE_CALL
# is skipped, nothing clobbers the argument register, and a v1-arity wrapper
# can forward the dropped argument by accident.
add_test(NAME nvml_v2_signatures
    COMMAND test_nvml_v2_signatures)
set_tests_properties(nvml_v2_signatures PROPERTIES
    ENVIRONMENT "LIBCUDA_LOG_LEVEL=4"
    TIMEOUT 10)

add_test(NAME pid_discovery
    COMMAND test_pid_discovery)
set_tests_properties(pid_discovery PROPERTIES TIMEOUT 10)
if (TARGET vgpu)
    add_test(NAME dlsym_rtld_next_preloaded
        COMMAND test_dlsym_rtld_next)
    set_tests_properties(dlsym_rtld_next_preloaded PROPERTIES
        ENVIRONMENT "LD_PRELOAD=$<TARGET_FILE:vgpu>"
        TIMEOUT 10)
endif()


add_custom_target(python_test ALL
    COMMAND cp -r ${CMAKE_CURRENT_SOURCE_DIR}/python ${CMAKE_CURRENT_BINARY_DIR})

# add_custom_target(multiprocess_test_scripts ALL
#    COMMAND cp -r ${CMAKE_CURRENT_SOURCE_DIR}/multiprocess/scripts
#                  ${CMAKE_CURRENT_BINARY_DIR}/multiprocess)
