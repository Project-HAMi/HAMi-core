
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

    if (TEST_TARGET_NAME STREQUAL "test_postinit_owner_death" OR TEST_TARGET_NAME STREQUAL "test_pid_discovery")
        if (TEST_TARGET_NAME STREQUAL "test_postinit_owner_death")
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
    elseif (TEST_TARGET_NAME STREQUAL "test_alloc_async_free_wrong_device")
        # Needs get_gpu_memory_usage() from multiprocess_memory_limit.c linked
        # directly into the test binary (same reasoning as
        # test_postinit_owner_death above), on top of the normal CUDA driver
        # link every allocation test needs. -ffunction-sections/--gc-sections
        # let the linker drop the parts of multiprocess_memory_limit.c this
        # test never calls, instead of requiring their own transitive deps
        # (e.g. cuda_to_nvml_map from multiprocess_utilization_watcher.c).
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
    elseif (TEST_TARGET_NAME STREQUAL "test_dlsym_rtld_next")
        add_executable(${TEST_TARGET_NAME} ${TEST_SCRIPT})
    elseif (TEST_SCRIPT MATCHES ".*cu")
        cuda_add_executable(${TEST_TARGET_NAME}  ${TEST_SCRIPT})
    else()
        add_executable(${TEST_TARGET_NAME}  ${TEST_SCRIPT})
    endif()

    list(APPEND TEST_TARGET_NAMES_LIST ${TEST_TARGET_NAME})
    if (TEST_TARGET_NAME STREQUAL "test_postinit_owner_death" OR TEST_TARGET_NAME STREQUAL "test_pid_discovery")
        target_link_libraries(${TEST_TARGET_NAME} -lrt -lpthread)
    elseif (TEST_TARGET_NAME STREQUAL "test_dlsym_rtld_next")
        target_link_libraries(${TEST_TARGET_NAME} -ldl)
    else()
        # test_alloc_async_free_wrong_device also lands here: it needs the
        # same CUDA driver link as every other allocation test.
        target_link_libraries(${TEST_TARGET_NAME} -lrt -lpthread
            -lnvidia-ml -lcuda -lcudart -L${CUDA_HOME}/lib64)
    endif()
    target_compile_options(${TEST_TARGET_NAME} PUBLIC ${TEST_COMPILE_FLAGS})

    if (DEFINED TEST_DEVICE_ID)
    	target_compile_definitions(${TEST_TARGET_NAME} PUBLIC -DTEST_DEVICE_ID=${TEST_DEVICE_ID})
	endif()
endforeach()
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

add_test(NAME postinit_owner_death
    COMMAND test_postinit_owner_death)
set_tests_properties(postinit_owner_death PROPERTIES TIMEOUT 20)

add_test(NAME dlsym_rtld_next
    COMMAND test_dlsym_rtld_next)
set_tests_properties(dlsym_rtld_next PROPERTIES TIMEOUT 10)

add_test(NAME pid_discovery
    COMMAND test_pid_discovery)
set_tests_properties(pid_discovery PROPERTIES TIMEOUT 10)
if (TARGET vgpu)
    add_test(NAME dlsym_rtld_next_preloaded
        COMMAND test_dlsym_rtld_next)
    set_tests_properties(dlsym_rtld_next_preloaded PROPERTIES
        ENVIRONMENT "LD_PRELOAD=$<TARGET_FILE:vgpu>"
        TIMEOUT 10)

    # Needs >= 2 real GPUs to reproduce the cross-device free; the test
    # itself detects a single-GPU environment and exits with
    # SKIP_EXIT_CODE (125) instead of failing -- SKIP_RETURN_CODE tells
    # CTest to report that as skipped rather than passed/failed.
    add_test(NAME alloc_async_free_wrong_device
        COMMAND test_alloc_async_free_wrong_device)
    set_tests_properties(alloc_async_free_wrong_device PROPERTIES
        ENVIRONMENT "LD_PRELOAD=$<TARGET_FILE:vgpu>"
        SKIP_RETURN_CODE 125
        TIMEOUT 30)
endif()


add_custom_target(python_test ALL
    COMMAND cp -r ${CMAKE_CURRENT_SOURCE_DIR}/python ${CMAKE_CURRENT_BINARY_DIR})

# add_custom_target(multiprocess_test_scripts ALL
#    COMMAND cp -r ${CMAKE_CURRENT_SOURCE_DIR}/multiprocess/scripts
#                  ${CMAKE_CURRENT_BINARY_DIR}/multiprocess)
