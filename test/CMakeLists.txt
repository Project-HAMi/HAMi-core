
find_package(CUDA REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)

set(TEST_CPP_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(TEST_TARGET_NAMES_LIST) 


file(GLOB_RECURSE TEST_SCRIPTS "${TEST_CPP_SOURCE_DIR}/*.c" "${TEST_CPP_SOURCE_DIR}/*.cu")

# fault_inject_driver.c is not a test executable: it is a fault-injecting
# stand-in for libcuda.so.1 (see the file itself for why), built as a shared
# library below and consumed only by the alloc_async_failure_leak test.
set(FAULT_INJECT_DRIVER_SRC ${TEST_CPP_SOURCE_DIR}/fault_inject_driver.c)
list(REMOVE_ITEM TEST_SCRIPTS ${FAULT_INJECT_DRIVER_SRC})

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
        target_link_libraries(${TEST_TARGET_NAME} -lrt -lpthread
            -lnvidia-ml -lcuda -lcudart -L${CUDA_HOME}/lib64)
    endif()
    target_compile_options(${TEST_TARGET_NAME} PUBLIC ${TEST_COMPILE_FLAGS})

    if (DEFINED TEST_DEVICE_ID)
    	target_compile_definitions(${TEST_TARGET_NAME} PUBLIC -DTEST_DEVICE_ID=${TEST_DEVICE_ID})
	endif()
endforeach()
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# Build the fault-injecting libcuda.so.1 stand-in used by
# test_alloc_async_failure_leak. It must be named exactly "libcuda.so.1"
# (no extra version suffix) and live in its own directory so that directory
# can be prepended to LD_LIBRARY_PATH for just that one test, without
# shadowing the real driver for every other test.
set(FAULT_INJECT_DRIVER_DIR ${CMAKE_CURRENT_BINARY_DIR}/fault_inject_driver)
add_library(fault_inject_driver SHARED ${FAULT_INJECT_DRIVER_SRC})
set_target_properties(fault_inject_driver PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${FAULT_INJECT_DRIVER_DIR}
    OUTPUT_NAME cuda
    PREFIX lib
    SUFFIX .so.1
    POSITION_INDEPENDENT_CODE ON)
target_link_libraries(fault_inject_driver -ldl)

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

    # LD_PRELOAD routes cuMemAllocAsync/cuMemAllocFromPoolAsync through
    # libvgpu.so's real hooks, and LD_LIBRARY_PATH puts the fault-injecting
    # libcuda.so.1 stand-in ahead of the real driver so load_cuda_libraries()
    # picks it up. Needs a real GPU to run; not exercised in CI (see
    # fault_inject_driver.c and test_alloc_async_failure_leak.c for details).
    add_test(NAME alloc_async_failure_leak
        COMMAND test_alloc_async_failure_leak)
    set_tests_properties(alloc_async_failure_leak PROPERTIES
        ENVIRONMENT "LD_PRELOAD=$<TARGET_FILE:vgpu>;LD_LIBRARY_PATH=${FAULT_INJECT_DRIVER_DIR}:$ENV{LD_LIBRARY_PATH}"
        TIMEOUT 30)
    add_dependencies(test_alloc_async_failure_leak fault_inject_driver)
endif()


add_custom_target(python_test ALL
    COMMAND cp -r ${CMAKE_CURRENT_SOURCE_DIR}/python ${CMAKE_CURRENT_BINARY_DIR})

# add_custom_target(multiprocess_test_scripts ALL
#    COMMAND cp -r ${CMAKE_CURRENT_SOURCE_DIR}/multiprocess/scripts
#                  ${CMAKE_CURRENT_BINARY_DIR}/multiprocess)
