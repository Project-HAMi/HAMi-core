cmake_minimum_required (VERSION 2.8.11)
project (libvgpu)


# set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
#set(CUDA_FILES libcuda_hook.c libcuda_context.c libcuda_device.c libcuda_stream.c libcuda_memory.c)

add_subdirectory(multiprocess)
add_subdirectory(allocator)
add_subdirectory(cuda)
add_subdirectory(nvml)
add_subdirectory(vulkan)

set(LIBVGPU vgpu)
# libvgpu.so: HAMi-core only. Vulkan layer code now lives in libvgpu_vk.so.
add_library(${LIBVGPU} SHARED libvgpu.c utils.c log_utils.c hami_core_export.c $<TARGET_OBJECTS:nvml_mod> $<TARGET_OBJECTS:cuda_mod> $<TARGET_OBJECTS:allocator_mod> $<TARGET_OBJECTS:multiprocess_mod>)
target_compile_options(${LIBVGPU} PUBLIC ${LIBRARY_COMPILE_FLAGS})
# Activate NVML dlsym redirect (libvgpu.c:#ifdef HOOK_NVML_ENABLE).
# Without this define the dispatcher in dlsym() falls through to the real
# libnvidia-ml so consumers like nvidia-smi / Isaac Sim Kit see the raw
# 46 GiB heap instead of the partitioned limit, which is inconsistent with
# the Vulkan/CUDA paths and trips Kit asserts during streaming init.
target_compile_definitions(${LIBVGPU} PUBLIC HOOK_NVML_ENABLE)
target_link_libraries(${LIBVGPU} PUBLIC -lcuda -lnvidia-ml)

# libvgpu_vk.so: Vulkan implicit-layer code. Activated via
# /etc/vulkan/implicit_layer.d/hami.json (see share/hami/hami.json).
# DT_NEEDED links libvgpu.so so the loader resolves the hami_core_*
# wrappers when the Vulkan loader dlopen()s us.
set(LIBVGPU_VK vgpu_vk)
add_library(${LIBVGPU_VK} SHARED $<TARGET_OBJECTS:vulkan_mod>)
target_compile_options(${LIBVGPU_VK} PUBLIC ${LIBRARY_COMPILE_FLAGS})
target_link_libraries(${LIBVGPU_VK} PUBLIC ${LIBVGPU} -lpthread)

if (NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
add_custom_target(strip_symbol ALL
    COMMAND strip -x ${CMAKE_BINARY_DIR}/lib${LIBVGPU}.so
    COMMAND strip -x ${CMAKE_BINARY_DIR}/lib${LIBVGPU_VK}.so
    DEPENDS ${LIBVGPU} ${LIBVGPU_VK})
endif()
