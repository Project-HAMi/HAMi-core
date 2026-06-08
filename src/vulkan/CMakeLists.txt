find_path(VULKAN_HEADERS vulkan/vulkan.h
          HINTS ENV VULKAN_SDK
          PATH_SUFFIXES include
          PATHS /usr/include /usr/local/include)
if(NOT VULKAN_HEADERS)
    message(FATAL_ERROR "vulkan/vulkan.h not found. Install libvulkan-dev or set VULKAN_SDK.")
endif()

add_library(vulkan_mod OBJECT
    layer.c
    dispatch.c
    hooks_memory.c
    hooks_alloc.c
    hooks_submit.c
    throttle_adapter.c
    budget.c
    physdev_index.c
)

target_include_directories(vulkan_mod PRIVATE
    ${VULKAN_HEADERS}
    ${CMAKE_SOURCE_DIR}/src
)

target_compile_options(vulkan_mod PUBLIC ${LIBRARY_COMPILE_FLAGS})
