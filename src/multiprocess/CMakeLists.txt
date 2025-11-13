
add_library(multiprocess_mod OBJECT multiprocess_memory_limit.c multiprocess_utilization_watcher.c)
target_compile_options(multiprocess_mod PUBLIC ${LIBRARY_COMPILE_FLAGS})
target_link_libraries(multiprocess_mod PUBLIC nvidia-ml)

add_executable(shrreg-tool shrreg_tool.c)
# Link log_file.c to provide fp1 for logging (minimal, no CUDA dependencies)
target_sources(shrreg-tool PRIVATE ${CMAKE_SOURCE_DIR}/src/log_file.c)
target_link_libraries(shrreg-tool multiprocess_mod -lpthread -lcuda -lnvidia-ml)

