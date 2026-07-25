cmake_minimum_required(VERSION 3.20)
project(omniserve_native C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(ONATIVE_NATIVE_ARCH "Tune generated code for the build host" OFF)
option(ONATIVE_LTO "Enable release link-time optimization" ON)
option(ONATIVE_PROFILE "Keep symbols and frame pointers in optimized profiling builds" OFF)

add_compile_options(-Wall -Wextra -pthread)
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-fno-plt)
endif()
if(ONATIVE_NATIVE_ARCH AND CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-march=native -mtune=native)
endif()
if(ONATIVE_PROFILE AND CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-g -fno-omit-frame-pointer)
    add_link_options(-g -fno-omit-frame-pointer)
endif()
if(ONATIVE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT ONATIVE_IPO_SUPPORTED OUTPUT ONATIVE_IPO_ERROR LANGUAGES C)
    if(ONATIVE_IPO_SUPPORTED)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
    else()
        message(WARNING "LTO unavailable: ${ONATIVE_IPO_ERROR}")
    endif()
endif()

option(WITH_LLAMA "Link llama.cpp for LLM inference" ON)
option(WITH_SD "Link stable-diffusion.cpp for image generation" OFF)
set(LLAMA_DIR "${CMAKE_SOURCE_DIR}/../llama.cpp" CACHE PATH "llama.cpp checkout")
set(SD_DIR "${CMAKE_SOURCE_DIR}/../stable-diffusion.cpp" CACHE PATH "stable-diffusion.cpp checkout")

set(CORE_SRC src/ocapacity.c src/ohttp.c src/ojson.c src/oproxy.c src/osched.c
             src/oscale.c src/otext.c src/otune.c)

add_library(onative_core STATIC ${CORE_SRC})
target_include_directories(onative_core PUBLIC include)
target_link_libraries(onative_core PUBLIC pthread dl m)

add_executable(omniserve-native src/main.c src/docs.c src/backend_llama.c src/backend_sd.c)
target_include_directories(omniserve-native PRIVATE include)
target_link_libraries(omniserve-native PRIVATE onative_core)

if(WITH_LLAMA AND EXISTS "${LLAMA_DIR}/build/bin/libllama.so")
    target_compile_definitions(omniserve-native PRIVATE USE_LLAMA)
    target_include_directories(omniserve-native PRIVATE "${LLAMA_DIR}/include" "${LLAMA_DIR}/ggml/include")
    target_link_libraries(omniserve-native PRIVATE "${LLAMA_DIR}/build/bin/libllama.so" "${LLAMA_DIR}/build/bin/libggml.so")
    set_target_properties(omniserve-native PROPERTIES BUILD_RPATH "${LLAMA_DIR}/build/bin")
endif()

if(WITH_SD AND EXISTS "${SD_DIR}/build/bin/libstable-diffusion.so")
    target_compile_definitions(omniserve-native PRIVATE USE_SD "OMNISERVE_SD_LIB_DEFAULT=\"${SD_DIR}/build/bin/libstable-diffusion.so\"")
    target_include_directories(omniserve-native BEFORE PRIVATE "${SD_DIR}/ggml/examples")
    target_include_directories(omniserve-native PRIVATE "${SD_DIR}/include")
    target_include_directories(omniserve-native SYSTEM PRIVATE "${SD_DIR}/thirdparty")
endif()

enable_testing()
add_executable(onative_tests tests/test_core.c src/backend_llama.c src/docs.c)
target_include_directories(onative_tests PRIVATE include)
target_link_libraries(onative_tests PRIVATE onative_core)
add_test(NAME core COMMAND onative_tests)
find_package(Python3 COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
    add_test(
        NAME trellis2_worker
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/test_trellis2_worker.py"
    )
    add_test(
        NAME asr_training
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/test_asr_training.py"
    )
    add_test(
        NAME asr_router
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/test_asr_router.py"
    )
endif()

add_executable(onative_bench tools/bench_http.c)
target_link_libraries(onative_bench PRIVATE pthread)
