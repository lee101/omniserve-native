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
    # -Wshadow and the prototype warnings catch the two mistakes that actually
    # bite in a threaded C server: a local quietly masking shared state, and a
    # call through an unchecked implicit declaration.
    add_compile_options(-fno-plt -Wshadow -Wstrict-prototypes -Wmissing-prototypes
                        -Wvla -Wpointer-arith -Wwrite-strings -Wundef
                        -Wold-style-definition -Wredundant-decls)
endif()

# Faster links and rebuilds when the tooling is present. None of this changes
# generated code; LTO stays under ONATIVE_LTO so a dev build can skip it (it is
# the dominant incremental cost: ~4.0s to ~1.4s on this tree).
find_program(ONATIVE_CCACHE ccache)
if(ONATIVE_CCACHE)
    set(CMAKE_C_COMPILER_LAUNCHER "${ONATIVE_CCACHE}")
    message(STATUS "using ccache: ${ONATIVE_CCACHE}")
endif()
find_program(ONATIVE_MOLD mold)
find_program(ONATIVE_LLD ld.lld)
if(ONATIVE_MOLD)
    add_link_options(-fuse-ld=mold)
    message(STATUS "using mold linker")
elseif(ONATIVE_LLD)
    add_link_options(-fuse-ld=lld)
    message(STATUS "using lld linker")
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
# Foreground estimation / colour decontamination. The CPU path is always built;
# the CUDA kernel is opt-in because CI has no nvcc.
option(WITH_MATTE_CUDA "Build the CUDA foreground-estimation kernel" OFF)
if(WITH_MATTE_CUDA)
    set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90;120" CACHE STRING "CUDA archs for the matte kernel")
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()
set(LLAMA_DIR "${CMAKE_SOURCE_DIR}/../llama.cpp" CACHE PATH "llama.cpp checkout")
set(SD_DIR "${CMAKE_SOURCE_DIR}/../stable-diffusion.cpp" CACHE PATH "stable-diffusion.cpp checkout")

set(CORE_SRC src/ocapacity.c src/ohttp.c src/ojson.c src/omatte.c src/oproxy.c src/osched.c
             src/oscale.c src/otext.c src/otune.c)

add_library(onative_core STATIC ${CORE_SRC})
target_include_directories(onative_core PUBLIC include)
target_link_libraries(onative_core PUBLIC pthread dl m)

add_executable(omniserve-native src/main.c src/docs.c src/backend_llama.c src/backend_sd.c)
target_include_directories(omniserve-native PRIVATE include)
target_link_libraries(omniserve-native PRIVATE onative_core)

if(WITH_LLAMA AND EXISTS "${LLAMA_DIR}/build/bin/libllama.so")
    target_compile_definitions(omniserve-native PRIVATE USE_LLAMA)
    target_include_directories(omniserve-native SYSTEM PRIVATE "${LLAMA_DIR}/include" "${LLAMA_DIR}/ggml/include")
    target_link_libraries(omniserve-native PRIVATE "${LLAMA_DIR}/build/bin/libllama.so" "${LLAMA_DIR}/build/bin/libggml.so")
    set_target_properties(omniserve-native PROPERTIES BUILD_RPATH "${LLAMA_DIR}/build/bin")
endif()

if(WITH_SD AND EXISTS "${SD_DIR}/build/bin/libstable-diffusion.so")
    target_compile_definitions(omniserve-native PRIVATE USE_SD "OMNISERVE_SD_LIB_DEFAULT=\"${SD_DIR}/build/bin/libstable-diffusion.so\"")
    # SYSTEM so the vendored stb/sd headers do not trip our own strict warning
    # set; they are third-party code and are not ours to fix.
    target_include_directories(omniserve-native SYSTEM BEFORE PRIVATE "${SD_DIR}/ggml/examples")
    target_include_directories(omniserve-native SYSTEM PRIVATE "${SD_DIR}/include")
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
    add_test(
        NAME birefnet_jobs
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/test_birefnet_jobs.py"
    )
    add_test(
        NAME object_store
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/test_object_store.py"
    )
    # Runs the real gateway with a stub upstream and no GGUF, so it needs no GPU.
    add_test(
        NAME internal_trust
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/test_internal_trust.py"
    )
    set_tests_properties(internal_trust PROPERTIES
        ENVIRONMENT "OMNISERVE_NATIVE_BIN=$<TARGET_FILE:omniserve-native>")
    # Skips itself when numpy or libomatte.so is missing (CPU-only CI).
    add_test(
        NAME matte_decontamination
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/test_matte_decontamination.py"
    )
    set_tests_properties(matte_decontamination PROPERTIES
        ENVIRONMENT "OMATTE_LIB=$<TARGET_FILE:omatte>")
endif()

if(WITH_MATTE_CUDA)
    target_sources(onative_core PRIVATE cuda/omatte_cuda.cu)
    target_compile_definitions(onative_core PUBLIC OMATTE_WITH_CUDA)
    target_link_libraries(onative_core PUBLIC CUDA::cudart)
endif()

# Shared build of just the matting library, for the Python workers (ctypes).
add_library(omatte SHARED src/omatte.c)
target_include_directories(omatte PUBLIC include)
target_link_libraries(omatte PUBLIC pthread m)
set_target_properties(omatte PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(WITH_MATTE_CUDA)
    target_sources(omatte PRIVATE cuda/omatte_cuda.cu)
    target_compile_definitions(omatte PUBLIC OMATTE_WITH_CUDA)
    target_link_libraries(omatte PUBLIC CUDA::cudart)
    set_target_properties(omatte PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()

add_executable(omatte_cli tools/omatte_cli.c)
target_include_directories(omatte_cli PRIVATE include)
target_link_libraries(omatte_cli PRIVATE onative_core)

add_executable(onative_bench tools/bench_http.c)
target_link_libraries(onative_bench PRIVATE pthread)
