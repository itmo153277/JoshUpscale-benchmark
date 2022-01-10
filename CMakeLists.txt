cmake_minimum_required(VERSION 3.12)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
project(benchmark VERSION 1.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(PNG REQUIRED)
find_package(yaml-cpp 0.7.0 REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(TensorRT REQUIRED)
find_package(tensorflow REQUIRED)
find_package(onnxruntime REQUIRED)

if(MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

option(ENABLE_LINT "Enable linting" OFF)
if(ENABLE_LINT)
  if(MSVC)
    add_compile_options(/W4 /WX)
  else()
    add_compile_options(-Wall -Wextra -Wpedantic -Werror)
  endif()
  if (NOT MSVC)
    find_program(CLANG_TIDY "clang-tidy" REQUIRED)
    find_program(CPPLINT "cpplint" REQUIRED)
    find_program(CPPCHECK "cppcheck" REQUIRED)
    set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY};--checks=-clang-diagnostic-unused-command-line-argument;--warnings-as-errors=*")
    set(CMAKE_CXX_CPPLINT "${CPPLINT};--quiet")
    set(CMAKE_CXX_CPPCHECK "${CPPCHECK};-q;--enable=warning;--error-exitcode=1")
  endif()
endif()
file(GLOB_RECURSE SRC src/*.cc)
file(GLOB_RECURSE HEADERS include/*.h)
add_executable(benchmark ${SRC} ${HEADERS})
target_include_directories(benchmark PRIVATE include)
target_link_libraries(benchmark PRIVATE
  PNG::PNG
  yaml-cpp
  CUDA::cudart
  ${TensorRT_LIBS}
  tensorflow
  onnxruntime
  onnxruntime::cuda
  onnxruntime::tensorrt
)
if (ENABLE_LINT)
  if (MSVC)
    set_target_properties(benchmark PROPERTIES
      VS_GLOBAL_RunCodeAnalysis true
      VS_GLOBAL_EnableMicrosoftCodeAnalysis true
      VS_GLOBAL_EnableClangTidyCodeAnalysis true
    )
  endif()
endif()
