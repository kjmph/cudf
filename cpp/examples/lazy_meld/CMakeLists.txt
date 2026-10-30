# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

cmake_minimum_required(VERSION 3.24)
project(cudf_lazy_meld LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(LAZY_MELD_BUILD_COMPILER "Build the CPU/PTX compiler (needs LLVM/MLIR 21 and Arrow)" ON)
option(LAZY_MELD_BUILD_GPU "Build the cuDF/CUDA executor and GPU checks" ON)
include(CTest)

add_library(cudf_lazy_plan STATIC src/plan.cpp src/graph.cpp)
add_library(cudf::experimental_execution ALIAS cudf_lazy_plan)
target_include_directories(cudf_lazy_plan PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(cudf_lazy_plan PRIVATE -Wall -Wextra -Wpedantic)

if(BUILD_TESTING)
  add_executable(lazy_plan_test tests/plan_test.cpp)
  target_link_libraries(lazy_plan_test PRIVATE cudf_lazy_plan)
  add_test(NAME lazy_plan_test COMMAND lazy_plan_test)
  add_executable(graph_plan_test tests/graph_plan_test.cpp)
  target_link_libraries(graph_plan_test PRIVATE cudf_lazy_plan)
  add_test(NAME graph_plan_test COMMAND graph_plan_test)
endif()

if(LAZY_MELD_BUILD_COMPILER)
  get_filename_component(default_meld "${CMAKE_CURRENT_SOURCE_DIR}/../../../../meld" ABSOLUTE)
  set(MELD_SOURCE_DIR
      "${default_meld}"
      CACHE PATH "Sibling Meld source checkout"
  )
  if(NOT EXISTS "${MELD_SOURCE_DIR}/include/meld/Execution/PtxCompiler.h")
    message(FATAL_ERROR "Set MELD_SOURCE_DIR to the Meld source checkout")
  endif()
  # Embed only Meld's compiler. Its host-Arrow GPU runner is deliberately not used.
  set(MELD_ENABLE_NVPTX
      ON
      CACHE BOOL "" FORCE
  )
  set(MELD_ENABLE_CUDA
      OFF
      CACHE BOOL "" FORCE
  )
  set(MELD_ENABLE_CUDF
      OFF
      CACHE BOOL "" FORCE
  )
  set(MELD_ENABLE_DEVICE_RUNTIME
      OFF
      CACHE BOOL "" FORCE
  )
  set(MELD_BUILD_APPS
      OFF
      CACHE BOOL "" FORCE
  )
  set(CMAKE_FIND_PACKAGE_TARGETS_GLOBAL TRUE)
  add_subdirectory("${MELD_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/meld" EXCLUDE_FROM_ALL)
  find_package(MLIR REQUIRED CONFIG)
  if(NOT LLVM_VERSION_MAJOR EQUAL 21)
    message(FATAL_ERROR "This proof of concept is built against Meld's LLVM/MLIR 21 API")
  endif()
  add_library(cudf_lazy_meld_compiler STATIC src/compiler.cpp)
  target_include_directories(
    cudf_lazy_meld_compiler PRIVATE "${MELD_SOURCE_DIR}/include"
                                    "${CMAKE_CURRENT_BINARY_DIR}/meld/include"
  )
  target_include_directories(
    cudf_lazy_meld_compiler SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS}
  )
  # Match LLVM's RTTI mode without applying LLVM flags to the cuDF/CUDA translation units.
  if(NOT LLVM_ENABLE_RTTI)
    target_compile_options(cudf_lazy_meld_compiler PRIVATE -fno-rtti)
  endif()
  target_link_libraries(
    cudf_lazy_meld_compiler PRIVATE MLIRMeldExecution MLIRParser MeldBuiltinLowering
  )
  find_package(Threads REQUIRED)
  add_library(cudf_lazy_meld_cpu STATIC src/cpu_executor.cpp src/graph_cpu.cpp)
  add_library(cudf::lazy_meld_cpu ALIAS cudf_lazy_meld_cpu)
  add_library(cudf::experimental_execution_cpu ALIAS cudf_lazy_meld_cpu)
  target_include_directories(cudf_lazy_meld_cpu PRIVATE "${MELD_SOURCE_DIR}/include")
  target_link_libraries(
    cudf_lazy_meld_cpu
    PUBLIC cudf_lazy_plan
    PRIVATE cudf_lazy_meld_compiler Threads::Threads
  )
  add_executable(graph_demo tests/graph_demo.cpp)
  target_link_libraries(graph_demo PRIVATE cudf_lazy_meld_cpu)
  add_executable(q6_demo tests/q6_demo.cpp)
  target_link_libraries(q6_demo PRIVATE cudf_lazy_meld_cpu)
  if(BUILD_TESTING)
    add_executable(lazy_compiler_test tests/compiler_test.cpp)
    target_include_directories(lazy_compiler_test PRIVATE src "${MELD_SOURCE_DIR}/include")
    target_link_libraries(lazy_compiler_test PRIVATE cudf_lazy_plan cudf_lazy_meld_compiler)
    add_test(NAME lazy_compiler_test COMMAND lazy_compiler_test)
    add_executable(lazy_cpu_test tests/cpu_test.cpp)
    target_link_libraries(lazy_cpu_test PRIVATE cudf_lazy_meld_cpu)
    add_test(NAME lazy_cpu_test COMMAND lazy_cpu_test)
    add_executable(graph_cpu_test tests/graph_cpu_test.cpp)
    target_link_libraries(graph_cpu_test PRIVATE cudf_lazy_meld_cpu)
    add_test(NAME graph_cpu_test COMMAND graph_cpu_test)
    add_executable(date_test tests/date_test.cpp)
    target_link_libraries(date_test PRIVATE cudf_lazy_meld_cpu)
    add_test(NAME date_test COMMAND date_test)
  endif()
endif()

if(LAZY_MELD_BUILD_GPU)
  if(NOT LAZY_MELD_BUILD_COMPILER)
    message(FATAL_ERROR "LAZY_MELD_BUILD_GPU requires LAZY_MELD_BUILD_COMPILER")
  endif()
  if(CMAKE_VERSION VERSION_LESS 4.0)
    message(FATAL_ERROR "The cuDF GPU adapter requires CMake 4.0 or later")
  endif()
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  foreach(dependency CCCL nvtx3 rapids_logger rmm bs_thread_pool kvikio nvcomp)
    find_package(${dependency} CONFIG QUIET)
  endforeach()
  file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/../../../VERSION" cudf_source_version LIMIT_COUNT 1)
  string(REGEX MATCH "^[0-9]+[.][0-9]+" cudf_required_version "${cudf_source_version}")
  find_package(cudf ${cudf_required_version} REQUIRED CONFIG)
  add_library(cudf_lazy_meld STATIC src/executor.cpp src/graph_gpu.cpp src/kernels.cu)
  add_library(cudf::lazy_meld ALIAS cudf_lazy_meld)
  add_library(cudf::experimental_execution_gpu ALIAS cudf_lazy_meld)
  target_include_directories(cudf_lazy_meld PRIVATE "${MELD_SOURCE_DIR}/include")
  target_link_libraries(
    cudf_lazy_meld
    PUBLIC cudf_lazy_plan cudf::cudf
    PRIVATE cudf_lazy_meld_compiler CUDA::cuda_driver CUDA::cudart
  )
  set_target_properties(cudf_lazy_meld PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
  if(BUILD_TESTING)
    add_executable(lazy_gpu_test tests/gpu_test.cpp)
    target_link_libraries(lazy_gpu_test PRIVATE cudf_lazy_meld CUDA::cudart)
    add_test(NAME lazy_gpu_test COMMAND lazy_gpu_test)
    set_tests_properties(lazy_gpu_test PROPERTIES LABELS GPU)
    add_executable(graph_gpu_test tests/graph_gpu_test.cpp)
    target_link_libraries(graph_gpu_test PRIVATE cudf_lazy_meld cudf_lazy_meld_cpu CUDA::cudart)
    add_test(NAME graph_gpu_test COMMAND graph_gpu_test)
    set_tests_properties(graph_gpu_test PROPERTIES LABELS GPU)
  endif()
endif()
