cmake_minimum_required(VERSION 3.8)
project(jetnet LANGUAGES CXX CUDA)

# Request C++11 standard, using new CMake variables.
set (CMAKE_CXX_STANDARD 11)
set (CMAKE_CXX_STANDARD_REQUIRED True)

find_package (OpenCV REQUIRED)
find_package (CUDA REQUIRED)

# for YCM
#set( CMAKE_EXPORT_COMPILE_COMMANDS ON )

set(CMAKE_MODULE_PATH "${jetnet_SOURCE_DIR}/cmake")

include(build_type)

# automatic detection of compute arch
include(select_compute_arch)
CUDA_SELECT_NVCC_ARCH_FLAGS(DETECTED_COMPUTE_ARCH_FLAGS)
set (COMPUTE_ARCH_FLAGS ${DETECTED_COMPUTE_ARCH_FLAGS} CACHE STRING "Compute capability flags to compile for")

# append to cuda flags
string (REPLACE ";" " " COMPUTE_ARCH_FLAGS "${COMPUTE_ARCH_FLAGS}")
set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${COMPUTE_ARCH_FLAGS}")

set (CMAKE_WITH_ADDRESS_SANITIZER false CACHE BOOL "Enable address sanitizer")

if (WITH_ADDRESS_SANITIZER)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer -fsanitize=address")
    set (CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} -fno-omit-frame-pointer -fsanitize=address")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

message (STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message (STATUS "Asan: ${CMAKE_WITH_ADDRESS_SANITIZER}")

add_subdirectory (src)
add_subdirectory (examples)
