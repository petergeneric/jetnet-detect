project(jetnet)
cmake_minimum_required(VERSION 2.8)

# Request C++11 standard, using new CMake variables.
set (CMAKE_CXX_STANDARD 11)
set (CMAKE_CXX_STANDARD_REQUIRED True)

# for YCM
#set( CMAKE_EXPORT_COMPILE_COMMANDS ON )

set(CMAKE_MODULE_PATH "${jetnet_SOURCE_DIR}/cmake")
include(build_type)

set (CMAKE_WITH_ADDRESS_SANITIZER false CACHE BOOL "Enable address sanitizer")

if (WITH_ADDRESS_SANITIZER)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer -fsanitize=address")
    set (CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} -fno-omit-frame-pointer -fsanitize=address")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

message (STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message (STATUS "Asan: ${CMAKE_WITH_ADDRESS_SANITIZER}")

find_package (OpenCV REQUIRED)
find_package (CUDA REQUIRED)

add_subdirectory (src)
add_subdirectory (examples)
