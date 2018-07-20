include_directories (
    /usr/local/cuda/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/engine
    ${CMAKE_SOURCE_DIR}/src/models
    ${CMAKE_SOURCE_DIR}/src/models/layers
    ${CMAKE_SOURCE_DIR}/src/utils
    ${OpenCV_INCLUDE_DIRS}
)

link_libraries (
    jetnet_static
    ${jetnet_LIBS}
)

add_executable (jetnet_build_yolov2 build_yolov2.cpp)

install (TARGETS
    jetnet_build_yolov2
    DESTINATION "bin"
)
