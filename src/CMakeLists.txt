set (SOURCES
    utils/logger.cpp
    utils/file_io.cpp
    utils/fp16.h
    models/darknet_weights_loader.cpp
    layers/leaky_relu_native.cpp
    layers/leaky_relu_plugin.cpp
    layers/conv2d_batch_leaky.cpp
)

set (SRC ${SOURCES})

include_directories (
    /usr/local/cuda/include
    ${CMAKE_SOURCE_DIR}/src/utils
    ${CMAKE_SOURCE_DIR}/src/layers
    ${CMAKE_SOURCE_DIR}/src/models
    ${OpenCV_INCLUDE_DIRS}
)

add_library (jetnet_static STATIC ${SRC})
add_library (jetnet SHARED ${SRC})

set (jetnet_LIBS
    -lm;
    -lstdc++;
    ${OpenCV_LIBS};
    -lnvinfer_plugin;
    -lnvinfer;
    -lcuda;
    -lcublas;
    -lcurand;
    -lcudart;
    CACHE INTERNAL ""
)

target_link_libraries (jetnet ${jetnet_LIBS})

# install library target and header files
install (TARGETS
    jetnet
    DESTINATION "lib"
)

install (FILES ${HEADERS}
    DESTINATION "include/jetnet"
)
