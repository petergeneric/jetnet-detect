set (HEADERS
    jetnet.h
    engine/model_builder.h
    models/darknet_weights_loader.h
    models/yolov2_builder.h
    models/layers/leaky_relu_native.h
    models/layers/leaky_relu_plugin.h
    models/layers/conv2d_batch_leaky.h
    utils/logger.h
    utils/file_io.h
    utils/fp16.h
)

set (SOURCES
    engine/model_builder.cpp
    models/darknet_weights_loader.cpp
    models/yolov2_builder.cpp
    models/layers/leaky_relu_native.cpp
    models/layers/leaky_relu_plugin.cpp
    models/layers/conv2d_batch_leaky.cpp
    utils/logger.cpp
    utils/file_io.cpp
    utils/fp16.cpp
)

set (SRC ${SOURCES} ${HEADERS})

include_directories (
    /usr/local/cuda/include
    ${CMAKE_SOURCE_DIR}/src/engine
    ${CMAKE_SOURCE_DIR}/src/models
    ${CMAKE_SOURCE_DIR}/src/models/layers
    ${CMAKE_SOURCE_DIR}/src/utils
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
