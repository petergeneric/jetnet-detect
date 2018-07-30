set (HEADERS
    jetnet.h
    data/bgr8_letterbox_pre_processor.h
    data/detection.h
    data/fake_post_processor.h
    data/fake_pre_processor.h
    data/nms.h
    data/post_processor.h
    data/pre_processor.h
    data/yolo_post_processor.h
    engine/model_builder.h
    engine/model_runner.h
    engine/gpu_blob.h
    models/darknet_weights_loader.h
    models/yolov2_builder.h
    models/yolov3_builder.h
    models/layers/leaky_relu_native.h
    models/layers/leaky_relu_plugin.h
    models/layers/conv2d_batch_leaky.h
    models/plugins/blas_cuda.h
    models/plugins/upsample_plugin.h
    models/plugins/yolo_plugin_factory.h
    utils/file_io.h
    utils/fp16.h
    utils/logger.h
    utils/profiler.h
    utils/visual.h
)

set (SOURCES
    data/bgr8_letterbox_pre_processor.cpp
    data/fake_post_processor.cpp
    data/fake_pre_processor.cpp
    data/nms.cpp
    data/yolo_post_processor.cpp
    engine/model_builder.cpp
    engine/model_runner.cpp
    engine/gpu_blob.cpp
    models/darknet_weights_loader.cpp
    models/yolov2_builder.cpp
    models/yolov3_builder.cpp
    models/layers/leaky_relu_native.cpp
    models/layers/leaky_relu_plugin.cpp
    models/layers/conv2d_batch_leaky.cpp
    models/plugins/upsample_plugin.cpp
    models/plugins/blas_cuda.cu
    models/plugins/yolo_plugin_factory.cpp
    utils/file_io.cpp
    utils/fp16.cpp
    utils/logger.cpp
    utils/profiler.cpp
    utils/visual.cpp
)

set (SRC ${SOURCES} ${HEADERS})

set (jetnet_INCLUDES
    /usr/local/cuda/include;
    ${OpenCV_INCLUDE_DIRS};
    ${CMAKE_SOURCE_DIR}/src;
    ${CMAKE_SOURCE_DIR}/src/data;
    ${CMAKE_SOURCE_DIR}/src/engine;
    ${CMAKE_SOURCE_DIR}/src/models;
    ${CMAKE_SOURCE_DIR}/src/models/layers;
    ${CMAKE_SOURCE_DIR}/src/models/plugins;
    ${CMAKE_SOURCE_DIR}/src/utils;
    CACHE INTERNAL ""
)

set (jetnet_LIBS
    m;
    nvinfer_plugin;
    nvinfer;
    ${CUDA_LIBRARIES};
    ${CUDA_CUBLAS_LIBRARIES};
    ${OpenCV_LIBS};
    CACHE INTERNAL ""
)

include_directories (${jetnet_INCLUDES})

add_library (jetnet_static STATIC ${SRC})
add_library (jetnet SHARED ${SRC})

set_target_properties( jetnet_static PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties( jetnet PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries (jetnet ${jetnet_LIBS})

# install library target and header files
install (TARGETS
    jetnet
    DESTINATION "lib"
)

install (FILES ${HEADERS}
    DESTINATION "include/jetnet"
)
