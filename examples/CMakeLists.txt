include_directories (${jetnet_INCLUDES})

link_libraries (
    jetnet_static
    ${jetnet_LIBS}
)

add_executable (jetnet_build_yolov2 build_yolov2.cpp)
add_executable (jetnet_build_yolov3 build_yolov3.cpp)
add_executable (jetnet_run_yolo run_yolo.cpp fast_image_reader.cpp)
add_executable (jetnet_validate_yolo validate_yolo.cpp fast_image_reader.cpp)

install (TARGETS
    jetnet_build_yolov2
    jetnet_build_yolov3
    jetnet_run_yolo
    jetnet_validate_yolo
    DESTINATION "bin"
)
