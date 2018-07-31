include_directories (${jetnet_INCLUDES})

link_libraries (
    jetnet_static
    ${jetnet_LIBS}
)

add_executable (jetnet_build_yolov2 build_yolov2.cpp)
add_executable (jetnet_build_yolov3 build_yolov3.cpp)
add_executable (jetnet_run_yolov2 run_yolov2.cpp)
add_executable (jetnet_run_yolov3 run_yolov3.cpp)
add_executable (jetnet_validate_yolov3 validate_yolov3.cpp)

install (TARGETS
    jetnet_build_yolov2
    jetnet_build_yolov3
    jetnet_run_yolov2
    jetnet_run_yolov3
    jetnet_validate_yolov3
    DESTINATION "bin"
)
