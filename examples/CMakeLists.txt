include_directories (${jetnet_INCLUDES})

link_libraries (
    jetnet_static
    ${jetnet_LIBS}
)

add_executable (jetnet_build_yolov2 build_yolov2.cpp)
add_executable (jetnet_run_yolov2 run_yolov2.cpp)

install (TARGETS
    jetnet_build_yolov2
    jetnet_run_yolov2
    DESTINATION "bin"
)
