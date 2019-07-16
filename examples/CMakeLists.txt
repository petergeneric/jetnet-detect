include_directories (${TARGET_INCLUDE_DIRS})

link_libraries (
    jetnet_static
    ${CURL_LIBRARIES}
    ${TARGET_LIBS}
    $<$<AND:$<CXX_COMPILER_ID:GNU>,$<VERSION_LESS:$<CXX_COMPILER_VERSION>,9.0>>:stdc++fs>
)



add_executable (jetnet_build_darknet_model build_darknet_model.cpp)
add_executable (jetnet_run_yolo run_yolo.cpp)
add_executable (jetnet_run_yolo_folder yolo_folder.cpp)
add_executable (jetnet_validate_yolo validate_yolo.cpp)


install (TARGETS
    jetnet_build_darknet_model
    jetnet_run_yolo
    jetnet_run_yolo_folder
    jetnet_validate_yolo
    DESTINATION "bin"
)
