#ifndef JETNET_CAMERAUTILS_H
#define JETNET_CAMERAUTILS_H

struct CameraDetectionState {
    int vehicles = 0;
    int people = 0;
};

struct CameraDefinition {
    bool has_previous_checks = false;
    const char *name;
    const char *snapshot_url;

    float detect_threshold = 0.3f;

    // If true, re-checks if there are more people/vehicles detected in this frame vs last time
    bool retest_if_new_objects_found = true;

    // If true, re-checks if there are fewer people/vehicles detected in this frame vs last time
    bool retest_if_objects_disappear = false;

    // The minimum area required to be classified as a person
    float min_person_area = 0;

    // The maximum height possible when being classified as a person; set to non-zero to enable
    float max_person_height = 0;

    // The minimum area required to be classified as a vehicle
    float min_vehicle_area = 0;

    // The maximum area possible when being classified as a vehicle; set to non-zero to enable
    float max_vehicle_area = 0;

    // Ignore any objects whose lowest point is above the named start line; set to 0 to disable
    float ignore_all_above_line = 0;

    // Optionally apply special height limits to objects based on their column; this is for cameras where the RHS is far away and the LHS is close
    float special_rule_limit_column = 0;

    // If a person is wholly to the right of special_rule_limit_column, the maximum height they can be
    // Designed for cases where the RHS is far away from the camera, to avoid false detections of people who are too tall to be people on the right, but could be people on the LHS
    float special_rule_limit_max_person_height = 0;

    bool ignore_all_above_point_line = false;
    cv::Point ignore_line_left;
    cv::Point ignore_line_right;

    CameraDetectionState state = {};
};


#endif //JETNET_CAMERAUTILS_H
