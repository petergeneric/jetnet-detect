#ifndef JETNET_CAMERAUTILS_H
#define JETNET_CAMERAUTILS_H

struct CameraDetectionState {
    int vehicles = 0;
    int people = 0;
};

struct CameraDefinition {
    const char *name;
    const char *snapshot_url;

    float detect_threshold = 0.3f;

    // If true, re-checks if there are more people/vehicles detected in this frame vs last time
    bool retest_if_new_objects_found = true;

    // If true, re-checks if there are fewer people/vehicles detected in this frame vs last time
    bool retest_if_objects_disappear = false;

    // The minimum area required to be classified as a person
    float min_person_area = 0;

    // The minimum area required to be classified as a vehicle
    float min_vehicle_area = 0;

    // Ignore any objects whose lowest point is above the named start line; set to 0 to disable
    float ignore_all_above_line = 0;

    CameraDetectionState state = {};
};


#endif //JETNET_CAMERAUTILS_H
