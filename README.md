# Jetnet
<img src="jetnet_logo.png" alt="Logo" width="250">

Super fast TensorRT implementation of YOLOv2

| Platform             | FP32 mode    | FP16 mode |
|:---------------------|:-------------|:----------|
| JETSON TX2           | 18.5 FPS     | 27 FPS    |
| GTX1080              | 220 FPS      | N.A.      |


NOTE: stats without pre/post processing, 416x416 input resolution

## Building the code

Dependencies:

* TensorRT 3.0.2 (also works with TensorRT 4.0.1.6)
* OpenCV 3.X

Building:

```
mkdir build
cd build
cmake CMAKE_BUILD_TYPE=Release ..
make
```

Building in debug mode:

```
cmake -DCMAKE_BUILD_TYPE=Debug ..
make
```

Building with address sanitizer (memory sanity checking)

```
cmake -DWITH_ADDRESS_SANITIZER=true ..
make
```

Installing on your system (path set with -DCMAKE_INSTALL_PREFIX):

```
make install
```

## Compiling a network

FP32 mode:

```
jetnet_build_yolov2 darknet_weights_file.weights out.plan
```

FP16 mode:

```
jetnet_run_yolov2 darknet_weights_file.weights out.plan --fp16
```

Building might take a while depending on the speed of your target

NOTE: if you did not install jetnet on your system, you will find the executables in the 'examples' folder inside
the build folder.

## Running a compiled network

```
jetnet_run_yolov2 out.plan names_file.names input_image.jpg
```

The current example will run the input image once while printing preprocessing, network execution and postprocessing times in seconds, showing the detection result in a window at the end.
