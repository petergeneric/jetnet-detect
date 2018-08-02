# Jetnet
<img src="jetnet_logo.png" alt="Logo" width="250">

Super fast TensorRT implementation of YOLOv2 and YOLOv3

| YOLOv2               | FP32 mode    | FP16 mode |
|:---------------------|:-------------|:----------|
| JETSON TX2           | 18.5 FPS     | 27 FPS    |
| GTX1080              | 220 FPS      | N.A.      |

TODO: extend yolov2 stats with batch=1 and batch>1 and add the same stats for yolov3

NOTE: stats without pre/post processing, 416x416 input resolution

## Building the code

Dependencies:

* cmake 3.8+
* TensorRT 3.0.2 (also works with TensorRT 4.0.1.6)
* OpenCV 3.X

Building:

```
mkdir build
cd build
cmake ..
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
jetnet_build_yolovX darknet_weights_file.weights out.plan
```

FP16 mode:

```
jetnet_run_yolovX darknet_weights_file.weights out.plan --fp16
```

Building might take a while depending on the speed of your target

NOTE: if you did not install jetnet on your system, you will find the executables in the 'examples' folder inside
the build folder.

## Running a compiled network

```
jetnet_run_yolovX out.plan names_file.names input_image.jpg
```

Profiling:

```
jetnet_run_yolovX out.plan names_file.names input_image.jpg --profile
```

When profiling, the network will run the input image 10 times through the network, printing execution times
of pre-processing, post-processing, and network inference. Each network layer is also profiled and printed.

Change batch size:

```
jetnet_run_yolovX out.plan names_file.names input_image.jpg --batch=8
```
