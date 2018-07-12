# Jetnet

TensorRT implementation of YOLOv2.

## Building the code

Dependencies:

* TensorRT 4.0.1.6 (needs cuDNN 7.1.x and CUDA 8.0 or CUDA 9.0)
* OpenCV 3.X

Building:

```
make
```

## Compiling a network

```
./buildYolov2 darknet_weights_file.weights out.plan
```

This might take a couple of seconds to build

## Running a compiled network

```
./runYolov2 out.plan names_file.names input_image.jpg
```

The current example will run the input image 10 times while printing preprocessing, network execution and postprocessing times in seconds, showing the detection
result in a window at the end.