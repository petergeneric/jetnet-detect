# Jetnet
TensorRT implementation of YOLOv2.

## Building the code

Dependencies:

* TensorRT 3.0.2 (also works with TensorRT 4.0.1.6)
* OpenCV 3.X

Building:

```
make
```

Building in debug mode:

```
make DEBUG=1
```

Building with address sanitizer (memory sanity checking)

```
make DEBUG=1 ASAN=1
```

## Compiling a network

FP32 mode:

```
./buildYolov2 darknet_weights_file.weights out.plan
```

FP16 mode:

```
./buildYolov2 darknet_weights_file.weights out.plan --fp16
```

Building might take a while depending on the speed of your target

## Running a compiled network

```
./runYolov2 out.plan names_file.names input_image.jpg
```

The current example will run the input image 10 times while printing preprocessing, network execution and postprocessing times in seconds, showing the detection
result in a window at the end.
