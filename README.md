# Test SSD network via TensorRT

`libuffssd.so` compiled for Jetson Nano (JetPack 4.4).
Code for `.so` [here](https://github.com/jumpercc/TensorRT/blob/release/7.1-jetson-nano-crosscompile-ssd/samples/opensource/sampleUffSSD/sampleUffSSD.cpp)

In this project we are in loop
- reading an image from MPI-CSI camera
- infer'ing the image via SSD Mobilenet (COCO trained)
- printing detected objects (class, confidence, coords (in [0, 1]) of left-top and right-bottom corners of an object)
