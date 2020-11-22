# Test SSD network via TensorRT

`libuffssd.so` compiled for Jetson Nano (JetPack 4.4).

In this project we are in loop
- reading an image from MPI-CSI camera
- infer'ing the image via SSD Mobilenet (COCO trained)
- printing detected objects (class, confidence, coords (in [0, 1]) of left-top and right-bottom corners of an object)
