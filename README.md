# d2k: YOLOv3 in Keras / Tensorflow 2.1, test-first

[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)

## Welcome to d2k

D2K reimplements the [YOLOv3](https://pjreddie.com/media/files/papers/YOLOv3.pdf) object detection algorithm
by Joseph Redmon and Ali Farhadi in Keras/TensorFlow 2.1.0.
Most everything was implemented test-first and matches [Darknet](https://github.com/pjreddie/darknet)
(allowing for floating point error fun).

D2K is inference-only so far...  I'll look into adding training as time allows.

![Sample YOLOv3 detections for COCO classes](etc/dog-detection.png)

## Quick Start

The YOLOv3 weights file is too big for checking into GitHub directly, but
if you have [Git LFS](https://git-lfs.github.com/) set up, cloning the repository
should get you a copy.  Otherwise, after cloning you'll need to get it from the
Darknet site:
```bash
pushd darknet
wget https://pjreddie.com/media/files/yolov3.weights
popd
```

You shouldn't have to install anything.  On an Python 3, Tensorflow 2 environment try:
```bash
python yolov3.py tests/data/dog.png
```

In Python, using it is as simple as, for example,
```python
network = d2k.network.load(Path('darknet-files/yolov3.cfg').read_text())
model = network.make_model(Path('darknet-files/yolov3.weights').read_bytes())
image = d2k.image.load(image_file)
boxes = d2k.network.detect_image(model, image)

im = Image.open(image_file)
d2k.box.draw_boxes(im, boxes)
im.show()
```

### To run the tests

For the tests, you'll need Darknet built on `../darknet`, as I embed it using
[ctypes](https://docs.python.org/3/library/ctypes.html).  I suggest you use
[my clone of Darknet](https://github.com/jaltmayerpizzorno/darknet) given a couple
of small bugs I fixed (found while coding this project) and also an adjustment
to gcc options.
```bash
pushd ..
git clone https://github.com/jaltmayerpizzorno/darknet.git
cd darknet
make
```

Then return to D2K and simply `make test`:
```bash
popd
make test
```

## More Information

The `d2k.network.Network` class reads a Darknet configuration file and generates an equivalent
Keras model;  its `convert()` outputs a list of Python statements building the model, making it
easy to check (and incorporate elsewhere if desired).  It can also read Darknet YOLOv3
weights into the resulting model, for use and/or for serializing for later use.

Some of the computation is done in a custom Keras layer.

The files under `darknet` are all originally from [Darknet](https://github.com/pjreddie/darknet),
included here for convenience.

## TODOs

- add training support
- look to support additional (non-YOLOv3) layers

## Other Darknet and YOLO

There are YOLOs aplenty.  Here are some I find particularly noteworthy:

- [Darknet YOLO project](https://pjreddie.com/darknet/yolo/)
- [Darknet on GitHub](https://github.com/pjreddie/darknet)
- [YOLOv4 on GitHub](https://github.com/AlexeyAB/darknet)
- [Allan Zellener's YAD2K for Yolo9000](https://github.com/allanzelener/YAD2K)
- [Ultralytics' YOLOv3 in PyTorch](https://github.com/ultralytics/yolov3)
- [Huynh Ngoc Anh's YOLOv3 in Keras](https://github.com/experiencor/keras-yolo3)
- [Anton Muehlemann's (very cool) YOLOv3](http://github.com/antonmu/TrainYourOwnYOLO)
