# d2k: YOLOv4/v3 in Keras / Tensorflow 2.1, test-first

[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)

## Welcome to d2k

D2K implements the [YOLOv4](https://arxiv.org/abs/2004.10934) and
[YOLOv3](https://pjreddie.com/media/files/papers/YOLOv3.pdf) object detection algorithms
 in Keras/TensorFlow 2.1.0.
Most everything was implemented test-first and the results match
[the original Darknet](https://github.com/pjreddie/darknet)
and [YOLOv4's Darknet](https://github.com/AlexeyAB/darknet)
(allowing for floating point error fun).

D2K is inference-only so far...  (Re-)training is where things really get fun, though,
so I'll be looking into adding it as time allows.

![Sample YOLOv3 detections for COCO classes](etc/dog-detection.png)

## Quick Start

The YOLOv3/YOLOv4 weights files are too big for checking into GitHub directly, but
if you have [Git LFS](https://git-lfs.github.com/) set up, cloning the repository
should get you copies.  Otherwise, after cloning you'll need to get it from the
Darknet site:
```bash
wget https://pjreddie.com/media/files/yolov3.weights
mv yolov3.weights darknet-files/
```
See [the YOLOv4 GitHub page](https://github.com/AlexeyAB/darknet) for where to
download the yolov4 weights.

You shouldn't have to install anything.  On an Python 3, Tensorflow 2 environment try:
```bash
python yolo.py tests/data/dog.png
```

In Python, using it is as simple as, for example,
```python
net = d2k.network.load(Path('darknet-files/yolov3.cfg').read_text())
net.read_darknet_weights(Path('darknet-files/yolov3.weights').read_bytes())
model = net.make_model()
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
There is an equivalent [branch for YOLOv4](https://github.com/jaltmayerpizzorno/darknet/tree/alexeyab-master).
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

### To see the network used
To see the Keras/Tensorflow NN used, you can pass `--print` to `yolo.py`:
```bash
python yolo.py --version 4 --print

layer_in = keras.Input(shape=(608, 608, 3))
layer_0 = keras.layers.ZeroPadding2D(((1,1),(1,1)))(layer_in)
layer_0 = keras.layers.Conv2D(32, 3, strides=1, use_bias=False, name='conv_0')(layer_0)
layer_0 = keras.layers.BatchNormalization(epsilon=.00001, name='bn_0')(layer_0)
layer_0 = layer_0 * K.tanh(K.softplus(layer_0))
layer_1 = keras.layers.ZeroPadding2D(((1,1),(1,1)))(layer_0)
layer_1 = keras.layers.Conv2D(64, 3, strides=2, use_bias=False, name='conv_1')(layer_1)
...

```


## More Information

The `d2k.network.Network` class reads a Darknet configuration file and generates an equivalent
Keras model;  its `convert()` outputs a list of Python statements building the model, making it
easy to check (and incorporate elsewhere if desired).  It can also read Darknet YOLOv3/YOLOv4
weights into the resulting model, for use and/or for serializing for later use.

The files under `darknet-files` are all originally from the Darknet authors,
included here for convenience.

## TODOs

- add training support
- move non-max suppression into the neural network
- look to support additional layers

## Other Darknet and YOLO

There are YOLOs aplenty.  Here are some I find particularly noteworthy:

- [Darknet YOLO project](https://pjreddie.com/darknet/yolo/)
- [Darknet on GitHub](https://github.com/pjreddie/darknet)
- [YOLOv4 on GitHub](https://github.com/AlexeyAB/darknet)
- [Allan Zellener's YAD2K for Yolo9000](https://github.com/allanzelener/YAD2K)
- [Ultralytics' YOLOv3 in PyTorch](https://github.com/ultralytics/yolov3)
- [Huynh Ngoc Anh's YOLOv3 in Keras](https://github.com/experiencor/keras-yolo3)
- [Anton Muehlemann's (very cool) YOLOv3](http://github.com/antonmu/TrainYourOwnYOLO)
