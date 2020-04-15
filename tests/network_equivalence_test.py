import pytest
import numpy as np
from darknet import darknet
import d2k
import struct
from PIL import Image
from pathlib import Path


test_data_path = Path('tests/data')
test_cache_path = Path('tests/cache')
yolov3_cfg = Path('darknet/yolov3.cfg')
yolov3_weights = Path('darknet/yolov3.weights')
coco_names = Path('darknet/coco.names')


def setup_function():
    np.random.seed(0)   # to make tests reproducible


def rand_float32(shape, decimals=5):
    """Shortcut to generating random numbers.  Round them a bit to try to keep
    float32 errors low."""
    shape = [shape] if not isinstance(shape, tuple) else shape
    return np.random.randn(*shape).astype(np.float32).round(decimals=decimals)


class DummyWeightsWriter:
    def __init__(self, config, version_0_2=True):
        self.data = []
        if version_0_2:
            self.write_int32(0)  # major
            self.write_int32(2)  # minor
            self.write_int32(0)  # revision
            self.write_int64(0)  # "seen"
        else:
            self.write_int32(0)  # major
            self.write_int32(1)  # minor
            self.write_int32(0)  # revision
            self.write_int32(0)  # "seen"

        in_dim = config[0][1]['channels']
        out_dim = dict()

        for (i, (section, options)) in enumerate(config[1:]):
            if section == '[convolutional]':
                filters = options['filters']
                size = options['size']
                batch_normalize = options['batch_normalize']

                if batch_normalize:
                    # BN: y = ((x - mean) / sqrt(var + epsilon)) * gamma + beta
                    # Darknet saves (beta)(gamma)(mean)(var) (out_dim, in_dim, height, width)
                    self.write_float32(rand_float32(filters))  # beta / 'bias'
                    self.write_float32(rand_float32(filters))  # gamma / 'scale'
                    self.write_float32(rand_float32(filters))  # mean
                    self.write_float32(rand_float32(filters))  # var
                else:
                    # Darknet saves (bias) (out_dim, in_dim, height, width)
                    self.write_float32(rand_float32(filters))  # bias

                self.write_float32(rand_float32((filters, in_dim, size, size)))

                in_dim = filters

            elif section == '[route]':
                layers = list(options['layers'])
                in_dim = sum([out_dim[i] for i in layers])

            else:
                # none of these have weights or change number of filters
                assert section in ['[shortcut]', '[upsample]', '[yolo]']

            out_dim[i] = in_dim


    def write_int32(self, value):
        self.data.append(struct.pack('i', value))

    def write_int64(self, value):
        self.data.append(struct.pack('l', value))

    def write_float32(self, value):
        if isinstance(value, float): return self.write_float32(list(value))
        if isinstance(value, np.ndarray): return self.write_float32(list(value.flatten()))
        self.data.append(struct.pack(f'{len(value)}f', *value))

    def get_weights(self):
        return bytes().join(self.data)


def make_networks(tmp_path, cfg_text):
    cfg_file = tmp_path / "testnet.cfg"
    weights_file = tmp_path / "testnet.weights"

    network = d2k.network.load(cfg_text)

    ww = DummyWeightsWriter(network.config)

    k = network.make_model(ww.get_weights(), just_activate_yolo=True)

    cfg_file.write_text(cfg_text)
    weights_file.write_bytes(ww.get_weights())
    dn = darknet(cfg_file, weights_file)

    return (dn, k, network)


def compare_dn_to_keras(tmp_path, cfg_text):
    dn, k, network = make_networks(tmp_path, cfg_text)

    net_input = rand_float32(network.input_shape())

    print("net_input=", net_input, "shape:", net_input.shape)

    dn_output = dn.predict(net_input)
    dn_output = [dn_output] if not isinstance(dn_output, list) else dn_output
    print("dn_output=", dn_output, "shape:", [x.shape for x in list(dn_output)])

    k_output = k.predict(np.expand_dims(net_input, axis=0))
    k_output = [k_output] if not isinstance(k_output, list) else k_output

    k_output = [x.squeeze(axis=0) for x in k_output]
    print("k_output=", k_output, "shape:", [x.shape for x in list(k_output)])

    # XXX decimal=6 or 7 would be better...
    for dn_out, k_out in zip(dn_output, k_output):
        np.testing.assert_almost_equal(k_out, dn_out, decimal=5)


def test_read_weights_not_0_2(tmp_path):
    cfg_text = '\n'.join([
        "[net]",
        "height=3",
        "width=3",
        "channels=2",
        "",
        "[convolutional]",
        "size=1",
        "activation=linear",
    ])

    cfg_file = tmp_path / "testnet.cfg"
    weights_file = tmp_path / "testnet.weights"

    network = d2k.network.load(cfg_text)

    ww = DummyWeightsWriter(network.config, version_0_2=False)

    with pytest.raises(d2k.network.ConversionError):
        network.make_model(ww.get_weights())


# size=1 stride>1 not supported by darknet
@pytest.mark.parametrize("size, stride", [(1,1),(2,1),(2,2),(3,1),(3,2)])
@pytest.mark.parametrize("pad", [0,1])
@pytest.mark.parametrize("activation", ['linear','leaky'])
@pytest.mark.parametrize("bn", [0,1])
def test_convolutional(tmp_path, size, stride, pad, activation, bn):
    cfg_text = '\n'.join([
        "[net]",
        "height=3",
        "width=3",
        "channels=2",
        "",
        "[convolutional]",
        f"batch_normalize={bn}",
        "filters=2",
        f"size={size}",
        f"stride={stride}",
        f"pad={pad}",
        f"activation={activation}",
    ])

    compare_dn_to_keras(tmp_path, cfg_text)


# size=1 stride>1 not supported by darknet
@pytest.mark.parametrize("stride", [1, 2, 3])
def test_upsample(tmp_path, stride):
    cfg_text = '\n'.join([
        "[net]",
        "height=3",
        "width=3",
        "channels=2",
        "",
        "[upsample]",
        f"stride={stride}",
    ])

    compare_dn_to_keras(tmp_path, cfg_text)


def test_shortcut(tmp_path):
    cfg_text = '\n'.join([
        "[net]",
        "height=3",
        "width=3",
        "channels=3",
        "",
        "[convolutional]",
        "filters=1",
        "size=2",
        "stride=1",
        "pad=1",
        "activation=leaky",
        "",
        "[convolutional]",
        "filters=1",
        "size=3",
        "stride=1",
        "pad=1",
        "activation=leaky",
        "",
        "[shortcut]",
        "from=-2",
        "activation=linear"
    ])

    compare_dn_to_keras(tmp_path, cfg_text)


def test_route_single_layers(tmp_path):
    cfg_text = '\n'.join([
        "[net]",
        "height=3",
        "width=3",
        "channels=3",
        "",
        "[convolutional]",
        "filters=1",
        "pad=1",
        "activation=linear",
        "",
        "[convolutional]",
        "filters=1",
        "size=2",
        "stride=1",
        "pad=1",
        "activation=leaky",
        "",
        "[route]",
        "layers=-2",
        "",
        "[convolutional]",
        "filters=1",
        "size=3",
        "stride=1",
        "pad=1",
        "activation=leaky",
    ])

    compare_dn_to_keras(tmp_path, cfg_text)


def test_route_multiple_layers(tmp_path):
    cfg_text = '\n'.join([
        "[net]",
        "height=3",
        "width=3",
        "channels=3",
        "",
        "[convolutional]",
        "filters=1",
        "size=2",
        "stride=1",
        "pad=1",
        "activation=leaky",
        "",
        "[convolutional]",
        "filters=1",
        "size=3",
        "stride=1",
        "pad=1",
        "activation=leaky",
        "",
        "[route]",
        "layers=-1,-2",
    ])

    compare_dn_to_keras(tmp_path, cfg_text)


#@pytest.mark.skip()
def test_bigger_net(tmp_path):
    cfg_text = '\n'.join([
        "[net]",
        "width=608",
        "height=608",
        "channels=3",
        "",
        "[convolutional]",
        "batch_normalize=1",
        "filters=32",
        "size=3",
        "stride=1",
        "pad=1",
        "activation=leaky",
        "",
        "# Downsample",
        "",
        "[convolutional]",
        "batch_normalize=1",
        "filters=64",
        "size=3",
        "stride=2",
        "pad=1",
        "activation=leaky",
        "",
        "[convolutional]",
        "batch_normalize=1",
        "filters=32",
        "size=1",
        "stride=1",
        "pad=1",
        "activation=leaky",
        "",
        "[convolutional]",
        "batch_normalize=1",
        "filters=64",
        "size=3",
        "stride=1",
        "pad=1",
        "activation=leaky",
        "",
        "[shortcut]",
        "from=-3",
        "activation=linear",
        "",
        "# Downsample",
        "",
        "[convolutional]",
        "batch_normalize=1",
        "filters=128",
        "size=3",
        "stride=2",
        "pad=1",
        "activation=leaky",
        "",
        "[convolutional]",
        "batch_normalize=1",
        "filters=64",
        "size=1",
        "stride=1",
        "pad=1",
        "activation=leaky",
        "",
        "[convolutional]",
        "batch_normalize=1",
        "filters=128",
        "size=3",
        "stride=1",
        "pad=1",
        "activation=leaky",
        "",
        "[shortcut]",
        "from=-3",
        "activation=linear",
        "",
        "[convolutional]",
        "batch_normalize=1",
        "filters=64",
        "size=1",
        "stride=1",
        "pad=1",
        "activation=leaky",
        "",
        "[convolutional]",
        "batch_normalize=1",
        "filters=128",
        "size=3",
        "stride=1",
        "pad=1",
        "activation=leaky",
        "",
        "[shortcut]",
        "from=-3",
        "activation=linear",
        "",
        "# Downsample",
        "",
        "[convolutional]",
        "batch_normalize=1",
        "filters=256",
        "size=3",
        "stride=2",
        "pad=1",
        "activation=leaky",
        "",
        "[convolutional]",
        "batch_normalize=1",
        "filters=128",
        "size=1",
        "stride=1",
        "pad=1",
        "activation=leaky",
        "",
        "[convolutional]",
        "batch_normalize=1",
        "filters=256",
        "size=3",
        "stride=1",
        "pad=1",
        "activation=leaky",
        "",
        "[shortcut]",
        "from=-3",
        "activation=linear",
        "",
        "[convolutional]",
        "batch_normalize=1",
        "filters=128",
        "size=1",
        "stride=1",
        "pad=1",
        "activation=leaky",
        "",
        "[convolutional]",
        "batch_normalize=1",
        "filters=256",
        "size=3",
        "stride=1",
        "pad=1",
        "activation=leaky",
        "",
        "[shortcut]",
        "from=-3",
        "activation=linear",
        "",
        "[convolutional]",
        "batch_normalize=1",
        "filters=128",
        "size=1",
        "stride=1",
        "pad=1",
        "activation=leaky",
        "",
        "[convolutional]",
        "batch_normalize=1",
        "filters=256",
        "size=3",
        "stride=1",
        "pad=1",
        "activation=leaky",
        "",
        "[shortcut]",
        "from=-3",
        "activation=linear",
    ])

    dn, k, network = make_networks(tmp_path, cfg_text)

    net_input = rand_float32(network.input_shape())

    print("net_input=", net_input, "shape:", net_input.shape)

    dn_output = dn.predict(net_input)
    print("dn_output=", dn_output, "shape:", dn_output.shape)

    k_output = k.predict(np.expand_dims(net_input, axis=0)).squeeze(axis=0)
    print("k_output=", k_output, "shape:", k_output.shape)

    np.testing.assert_equal(k_output, dn_output)


@pytest.mark.parametrize("size", [2, 10, 20])
@pytest.mark.parametrize("classes", [3, 20])
@pytest.mark.parametrize("mask", [range(0,3), range(2,7), range(0,9)])
def test_yolo(tmp_path, size, classes, mask):
    height = size 
    width = size
    cfg_text = '\n'.join([
        "[net]",
        f"height={height}",
        f"width={width}",
        f"channels={(4+1+classes)*len(mask)}",
        "",
        "[yolo]",
        f"classes={classes}",
        "num=9",
        f"mask={','.join([str(x) for x in mask])}",
        "anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326",
    ])

    dn, k, network = make_networks(tmp_path, cfg_text)

    net_input = rand_float32(network.input_shape())

    print("net_input=", net_input, "shape:", net_input.shape)

    dn_output = dn.predict(net_input)
    dn_output = dn_output.reshape(height, width, len(mask), (4+1+classes))
    print("dn_output=", dn_output, "shape:", dn_output.shape)

    k_output = k.predict(np.expand_dims(net_input, axis=0)).squeeze(axis=0)
    print("k_output=", k_output, "shape:", k_output.shape)

    np.testing.assert_almost_equal(k_output, dn_output, decimal=5)


@pytest.mark.parametrize("use_dn_image", [True, False])
def test_predict_image(tmp_path, use_dn_image):
    cfg_text = '\n'.join([
        "[net]",
        "height=100",
        "width=100",
        "channels=3",
        "",
        "[convolutional]",
        "batch_normalize=0",
        "filters=2",
        "size=5",
        "stride=2",
        "pad=0",
        "activation=linear",
    ])

    dn, k, network = make_networks(tmp_path, cfg_text)

    height, width, _ = network.input_shape()

    image = Image.fromarray(np.random.randint(0, 255, (300, 300, 3)), 'RGB')
    image_file = tmp_path / "image.bmp"
    with open(image_file, "wb") as f:
        image.save(f, format='BMP')

    dn_image = darknet.image.load(image_file)

    if use_dn_image:
        k_image = dn_image.letterbox(width, height).to_array()
    else:
        k_image = d2k.image.load(image_file)
        k_image = d2k.image.letterbox(k_image, width, height)

    dn_output = dn.predict(dn_image)
    k_output = k.predict(np.expand_dims(k_image, axis=0)).squeeze(axis=0)

    # d2k.image.resize introduces differences, hence the lower bar
    np.testing.assert_almost_equal(k_output, dn_output, decimal=(6 if use_dn_image else 4))


def darknet_compute(cfg_file, image_file):
    # a little caching goes a long way when running this again and again...
    output_file    = test_cache_path / (image_file.stem + '-output.npz')
    boxes_file     = test_cache_path / (image_file.stem + '-boxes.npz')
    nms_boxes_file = test_cache_path / (image_file.stem + '-nms-boxes.npz')

    dn_image = darknet.image.load(image_file)

    try:
        output = np.load(output_file, allow_pickle=True)
        output = output[output.files[0]]

        boxes = np.load(boxes_file, allow_pickle=True)
        boxes = boxes[boxes.files[0]]

        nms_boxes = np.load(nms_boxes_file, allow_pickle=True)
        nms_boxes = nms_boxes[nms_boxes.files[0]]
    except IOError:
        dn = darknet(cfg_file, cfg_file.parent / (cfg_file.stem + '.weights'))

        output = dn.predict(dn_image)
        boxes = dn.get_network_boxes(dn_image, do_nms=False)
        nms_boxes = dn.get_network_boxes(dn_image, do_nms=True)

        np.savez(output_file, output)
        np.savez(boxes_file, boxes)
        np.savez(nms_boxes_file, nms_boxes)

    return dn_image.to_array(), output, boxes, nms_boxes


def round_boxes(boxes):
    # XXX is all this rounding hiding errors?
    return np.concatenate([boxes[:,:4].round(decimals=2),   # x, y, w, h
                           boxes[:,4:].round(decimals=5)],  # objectness, [classes]
                          axis=1)


def test_round_boxes():
    a = np.array([[.123456, .234567, .345678, .456789, .567890, .678901, .789012],
                  [.246802, .135791, .468024, .357913, .680246, .579135, .802468]])

    np.testing.assert_equal(round_boxes(a),
                            np.array([[.12, .23, .35, .46, .56789, .67890, .78901],
                                      [.25, .14, .47, .36, .68025, .57913, .80247]]))


@pytest.mark.parametrize("image_stem", ['zebra', 'dog'])
def test_boxes_from_darknet_output(image_stem):
    image, dn_output, dn_boxes, _ = darknet_compute(yolov3_cfg, test_data_path / (image_stem + '.png'))

    image_dim = (image.shape[1], image.shape[0])

    network = d2k.network.load(yolov3_cfg.read_text())

    k_boxes = d2k.network.get_network_boxes_old(dn_output, network.config, image_dim)
    print('k_boxes:', k_boxes)

    np.testing.assert_equal(round_boxes(k_boxes), round_boxes(dn_boxes))


@pytest.mark.parametrize("image_stem", ['zebra', 'dog'])
def test_nms_from_darknet_boxes(image_stem):
    _, _, dn_boxes, dn_nms = darknet_compute(yolov3_cfg, test_data_path / (image_stem + '.png'))


    dn_boxes = d2k.box.boxes_from_array(dn_boxes)
    dn_nms = d2k.box.boxes_from_array(dn_nms)

    k_nms = d2k.box.nms_boxes(dn_boxes, iou_thresh=.5, remove_all_zeros=False)

    print('dn_nms:', [x.objectness for x in dn_nms])
    print('k_nms:', [x.objectness for x in k_nms])

    assert sorted(k_nms) == sorted(dn_nms)


# corners obtained from './darknet detect cfg/yolov3.cfg yolov3.weights filename'
# (after adding code to print them)
@pytest.mark.parametrize("image, dn_result", [['zebra', [(['zebra'],(335,104,557,268)),
                                                         (['zebra'],(169,92,394,252)),
                                                         (['zebra'],(48,90,234,267))]],
                                              ['dog',   [(['bicycle'],(99,122,590,447)),
                                                         (['truck'], (476,81,684,168)),
                                                         (['dog'], (134,213,313,543))]]
                         ])
def test_yolov3_old(image, dn_result):
    image_file = test_data_path / (image + '.png')

    network = d2k.network.load(yolov3_cfg.read_text())
    k = network.make_model(yolov3_weights.read_bytes(), just_activate_yolo=True)
    names = coco_names.read_text().splitlines()

    net_h, net_w, _ = network.input_shape()

    image = d2k.image.load(image_file)
    image_dim = (image.shape[1], image.shape[0])
    image = d2k.image.letterbox(image, net_w, net_h)

    k_output = k.predict(np.expand_dims(image, axis=0))
    k_output = [x.squeeze(axis=0) for x in k_output]

    k_boxes = d2k.network.get_network_boxes_old(k_output, network.config, image_dim)
    k_boxes = d2k.box.nms_boxes(d2k.box.boxes_from_array(k_boxes))

    k_result = [([names[i]], b.corners()) for b in k_boxes for i in range(len(names)) if b.classes[i] > 0]

    assert sorted(dn_result) == sorted(k_result)


# corners obtained from './darknet detect cfg/yolov3.cfg yolov3.weights filename'
# (after adding code to print them)
@pytest.mark.parametrize("image, dn_result", [['zebra', [(['zebra'],(335,104,557,268)),
                                                         (['zebra'],(169,92,394,252)),
                                                         (['zebra'],(48,90,234,267))]],
                                              ['dog',   [(['bicycle'],(99,122,590,447)),
                                                         (['truck'], (476,81,684,168)),
                                                         (['dog'], (134,213,313,543))]],
                                              # other than 'zebra' and 'dog', 'cats' is higher
                                              # than it is wide -- needed for letterbox / box correction coverage
                                              ['cats',  [(['cat'], (265,35,474,366)),
                                                         (['cat'], (127,454,376,570))]]
                         ])
@pytest.mark.parametrize("use_detect_image", [False, True])
def test_yolov3(image, dn_result, use_detect_image):
    image_file = test_data_path / (image + '.png')

    network = d2k.network.load(yolov3_cfg.read_text())
    k = network.make_model(yolov3_weights.read_bytes())
    names = coco_names.read_text().splitlines()

    image = d2k.image.load(image_file)

    if use_detect_image:
        k_boxes = d2k.network.detect_image(k, image)
    else:
        net_h, net_w, _ = network.input_shape()

        image_dim = (image.shape[1], image.shape[0])
        image = d2k.image.letterbox(image, net_w, net_h)

        k_output = k.predict(np.expand_dims(image, axis=0))
        k_output = [x.squeeze(axis=0) for x in k_output]

        k_boxes = d2k.network.boxes_from_output(k_output, (net_w, net_h), image_dim)
        k_boxes = d2k.box.nms_boxes(k_boxes)

    k_result = [([names[i]], b.corners()) for b in k_boxes for i in range(len(names)) if b.classes[i] > 0]

    assert sorted(dn_result) == sorted(k_result)