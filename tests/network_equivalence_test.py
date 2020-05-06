import pytest
import numpy as np
from darknet import darknet
import d2k
import struct
from PIL import Image
from pathlib import Path


test_data_path = Path('tests/data')
test_cache_path = Path('tests/cache')
darknet_files = Path('darknet-files')

# assume YOLOv4 requirements supported if this file exists
darknet_has_yolov4 = Path('darknet/cfg/yolov4.cfg').exists()


def setup_function():
    np.random.seed(0)   # to make tests reproducible


def rand_float32(shape, decimals=4):
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
                    self.write_float32(rand_float32(filters))       # beta / 'bias'
                    self.write_float32(.5 * rand_float32(filters))  # gamma / 'scale'
                    self.write_float32(rand_float32(filters))       # mean
                    self.write_float32(1.0 + .5 * rand_float32(filters))  # var
                else:
                    # Darknet saves (bias) (out_dim, in_dim, height, width)
                    self.write_float32(rand_float32(filters))  # bias

                self.write_float32(.2 * rand_float32((filters, in_dim, size, size)))

                in_dim = filters

            elif section == '[route]':
                layers = list(options['layers'])
                in_dim = sum([out_dim[i] for i in layers])

            else:
                # none of these have weights or change number of filters
                assert section in ['[shortcut]', '[upsample]', '[maxpool]', '[yolo]']

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


def compare_dn_to_keras(tmp_path, cfg_text, decimal=8):
    dn, k, network = make_networks(tmp_path, cfg_text)

    net_input = rand_float32(network.input_shape())

    print("net_input=", net_input, "shape:", net_input.shape)

    dn_output = dn.predict(net_input)
    dn_output = [dn_output] if not isinstance(dn_output, list) else dn_output

    k_output = k.predict(np.expand_dims(net_input, axis=0))
    k_output = [k_output] if not isinstance(k_output, list) else k_output

    k_output = [x.squeeze(axis=0) for x in k_output]

    print("k-dn:", [x - y for x, y in zip(k_output, dn_output)])

    assert len(dn_output) == len(k_output)
    for dn_out, k_out in zip(dn_output, k_output):
        np.testing.assert_almost_equal(k_out, dn_out, decimal=decimal)
         # NaN results aren't useful for assessing precision between DarkNet and our Keras net
        assert not np.isnan(k_out).any()


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
@pytest.mark.parametrize("activation", ['linear','leaky'] + (['mish'] if darknet_has_yolov4 else []))
@pytest.mark.parametrize("bn", [0,1])
def test_convolutional(tmp_path, size, stride, pad, activation, bn):
    cfg_text = '\n'.join([
        "[net]",
        "height=100",
        "width=150",
        "channels=3",
        "",
        "[convolutional]",
        f"batch_normalize={bn}",
        "filters=2",
        f"size={size}",
        f"stride={stride}",
        f"pad={pad}",
        f"activation={activation}",
    ])

    compare_dn_to_keras(tmp_path, cfg_text, decimal=6 if activation != 'mish' else 5)


# size=1 stride>1 not supported by darknet
@pytest.mark.parametrize("stride", [1, 2, 3])
def test_upsample(tmp_path, stride):
    cfg_text = '\n'.join([
        "[net]",
        "height=100",
        "width=150",
        "channels=3",
        "",
        "[upsample]",
        f"stride={stride}",
    ])

    compare_dn_to_keras(tmp_path, cfg_text)


@pytest.mark.parametrize("stride", [1, 2, 3, 5, 7])
@pytest.mark.parametrize("size", [2, 3, 4, 5, 7])
@pytest.mark.parametrize("padding", [None, 0, 2, 3])
def test_maxpool(tmp_path, stride, size, padding):
    cfg_text = '\n'.join([
        "[net]",
        "height=100",
        "width=150",
        "channels=3",
        "",
        "[maxpool]",
        f"stride={stride}",
        f"size={size}",
        f"{f'padding={padding}' if padding != None else ''}"
    ])

    compare_dn_to_keras(tmp_path, cfg_text)


def test_shortcut(tmp_path):
    cfg_text = '\n'.join([
        "[net]",
        "height=100",
        "width=150",
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

    compare_dn_to_keras(tmp_path, cfg_text, decimal=6)


def test_route_single_layers(tmp_path):
    cfg_text = '\n'.join([
        "[net]",
        "height=100",
        "width=150",
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

    compare_dn_to_keras(tmp_path, cfg_text, decimal=6)


def test_route_multiple_layers(tmp_path):
    cfg_text = '\n'.join([
        "[net]",
        "height=100",
        "width=150",
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

    compare_dn_to_keras(tmp_path, cfg_text, decimal=6)


def test_route_jumps_layers(tmp_path):
    cfg_text = '\n'.join([
        "[net]",
        "height=100",
        "width=200",
        "channels=3",
        "",
        "[convolutional]",
        "activation=linear",
        "",
        "[convolutional]",
        "activation=linear",
        "",
        "[route]",
        "layers=-2",
        "",
        "[route]",
        "layers=-2, -1"
    ])

    compare_dn_to_keras(tmp_path, cfg_text, decimal=6)


@pytest.mark.parametrize("size", [2, 10, 20])
@pytest.mark.parametrize("classes", [3, 20])
@pytest.mark.parametrize("mask", [range(0,3), range(2,7), range(0,9)])
@pytest.mark.parametrize("scale_x_y", [None] + ([.9, 1.05, 1.1, 1.2] if darknet_has_yolov4 else []))
def test_yolo(tmp_path, size, classes, mask, scale_x_y):
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
        f"{f'scale_x_y={scale_x_y}' if scale_x_y != None else ''}",
    ])

    dn, k, network = make_networks(tmp_path, cfg_text)

    net_input = rand_float32(network.input_shape())

    dn_output = dn.predict(net_input)
    dn_output = dn_output.reshape(height, width, len(mask), (4+1+classes))

    k_output = k.predict(np.expand_dims(net_input, axis=0)).squeeze(axis=0)

    np.testing.assert_almost_equal(k_output, dn_output, decimal=7 if scale_x_y == None else 6)
    assert not np.isnan(k_output).any()


@pytest.mark.parametrize("img_dim", [(200,300,3), (300,200,3), (300,300,3)])
def test_predict_image(tmp_path, img_dim):
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

    image = Image.fromarray(np.random.randint(0, 256, img_dim, dtype=np.uint8), 'RGB')
    image_file = tmp_path / "image.bmp"
    with open(image_file, "wb") as f:
        image.save(f, format='BMP')

    dn_image = darknet.image.load(image_file)

    k_image = d2k.image.load(image_file)
    k_image = d2k.image.letterbox(k_image, width, height)

    dn_output = dn.predict(dn_image)
    k_output = k.predict(np.expand_dims(k_image, axis=0)).squeeze(axis=0)

    np.testing.assert_almost_equal(k_output, dn_output, decimal=5) # XXX why the high error?
    assert not np.isnan(k_output).any()


def darknet_compute(cfg_file, image_file):
    # a little caching goes a long way when running this again and again...
    output_file    = test_cache_path / (image_file.stem + '-' + cfg_file + '-output.npz')
    boxes_file     = test_cache_path / (image_file.stem + '-' + cfg_file + '-boxes.npz')
    nms_boxes_file = test_cache_path / (image_file.stem + '-' + cfg_file + '-nms-boxes.npz')

    dn_image = darknet.image.load(image_file)

    try:
        output = np.load(output_file, allow_pickle=True)
        output = output[output.files[0]]

        boxes = np.load(boxes_file, allow_pickle=True)
        boxes = boxes[boxes.files[0]]

        nms_boxes = np.load(nms_boxes_file, allow_pickle=True)
        nms_boxes = nms_boxes[nms_boxes.files[0]]
    except IOError:
        dn = darknet(darknet_files / (cfg_file + '.cfg'),
                     darknet_files / (cfg_file + '.weights'))

        output = dn.predict(dn_image)
        boxes = dn.get_network_boxes(dn_image, do_nms=False)
        nms_boxes = dn.get_network_boxes(dn_image, do_nms=True)

        np.savez(output_file, output)
        np.savez(boxes_file, boxes)
        np.savez(nms_boxes_file, nms_boxes)

    return dn_image.to_array(), output, boxes, nms_boxes


@pytest.mark.parametrize("image_stem", ['zebra', 'dog', 'cats'])
@pytest.mark.parametrize("yolo", ['yolov3'] + (['yolov4'] if darknet_has_yolov4 else []))
def test_boxes_from_darknet_output(image_stem, yolo):
    image, dn_output, dn_boxes, _ = darknet_compute(yolo, test_data_path / (image_stem + '.png'))

    network = d2k.network.load((darknet_files / (yolo + '.cfg')).read_text())

    image_dim = (image.shape[1], image.shape[0])
    net_dim = (network.input_shape()[1], network.input_shape()[0])

    dn_output = d2k.network.post_just_activate(dn_output, network.config)
    k_boxes = d2k.network.boxes_from_output(dn_output, net_dim, image_dim)
    k_boxes = np.array([b.to_list() for b in k_boxes], dtype=np.float32)

    np.testing.assert_almost_equal(k_boxes, dn_boxes, decimal=4) # XXX why the high error?


@pytest.mark.parametrize("image_stem", ['zebra', 'dog', 'cats'])
def test_nms_from_darknet_boxes(image_stem):
    _, _, dn_boxes, dn_nms = darknet_compute('yolov3', test_data_path / (image_stem + '.png'))

    dn_boxes = d2k.box.boxes_from_array(dn_boxes)
    dn_nms = d2k.box.boxes_from_array(dn_nms)

    k_nms = d2k.box.nms_boxes(dn_boxes, iou_thresh=.5, remove_all_zeros=False)

    print('dn_nms:', [x.objectness for x in dn_nms])
    print('k_nms:', [x.objectness for x in k_nms])

    assert sorted(k_nms) == sorted(dn_nms)


@pytest.mark.parametrize("image_stem", ['zebra', 'dog', 'cats'])
@pytest.mark.parametrize("yolo", ['yolov3'] + (['yolov4'] if darknet_has_yolov4 else []))
def test_yolo_network(image_stem, yolo):
    image_file = test_data_path / (image_stem + '.png')
    _, dn_output, _, _ = darknet_compute(yolo, image_file)

    network = d2k.network.load((darknet_files / (yolo + '.cfg')).read_text())
    k = network.make_model((darknet_files / (yolo + '.weights')).read_bytes(), just_activate_yolo=True)

    net_h, net_w, _ = network.input_shape()

    image = d2k.image.load(image_file)
    image = d2k.image.letterbox(image, net_w, net_h)

    k_output = k.predict(np.expand_dims(image, axis=0))
    k_output = [x.squeeze(axis=0) for x in k_output]

    assert len(dn_output) == len(k_output)
    for dn_out, k_out in zip(dn_output, k_output):
        print(k_out.shape)
        dn_out = dn_out.reshape(k_out.shape)
        np.testing.assert_almost_equal(k_out, dn_out, decimal=4) # XXX why the high error?
        assert not np.isnan(k_out).any()    # NaN results aren't useful for comparing


@pytest.mark.parametrize("image_stem", ['zebra', 'dog', 'cats'])
@pytest.mark.parametrize("yolo", ['yolov3'] + (['yolov4'] if darknet_has_yolov4 else []))
@pytest.mark.parametrize("use_detect_image", [False, True])
def test_end_to_end(image_stem, yolo, use_detect_image):
    image_file = test_data_path / (image_stem + '.png')
    _, _, _, dn_boxes = darknet_compute(yolo, image_file)

    network = d2k.network.load((darknet_files / (yolo + '.cfg')).read_text())
    k = network.make_model((darknet_files / (yolo + '.weights')).read_bytes())

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

    k_boxes = np.array([b.to_list() for b in k_boxes], dtype=np.float32)
    k_boxes = k_boxes[np.lexsort((k_boxes[:,3], k_boxes[:,2], k_boxes[:,1], k_boxes[:,0]))]

    dn_boxes = dn_boxes[np.sum(dn_boxes[...,5:], axis=1) > 0]
    dn_boxes = dn_boxes[np.lexsort((dn_boxes[:,3], dn_boxes[:,2], dn_boxes[:,1], dn_boxes[:,0]))]

    np.testing.assert_almost_equal(k_boxes, dn_boxes, decimal=3) # XXX why the high error?
