import pytest
from d2k.network import Network, ConversionError

yolov3_cfg = 'darknet-files/yolov3.cfg'


def test_convert_no_layers():
    cfg = '\n'.join([
        "[net]",
        "height=100",
        "width=200",
        "channels=4"
    ])

    net = Network.load(cfg).convert()
    assert net == [
        "layer_in = keras.Input(shape=(100, 200, 4))"
    ]


def test_convert_convolutional_defaults():
    cfg = '\n'.join([
        "[net]",
        "height=100",
        "width=200",
        "channels=3",
        "",
        "[convolutional]",
        "activation=linear",
    ])

    net = Network.load(cfg).convert()
    assert net == [
        "layer_in = keras.Input(shape=(100, 200, 3))",
        "layer_0 = keras.layers.Conv2D(1, 1, strides=1, use_bias=True, name='conv_0')(layer_in)",
        "layer_out = layer_0"
    ]


def test_convert_convolutional():
    cfg = '\n'.join([
        "[net]",
        "height=100",
        "width=200",
        "channels=3",
        "",
        "[convolutional]",
        "filters=32",
        "size=3",
        "stride=2",
        "pad=1",
        "activation=linear",
    ])

    net = Network.load(cfg).convert()
    assert net == [
        "layer_in = keras.Input(shape=(100, 200, 3))",
        "layer_0 = keras.layers.ZeroPadding2D(((1,1),(1,1)))(layer_in)",
        "layer_0 = keras.layers.Conv2D(32, 3, strides=2, use_bias=True, name='conv_0')(layer_0)",
        "layer_out = layer_0"
    ]


def test_convert_convolutional_activation_default():
    cfg = '\n'.join([
        "[net]",
        "height=100",
        "width=200",
        "channels=3",
        "",
        "[convolutional]",
    ])

    # default activation is 'logistic' and unsupported
    with pytest.raises(ConversionError):
        Network.load(cfg).convert()


def test_convert_convolutional_activation():
    cfg = '\n'.join([
        "[net]",
        "height=100",
        "width=200",
        "channels=3",
        "",
        "[convolutional]",
        "filters=32",
        "size=3",
        "stride=2",
        "pad=1",
        "activation=leaky",
    ])

    net = Network.load(cfg).convert()
    assert net == [
        "layer_in = keras.Input(shape=(100, 200, 3))",
        "layer_0 = keras.layers.ZeroPadding2D(((1,1),(1,1)))(layer_in)",
        "layer_0 = keras.layers.Conv2D(32, 3, strides=2, use_bias=True, name='conv_0')(layer_0)",
        "layer_0 = keras.layers.LeakyReLU(alpha=.1)(layer_0)",
        "layer_out = layer_0"
    ]


def test_convert_convolutional_batch_normalize():
    cfg = '\n'.join([
        "[net]",
        "height=100",
        "width=200",
        "channels=3",
        "",
        "[convolutional]",
        "batch_normalize=1",
        "filters=32",
        "size=3",
        "stride=2",
        "pad=1",
        "activation=leaky",
    ])

    net = Network.load(cfg).convert()
    assert net == [
        "layer_in = keras.Input(shape=(100, 200, 3))",
        "layer_0 = keras.layers.ZeroPadding2D(((1,1),(1,1)))(layer_in)",
        "layer_0 = keras.layers.Conv2D(32, 3, strides=2, use_bias=False, name='conv_0')(layer_0)",
        "layer_0 = keras.layers.BatchNormalization(epsilon=.00001, name='bn_0')(layer_0)",
        "layer_0 = keras.layers.LeakyReLU(alpha=.1)(layer_0)",
        "layer_out = layer_0"
    ]


def test_convert_convolutional_unsupported_option():
    cfg = '\n'.join([
        "[net]",
        "height=100",
        "width=200",
        "channels=3",
        "",
        "[convolutional]",
        "foo=bar",
    ])

    with pytest.raises(ConversionError):
        Network.load(cfg).convert()


def test_convert_route_layers_negative():
    cfg = '\n'.join([
        "[net]",
        "height=100",
        "width=200",
        "channels=3",
        "",
        "[convolutional]",
        "activation=linear",
        "",
        "[route]",
        "layers=-1"
    ])

    net = Network.load(cfg).convert()
    assert net == [
        "layer_in = keras.Input(shape=(100, 200, 3))",
        "layer_0 = keras.layers.Conv2D(1, 1, strides=1, use_bias=True, name='conv_0')(layer_in)",
        "layer_1 = layer_0",
        "layer_out = layer_1"
    ]


def test_convert_route_layers_positive():
    cfg = '\n'.join([
        "[net]",
        "height=100",
        "width=200",
        "channels=3",
        "",
        "[convolutional]",
        "activation=linear",
        "",
        "[route]",
        "layers=0"
    ])

    net = Network.load(cfg).convert()
    assert net == [
        "layer_in = keras.Input(shape=(100, 200, 3))",
        "layer_0 = keras.layers.Conv2D(1, 1, strides=1, use_bias=True, name='conv_0')(layer_in)",
        "layer_1 = layer_0",
        "layer_out = layer_1"
    ]


def test_convert_route_layers_multiple():
    cfg = '\n'.join([
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
        "[convolutional]",
        "activation=linear",
        "",
        "[route]",
        "layers=-1, -3"
    ])

    net = Network.load(cfg).convert()
    assert net == [
        "layer_in = keras.Input(shape=(100, 200, 3))",
        "layer_0 = keras.layers.Conv2D(1, 1, strides=1, use_bias=True, name='conv_0')(layer_in)",
        "layer_1 = keras.layers.Conv2D(1, 1, strides=1, use_bias=True, name='conv_1')(layer_0)",
        "layer_2 = keras.layers.Conv2D(1, 1, strides=1, use_bias=True, name='conv_2')(layer_1)",
        "layer_3 = keras.layers.Concatenate()([layer_2, layer_0])",
        "layer_out = layer_3"
    ]


def test_convert_route_unsupported_option():
    cfg = '\n'.join([
        "[net]",
        "height=100",
        "width=200",
        "channels=3",
        "",
        "[convolutional]",
        "activation=linear",
        "",
        "[route]",
        "layers=-1",
        "foo=bar"
    ])

    with pytest.raises(ConversionError):
        Network.load(cfg).convert()


def test_convert_shortcut_from_negative():
    cfg = '\n'.join([
        "[net]",
        "height=100",
        "width=200",
        "channels=3",
        "",
        "[convolutional]",
        "filters=1",
        "activation=linear",
        "",
        "[convolutional]",
        "filters=2",
        "activation=leaky",
        "",
        "[convolutional]",
        "filters=3",
        "activation=linear",
        "",
        "[shortcut]",
        "from=-3"
    ])

    net = Network.load(cfg).convert()
    assert net == [
        "layer_in = keras.Input(shape=(100, 200, 3))",
        "layer_0 = keras.layers.Conv2D(1, 1, strides=1, use_bias=True, name='conv_0')(layer_in)",
        "layer_1 = keras.layers.Conv2D(2, 1, strides=1, use_bias=True, name='conv_1')(layer_0)",
        "layer_1 = keras.layers.LeakyReLU(alpha=.1)(layer_1)",
        "layer_2 = keras.layers.Conv2D(3, 1, strides=1, use_bias=True, name='conv_2')(layer_1)",
        "layer_3 = keras.layers.Add()([layer_0, layer_2])",
        "layer_out = layer_3"
    ]


def test_convert_shortcut_from_positive():
    cfg = '\n'.join([
        "[net]",
        "height=100",
        "width=200",
        "channels=3",
        "",
        "[convolutional]",
        "filters=1",
        "activation=linear",
        "",
        "[convolutional]",
        "filters=2",
        "activation=linear",
        "",
        "[shortcut]",
        "from=0"
    ])

    net = Network.load(cfg).convert()
    assert net == [
        "layer_in = keras.Input(shape=(100, 200, 3))",
        "layer_0 = keras.layers.Conv2D(1, 1, strides=1, use_bias=True, name='conv_0')(layer_in)",
        "layer_1 = keras.layers.Conv2D(2, 1, strides=1, use_bias=True, name='conv_1')(layer_0)",
        "layer_2 = keras.layers.Add()([layer_0, layer_1])",
        "layer_out = layer_2"
    ]


def test_convert_shortcut_activation():
    cfg = '\n'.join([
        "[net]",
        "height=100",
        "width=200",
        "channels=3",
        "",
        "[convolutional]",
        "filters=1",
        "activation=linear",
        "",
        "[convolutional]",
        "filters=2",
        "activation=linear",
        "",
        "[shortcut]",
        "from=-2",
        "activation=linear"
    ])

    net = Network.load(cfg).convert()
    assert net == [
        "layer_in = keras.Input(shape=(100, 200, 3))",
        "layer_0 = keras.layers.Conv2D(1, 1, strides=1, use_bias=True, name='conv_0')(layer_in)",
        "layer_1 = keras.layers.Conv2D(2, 1, strides=1, use_bias=True, name='conv_1')(layer_0)",
        "layer_2 = keras.layers.Add()([layer_0, layer_1])",
        "layer_out = layer_2"
    ]


def test_convert_shortcut_activation_nonlinear():
    cfg = '\n'.join([
        "[net]",
        "height=100",
        "width=200",
        "channels=3",
        "",
        "[convolutional]",
        "filters=1",
        "activation=linear",
        "",
        "[shortcut]",
        "from=-1",
        "activation=leaky"
    ])

    with pytest.raises(ConversionError):
        Network.load(cfg).convert()


def test_convert_upsample_default():
    cfg = '\n'.join([
        "[net]",
        "height=100",
        "width=200",
        "channels=3",
        "",
        "[upsample]",
    ])

    net = Network.load(cfg).convert()
    assert net == [
        "layer_in = keras.Input(shape=(100, 200, 3))",
        "layer_0 = keras.layers.UpSampling2D(2)(layer_in)",
        "layer_out = layer_0"
    ]


def test_convert_upsample_stride():
    cfg = '\n'.join([
        "[net]",
        "height=100",
        "width=200",
        "channels=3",
        "",
        "[upsample]",
        "stride=3"
    ])

    net = Network.load(cfg).convert()
    assert net == [
        "layer_in = keras.Input(shape=(100, 200, 3))",
        "layer_0 = keras.layers.UpSampling2D(3)(layer_in)",
        "layer_out = layer_0"
    ]


def test_convert_upsample_unsupported_option():
    cfg = '\n'.join([
        "[net]",
        "height=100",
        "width=200",
        "channels=3",
        "",
        "[upsample]",
        "foo=bar"
    ])

    with pytest.raises(ConversionError):
        Network.load(cfg).convert()


def test_convert_yolo_single():
    cfg = '\n'.join([
        "[net]",
        "height=100",
        "width=200",
        "channels=3",
        "",
        "[yolo]",
        "classes=10",
        "mask=0,1",
        "anchors=10,10, 20,20",
    ])

    net = Network.load(cfg).convert()
    assert net == [
        "layer_in = keras.Input(shape=(100, 200, 3))",
        "layer_0 = d2k.layers.Yolo(classes=10, anchors=[(10, 10), (20, 20)], net_dims=(100, 200))(layer_in)",
        "layer_out = layer_0",
    ]


def test_convert_yolo_multiple():
    cfg = '\n'.join([
        "[net]",
        "height=100",
        "width=200",
        "channels=3",
        "",
        "[convolutional]",
        "filters=1",
        "activation=linear",
        "",
        "[yolo]",
        "anchors=10,10, 20,20",
        "classes=10",
        "mask=0"
        "",
        "[route]",
        "layers=-2",
        "",
        "[yolo]",
        "anchors=10,10, 20,20",
        "classes=10",
        "mask=1"
    ])

    net = Network.load(cfg).convert()
    assert net == [
        "layer_in = keras.Input(shape=(100, 200, 3))",
        "layer_0 = keras.layers.Conv2D(1, 1, strides=1, use_bias=True, name='conv_0')(layer_in)",
        "layer_1 = d2k.layers.Yolo(classes=10, anchors=[(10, 10)], net_dims=(100, 200))(layer_0)",
        "layer_2 = layer_0",
        "layer_3 = d2k.layers.Yolo(classes=10, anchors=[(20, 20)], net_dims=(100, 200))(layer_2)",
        "layer_out = [layer_1, layer_3]",
    ]


def test_convert_just_activate_yolo():
    cfg = '\n'.join([
        "[net]",
        "height=100",
        "width=200",
        "channels=3",
        "",
        "[yolo]",
        "classes=10",
        "mask=0,1",
        "anchors=10,10, 20,20",
    ])

    net = Network.load(cfg).convert(just_activate_yolo=True)
    assert net == [
        "layer_in = keras.Input(shape=(100, 200, 3))",
        "layer_0 = d2k.layers.Yolo(classes=10, anchors=[(10, 10), (20, 20)], net_dims=(100, 200), just_activate=True)(layer_in)",
        "layer_out = layer_0",
    ]


def test_convert_yolo_unsupported_option():
    cfg = '\n'.join([
        "[net]",
        "height=100",
        "width=200",
        "channels=2",
        "",
        "[yolo]",
        "anchors=10,10",
        "foo=bar"
    ])

    with pytest.raises(ConversionError):
        Network.load(cfg).convert()


def test_convert_unsupported_section():
    cfg = '\n'.join([
        "[net]",
        "height=100",
        "width=200",
        "channels=2",
        "",
        "[foo]"
    ])

    with pytest.raises(ConversionError):
        Network.load(cfg).convert()


def test_convert_yolov3_doesnt_throw():
    with open(yolov3_cfg, 'r') as f:
        cfg = f.read()

    Network.load(cfg).convert()
