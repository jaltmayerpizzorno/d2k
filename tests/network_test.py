import pytest
import d2k.network
from d2k.network import ConversionError
from pathlib import Path

yolov3_cfg = Path('darknet-files/yolov3.cfg')
yolov4_cfg = Path('darknet-files/yolov4.cfg')


def test_parse():
    cfg = '\n'.join([
        "[net]",
        "height=100",
        "width=200",
        "channels=2",
        "saturation = 1.5",
        "scales= .1, .05",
        "",
        "[convolutional]",
        "activation=leaky",
    ])

    out = d2k.network.Network.parse(cfg)

    assert isinstance(out, list)
    assert len(out) == 2

    name, options = out[0]
    assert "[net]" == name
    assert isinstance(options, dict)
    assert {'height','width','channels','saturation','scales'} == options.keys()

    assert 100 == options['height']
    assert isinstance(options['height'], int)

    assert 200 == options['width']
    assert isinstance(options['width'], int)

    assert 2 == options['channels']
    assert isinstance(options['channels'], int)

    assert 1.5 == options['saturation']
    assert isinstance(options['saturation'], float)

    assert [.1, .05] == options['scales']

    name, options = out[1]
    assert "[convolutional]" == name
    assert 'leaky' == options['activation']


def test_parse_ignore_comment_lines():
    cfg = '\n'.join([
        "# comment",
        "[net]",
        "; another comment",
        "height=100",
        "width=200",
        "# and another",
        "channels=2",
    ])

    out = d2k.network.Network.parse(cfg)

    assert isinstance(out, list)
    assert len(out) == 1

    name, options = out[0]
    assert "[net]" == name
    assert isinstance(options, dict)
    assert {'height','width','channels'} == options.keys()

    assert 100 == options['height']
    assert isinstance(options['height'], int)

    assert 200 == options['width']
    assert isinstance(options['width'], int)

    assert 2 == options['channels']
    assert isinstance(options['channels'], int)


def test_parse_option_ouside_section():
    cfg = '\n'.join([
        "foo=bar",
    ])

    with pytest.raises(d2k.network.ConfigurationError):
        d2k.network.Network.parse(cfg)


def test_load_first_section_not_network():
    cfg = '\n'.join([
        "[convolutional]",
        "activation=leaky"
    ])

    with pytest.raises(d2k.network.ConfigurationError):
        d2k.network.load(cfg)


def test_load_net_missing_height():
    cfg = '\n'.join([
        "[net]",
        "width=200",
        "channels=2",
    ])

    with pytest.raises(d2k.network.ConfigurationError):
        d2k.network.load(cfg)

def test_load_net_missing_width():
    cfg = '\n'.join([
        "[net]",
        "height=100",
        "channels=2",
    ])

    with pytest.raises(d2k.network.ConfigurationError):
        d2k.network.load(cfg)


def test_load_net_missing_channels():
    cfg = '\n'.join([
        "[net]",
        "height=100",
        "width=200",
    ])

    with pytest.raises(d2k.network.ConfigurationError):
        d2k.network.load(cfg)


def test_load_yolo_missing_anchors():
    cfg = '\n'.join([
        "[net]",
        "height=100",
        "width=200",
        "channels=3",
        "",
        "[yolo]",
        "num=1",
    ])

    with pytest.raises(d2k.network.ConfigurationError):
        d2k.network.load(cfg)


@pytest.mark.parametrize("anchors", ["1", "1,2, 3"])
def test_load_yolo_uneven_anchors(anchors):
    cfg = '\n'.join([
        "[net]",
        "height=100",
        "width=200",
        "channels=3",
        "",
        "[yolo]",
        f"anchors={anchors}",
    ])

    with pytest.raises(d2k.network.ConfigurationError):
        d2k.network.load(cfg)


def cfg_to_string(network):
    s = []
    for (section, options) in network.config:
        s.append(section)
        for opt in sorted(options.keys()):
            s.append(f'{opt}={options[opt]}')

        s.append('')

    return '\n'.join(s)


def test_load_set_defaults():
    cfg = '\n'.join([
        "[net]",
        "height=100",
        "width=200",
        "channels=2",
        "",
        "[convolutional]",
        "",
        "[route]",
        "layers=0",
        "",
        "[shortcut]",
        "",
        "[upsample]",
        "",
        "[maxpool]",
        "",
    ])

    network = d2k.network.load(cfg)

    assert cfg_to_string(network) == '\n'.join([
            "[net]",
            "channels=2",
            "height=100",
            "width=200",
            "",
            "[convolutional]",
            "activation=logistic",
            "batch_normalize=0",
            "filters=1",
            "pad=0",
            "size=1",
            "stride=1",
            "",
            "[route]",
            "layers=[0]",
            "",
            "[shortcut]",
            "activation=linear",
            "",
            "[upsample]",
            "stride=2",
            "",
            "[maxpool]",
            "padding=0",
            "size=1",
            "stride=1",
            "",
    ])


def test_load_maxpool_set_defaults():
    cfg = '\n'.join([
        "[net]",
        "height=100",
        "width=200",
        "channels=2",
        "",
        "[maxpool]",
        "stride=3",
        "",
    ])

    network = d2k.network.load(cfg)

    assert cfg_to_string(network) == '\n'.join([
            "[net]",
            "channels=2",
            "height=100",
            "width=200",
            "",
            "[maxpool]",
            "padding=2",
            "size=3",
            "stride=3",
            "",
    ])


def test_load_maxpool_with_size_set_defaults():
    cfg = '\n'.join([
        "[net]",
        "height=100",
        "width=200",
        "channels=2",
        "",
        "[maxpool]",
        "stride=3",
        "size=2",
        "",
    ])

    network = d2k.network.load(cfg)

    assert cfg_to_string(network) == '\n'.join([
            "[net]",
            "channels=2",
            "height=100",
            "width=200",
            "",
            "[maxpool]",
            "padding=1",
            "size=2",
            "stride=3",
            "",
    ])


def test_load_yolo_anchors_and_defaults():
    cfg = '\n'.join([
        "[net]",
        "height=100",
        "width=200",
        "channels=2",
        "",
        "[yolo]",
        "mask=0,1",
        "anchors=10,10, 20,20",
    ])

    network = d2k.network.load(cfg)

    assert cfg_to_string(network) == '\n'.join([
            "[net]",
            "channels=2",
            "height=100",
            "width=200",
            "",
            "[yolo]",
            "anchors=[(10, 10), (20, 20)]",
            "classes=20",
            "mask=[0, 1]",
            "nms_kind=default",
            "num=1",    # XXX actually invalid (num must be >=2*len(anchors))
            "scale_x_y=1.0",
            ""
    ])


@pytest.mark.parametrize("mask", ["-1", "2"])
def test_load_yolo_mask_out_of_range(mask):
    cfg = '\n'.join([
        "[net]",
        "height=100",
        "width=200",
        "channels=2",
        "",
        "[yolo]",
        f"mask={mask}",
        "anchors=10,10, 20,20",
    ])

    with pytest.raises(d2k.network.ConfigurationError):
        d2k.network.load(cfg)


@pytest.mark.parametrize("num", [None, 4])
def test_set_defaults_yolo_no_mask(num):
    cfg = '\n'.join([
        "[net]",
        "height=100",
        "width=200",
        "channels=2",
        "",
        "[yolo]",
        "anchors=10,10, 20,20, 30,30, 40,40",
        f"{('num=' + str(num)) if num != None else ''}",
    ])

    network = d2k.network.load(cfg)

    num_value = num if num != None else 1

    assert cfg_to_string(network) == '\n'.join([
            "[net]",
            "channels=2",
            "height=100",
            "width=200",
            "",
            "[yolo]",
            "anchors=[(10, 10), (20, 20), (30, 30), (40, 40)]",
            "classes=20",
            f"mask={list(range(num_value))}",
            "nms_kind=default",
            f"num={num_value}",
            "scale_x_y=1.0",
            "",
    ])


@pytest.mark.parametrize("spec, value", [('1', '[1]'), ('1, 2', '[1, 2]')])
def test_set_defaults_yolo_mask(spec, value):
    cfg = '\n'.join([
        "[net]",
        "height=100",
        "width=200",
        "channels=2",
        "",
        "[yolo]",
        "anchors=10,10, 20,20, 30,30, 40,40",
        f"mask={spec}",
    ])

    network = d2k.network.load(cfg)

    assert cfg_to_string(network) == '\n'.join([
            "[net]",
            "channels=2",
            "height=100",
            "width=200",
            "",
            "[yolo]",
            "anchors=[(10, 10), (20, 20), (30, 30), (40, 40)]",
            "classes=20",
            f"mask={value}",
            "nms_kind=default",
            "num=1",
            "scale_x_y=1.0",
            "",
    ])


def test_set_defaults_doesnt_override():
    cfg = '\n'.join([
        "[net]",
        "height=100",
        "width=200",
        "channels=2",
        "",
        "[convolutional]",
        "activation=linear",
        "batch_normalize=1",
        "filters=2",
        "pad=1",
        "size=3",
        "stride=2",
        "",
        "[shortcut]",
        "activation=leaky",
        "",
        "[upsample]",
        "stride=1",
        "",
        "[yolo]",
        "anchors=10,10, 20,20, 30,30, 40,40",
        "classes=10",
        "mask=1,2",
        "nms_kind=greedynms",
        "num=3",
    ])

    config = d2k.network.load(cfg)

    assert cfg_to_string(config) == '\n'.join([
            "[net]",
            "channels=2",
            "height=100",
            "width=200",
            "",
            "[convolutional]",
            "activation=linear",
            "batch_normalize=1",
            "filters=2",
            "pad=1",
            "size=3",
            "stride=2",
            "",
            "[shortcut]",
            "activation=leaky",
            "",
            "[upsample]",
            "stride=1",
            "",
            "[yolo]",
            "anchors=[(10, 10), (20, 20), (30, 30), (40, 40)]",
            "classes=10",
            "mask=[1, 2]",
            "nms_kind=greedynms",
            "num=3",
            "scale_x_y=1.0",
            "",
    ])


@pytest.mark.parametrize("spec, result", [('-1', '1'), ('-2', '0'), ('-1, -2', '1, 0')])
def test_load_resolves_negative_route_layers(spec, result):
    cfg = '\n'.join([
        "[net]",
        "height=100",
        "width=200",
        "channels=2",
        "",
        "[convolutional]",
        "activation=linear",
        "batch_normalize=0",
        "filters=1",
        "pad=0",
        "size=1",
        "stride=1",
        "",
        "[convolutional]",
        "activation=linear",
        "batch_normalize=0",
        "filters=1",
        "pad=0",
        "size=1",
        "stride=1",
        "",
        "[route]",
        f"layers={spec}"
    ])

    config = d2k.network.load(cfg)

    assert cfg_to_string(config) == '\n'.join([
            "[net]",
            "channels=2",
            "height=100",
            "width=200",
            "",
            "[convolutional]",
            "activation=linear",
            "batch_normalize=0",
            "filters=1",
            "pad=0",
            "size=1",
            "stride=1",
            "",
            "[convolutional]",
            "activation=linear",
            "batch_normalize=0",
            "filters=1",
            "pad=0",
            "size=1",
            "stride=1",
            "",
            "[route]",
            f"layers=[{result}]",
            ""
    ])


def test_load_route_missing_layers():
    cfg = '\n'.join([
        "[net]",
        "height=100",
        "width=200",
        "channels=2",
        "",
        "[convolutional]",
        "",
        "[route]",
    ])

    with pytest.raises(d2k.network.ConfigurationError):
        d2k.network.load(cfg)


@pytest.mark.parametrize("layer", [-2, 1, 2])   # layers=1 is a self-reference
def test_load_route_out_of_range(layer):
    cfg = '\n'.join([
        "[net]",
        "height=100",
        "width=200",
        "channels=2",
        "",
        "[convolutional]",
        "",
        "[route]",
        f"layers={layer}"
    ])

    with pytest.raises(d2k.network.ConfigurationError):
        d2k.network.load(cfg)


def test_load_route_as_first_layer():
    cfg = '\n'.join([
        "[net]",
        "height=100",
        "width=200",
        "channels=2",
        "",
        "[route]",
        "layers=0",
    ])

    with pytest.raises(d2k.network.ConfigurationError):
        d2k.network.load(cfg)


@pytest.mark.parametrize("spec, result", [('-1', '1'), ('-2', '0')])
def test_load_resolves_negative_shortcut_layers(spec, result):
    cfg = '\n'.join([
        "[net]",
        "height=100",
        "width=200",
        "channels=2",
        "",
        "[convolutional]",
        "activation=linear",
        "batch_normalize=0",
        "filters=1",
        "pad=0",
        "size=1",
        "stride=1",
        "",
        "[convolutional]",
        "activation=linear",
        "batch_normalize=0",
        "filters=1",
        "pad=0",
        "size=1",
        "stride=1",
        "",
        "[shortcut]",
        "activation=linear",
        f"from={spec}"
    ])

    config = d2k.network.load(cfg)

    assert cfg_to_string(config) == '\n'.join([
            "[net]",
            "channels=2",
            "height=100",
            "width=200",
            "",
            "[convolutional]",
            "activation=linear",
            "batch_normalize=0",
            "filters=1",
            "pad=0",
            "size=1",
            "stride=1",
            "",
            "[convolutional]",
            "activation=linear",
            "batch_normalize=0",
            "filters=1",
            "pad=0",
            "size=1",
            "stride=1",
            "",
            "[shortcut]",
            "activation=linear",
            f"from={result}",
            ""
    ])


@pytest.mark.parametrize("layer", [-2, 1, 2])
def test_load_shortcut_out_of_range(layer):
    cfg = '\n'.join([
        "[net]",
        "height=100",
        "width=200",
        "channels=2",
        "",
        "[convolutional]",
        "activation=leaky",
        "",
        "[shortcut]",
        f"from={layer}"
    ])

    with pytest.raises(d2k.network.ConfigurationError):
        d2k.network.load(cfg)


def test_load_yolov3_doesnt_throw():
    d2k.network.load(yolov3_cfg.read_text())


def test_load_yolov4_doesnt_throw():
    d2k.network.load(yolov4_cfg.read_text())


def test_convert_no_layers():
    cfg = '\n'.join([
        "[net]",
        "height=100",
        "width=200",
        "channels=4"
    ])

    net = d2k.network.load(cfg).convert()
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

    net = d2k.network.load(cfg).convert()
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

    net = d2k.network.load(cfg).convert()
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
        d2k.network.load(cfg).convert()


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

    net = d2k.network.load(cfg).convert()
    assert net == [
        "layer_in = keras.Input(shape=(100, 200, 3))",
        "layer_0 = keras.layers.ZeroPadding2D(((1,1),(1,1)))(layer_in)",
        "layer_0 = keras.layers.Conv2D(32, 3, strides=2, use_bias=True, name='conv_0')(layer_0)",
        "layer_0 = keras.layers.LeakyReLU(alpha=.1)(layer_0)",
        "layer_out = layer_0"
    ]


def test_convert_convolutional_activation_mish():
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
        "activation=mish",
    ])

    net = d2k.network.load(cfg).convert()
    assert net == [
        "layer_in = keras.Input(shape=(100, 200, 3))",
        "layer_0 = keras.layers.ZeroPadding2D(((1,1),(1,1)))(layer_in)",
        "layer_0 = keras.layers.Conv2D(32, 3, strides=2, use_bias=True, name='conv_0')(layer_0)",
        "layer_0 = layer_0 * K.tanh(K.softplus(layer_0))",
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

    net = d2k.network.load(cfg).convert()
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
        d2k.network.load(cfg).convert()


def test_convert_maxpool():
    cfg = '\n'.join([
        "[net]",
        "height=100",
        "width=200",
        "channels=3",
        "",
        "[maxpool]",
    ])

    net = d2k.network.load(cfg).convert()
    assert net == [
        "layer_in = keras.Input(shape=(100, 200, 3))",
        "layer_0 = keras.layers.MaxPool2D(pool_size=1, strides=1)(layer_in)",
        "layer_out = layer_0"
    ]


def test_convert_maxpool_with_padding():
    cfg = '\n'.join([
        "[net]",
        "height=100",
        "width=200",
        "channels=3",
        "",
        "[maxpool]",
        "size=4",
        "stride=5",
    ])

    net = d2k.network.load(cfg).convert()
    assert net == [
        "layer_in = keras.Input(shape=(100, 200, 3))",
        "layer_0 = tf.pad(layer_in, tf.constant([[0,0],[1,2],[1,2],[0,0]]), constant_values=-np.inf)",
        "layer_0 = keras.layers.MaxPool2D(pool_size=4, strides=5)(layer_0)",
        "layer_out = layer_0"
    ]


def test_convert_maxpool_unsupported_option():
    cfg = '\n'.join([
        "[net]",
        "height=100",
        "width=200",
        "channels=3",
        "",
        "[maxpool]",
        "stride_x=10",
    ])

    with pytest.raises(ConversionError):
        d2k.network.load(cfg).convert()


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

    net = d2k.network.load(cfg).convert()
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

    net = d2k.network.load(cfg).convert()
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

    net = d2k.network.load(cfg).convert()
    assert net == [
        "layer_in = keras.Input(shape=(100, 200, 3))",
        "layer_0 = keras.layers.Conv2D(1, 1, strides=1, use_bias=True, name='conv_0')(layer_in)",
        "layer_1 = keras.layers.Conv2D(1, 1, strides=1, use_bias=True, name='conv_1')(layer_0)",
        "layer_2 = keras.layers.Conv2D(1, 1, strides=1, use_bias=True, name='conv_2')(layer_1)",
        "layer_3 = keras.layers.Concatenate()([layer_2, layer_0])",
        "layer_out = layer_3"
    ]


def test_convert_route_jumps_layers():
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
        "[route]",
        "layers=-2",
        "",
        "[route]",
        "layers=-2, -1"
    ])

    net = d2k.network.load(cfg).convert()
    assert net == [
        "layer_in = keras.Input(shape=(100, 200, 3))",
        "layer_0 = keras.layers.Conv2D(1, 1, strides=1, use_bias=True, name='conv_0')(layer_in)",
        "layer_1 = keras.layers.Conv2D(1, 1, strides=1, use_bias=True, name='conv_1')(layer_0)",
        "layer_2 = layer_0",
        "layer_3 = keras.layers.Concatenate()([layer_1, layer_2])",
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
        d2k.network.load(cfg).convert()


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

    net = d2k.network.load(cfg).convert()
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

    net = d2k.network.load(cfg).convert()
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

    net = d2k.network.load(cfg).convert()
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
        d2k.network.load(cfg).convert()


def test_convert_upsample_default():
    cfg = '\n'.join([
        "[net]",
        "height=100",
        "width=200",
        "channels=3",
        "",
        "[upsample]",
    ])

    net = d2k.network.load(cfg).convert()
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

    net = d2k.network.load(cfg).convert()
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
        d2k.network.load(cfg).convert()


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

    net = d2k.network.load(cfg).convert()
    assert net == [
        "layer_in = keras.Input(shape=(100, 200, 3))",
        "layer_0 = K.reshape(layer_in, (-1, *K.int_shape(layer_in)[1:3], 2, 15))",
        'layer_0_xy = keras.activations.sigmoid(layer_0[...,0:2])',
        'layer_0_wh = layer_0[...,2:4]',
        'layer_0_obj_classes = keras.activations.sigmoid(layer_0[...,4:])',
        'layer_0 = K.concatenate((layer_0_xy, layer_0_wh, layer_0_obj_classes))',
        'layer_0_l_w, layer_0_l_h = K.int_shape(layer_0)[1:3]',
        'layer_0_range_w = K.reshape(K.arange(0, layer_0_l_w, dtype="float32"), (1, 1, layer_0_l_w, 1, 1))',
        'layer_0_range_h = K.reshape(K.arange(0, layer_0_l_h, dtype="float32"), (1, layer_0_l_h, 1, 1, 1))',
        'layer_0_anchors_w = K.constant([10, 20], dtype="float32", shape=(1,1,1,2,1))',
        'layer_0_anchors_h = K.constant([10, 20], dtype="float32", shape=(1,1,1,2,1))',
        'layer_0_x = (layer_0[...,0:1] + layer_0_range_w) / layer_0_l_w',
        'layer_0_y = (layer_0[...,1:2] + layer_0_range_h) / layer_0_l_h',
        'layer_0_w = (K.exp(layer_0[...,2:3]) * layer_0_anchors_w) / 200',
        'layer_0_h = (K.exp(layer_0[...,3:4]) * layer_0_anchors_h) / 100',
        'layer_0 = K.concatenate((layer_0_x, layer_0_y, layer_0_w, layer_0_h, layer_0_obj_classes))',
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

    net = d2k.network.load(cfg).convert()
    assert net == [
        "layer_in = keras.Input(shape=(100, 200, 3))",
        "layer_0 = keras.layers.Conv2D(1, 1, strides=1, use_bias=True, name='conv_0')(layer_in)",
        'layer_1 = K.reshape(layer_0, (-1, *K.int_shape(layer_0)[1:3], 1, 15))',
        'layer_1_xy = keras.activations.sigmoid(layer_1[...,0:2])',
        'layer_1_wh = layer_1[...,2:4]',
        'layer_1_obj_classes = keras.activations.sigmoid(layer_1[...,4:])',
        'layer_1 = K.concatenate((layer_1_xy, layer_1_wh, layer_1_obj_classes))',
        'layer_1_l_w, layer_1_l_h = K.int_shape(layer_1)[1:3]',
        'layer_1_range_w = K.reshape(K.arange(0, layer_1_l_w, dtype="float32"), (1, 1, layer_1_l_w, 1, 1))',
        'layer_1_range_h = K.reshape(K.arange(0, layer_1_l_h, dtype="float32"), (1, layer_1_l_h, 1, 1, 1))',
        'layer_1_anchors_w = K.constant([10], dtype="float32", shape=(1,1,1,1,1))',
        'layer_1_anchors_h = K.constant([10], dtype="float32", shape=(1,1,1,1,1))',
        'layer_1_x = (layer_1[...,0:1] + layer_1_range_w) / layer_1_l_w',
        'layer_1_y = (layer_1[...,1:2] + layer_1_range_h) / layer_1_l_h',
        'layer_1_w = (K.exp(layer_1[...,2:3]) * layer_1_anchors_w) / 200',
        'layer_1_h = (K.exp(layer_1[...,3:4]) * layer_1_anchors_h) / 100',
        'layer_1 = K.concatenate((layer_1_x, layer_1_y, layer_1_w, layer_1_h, layer_1_obj_classes))',
        "layer_2 = layer_0",
        'layer_3 = K.reshape(layer_2, (-1, *K.int_shape(layer_2)[1:3], 1, 15))',
        'layer_3_xy = keras.activations.sigmoid(layer_3[...,0:2])',
        'layer_3_wh = layer_3[...,2:4]',
        'layer_3_obj_classes = keras.activations.sigmoid(layer_3[...,4:])',
        'layer_3 = K.concatenate((layer_3_xy, layer_3_wh, layer_3_obj_classes))',
        'layer_3_l_w, layer_3_l_h = K.int_shape(layer_3)[1:3]',
        'layer_3_range_w = K.reshape(K.arange(0, layer_3_l_w, dtype="float32"), (1, 1, layer_3_l_w, 1, 1))',
        'layer_3_range_h = K.reshape(K.arange(0, layer_3_l_h, dtype="float32"), (1, layer_3_l_h, 1, 1, 1))',
        'layer_3_anchors_w = K.constant([20], dtype="float32", shape=(1,1,1,1,1))',
        'layer_3_anchors_h = K.constant([20], dtype="float32", shape=(1,1,1,1,1))',
        'layer_3_x = (layer_3[...,0:1] + layer_3_range_w) / layer_3_l_w',
        'layer_3_y = (layer_3[...,1:2] + layer_3_range_h) / layer_3_l_h',
        'layer_3_w = (K.exp(layer_3[...,2:3]) * layer_3_anchors_w) / 200',
        'layer_3_h = (K.exp(layer_3[...,3:4]) * layer_3_anchors_h) / 100',
        'layer_3 = K.concatenate((layer_3_x, layer_3_y, layer_3_w, layer_3_h, layer_3_obj_classes))',

        "layer_out = [layer_1, layer_3]",
    ]


def test_convert_decode_grid_False():
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

    net = d2k.network.load(cfg).convert(decode_grid=False)
    assert net == [
        "layer_in = keras.Input(shape=(100, 200, 3))",
        "layer_0 = K.reshape(layer_in, (-1, *K.int_shape(layer_in)[1:3], 2, 15))",
        'layer_0_xy = keras.activations.sigmoid(layer_0[...,0:2])',
        'layer_0_wh = layer_0[...,2:4]',
        'layer_0_obj_classes = keras.activations.sigmoid(layer_0[...,4:])',
        'layer_0 = K.concatenate((layer_0_xy, layer_0_wh, layer_0_obj_classes))',
        "layer_out = layer_0",
    ]


def test_convert_yolo_scale_x_y():
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
        "scale_x_y=1.1",
    ])

    net = d2k.network.load(cfg).convert(decode_grid=False)
    assert net == [
        "layer_in = keras.Input(shape=(100, 200, 3))",
        "layer_0 = K.reshape(layer_in, (-1, *K.int_shape(layer_in)[1:3], 2, 15))",
        'layer_0_scale_x_y = K.constant(1.1, dtype="float32")',
        'layer_0_xy = keras.activations.sigmoid(layer_0[...,0:2]) * layer_0_scale_x_y - .5*(layer_0_scale_x_y - 1)',
        'layer_0_wh = layer_0[...,2:4]',
        'layer_0_obj_classes = keras.activations.sigmoid(layer_0[...,4:])',
        'layer_0 = K.concatenate((layer_0_xy, layer_0_wh, layer_0_obj_classes))',
        "layer_out = layer_0",
    ]


def test_convert_yolo_unsupported_nms_kind():
    cfg = '\n'.join([
        "[net]",
        "height=100",
        "width=200",
        "channels=2",
        "",
        "[yolo]",
        "anchors=10,10",
        "nms_kind=foobar"
    ])

    with pytest.raises(ConversionError):
        d2k.network.load(cfg).convert()


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
        d2k.network.load(cfg).convert()


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
        d2k.network.load(cfg).convert()


def test_convert_yolov3_doesnt_throw():
    with open(yolov3_cfg, 'r') as f:
        cfg = f.read()

    d2k.network.load(cfg).convert()


def test_convert_yolov4_doesnt_throw():
    with open(yolov4_cfg, 'r') as f:
        cfg = f.read()

    d2k.network.load(cfg).convert()


@pytest.mark.parametrize("i", [-2, 1, 2])
def test_layer_output_shape_out_of_range(i):
    cfg_text = '\n'.join([
        "[net]",
        "height=100",
        "width=150",
        "channels=3",
        "",
        "[convolutional]",
    ])

    network = d2k.network.load(cfg_text)

    with pytest.raises(IndexError):
        network.layer_output_shape(i)


def compare_layer_output_shape(cfg_text):
    network = d2k.network.load(cfg_text)
    k = network.make_model()
    assert network.layer_output_shape(-1) == k.output_shape[1:] # skipping batch dim


@pytest.mark.parametrize("size, stride", [(1,1),(2,1),(2,2),(3,1),(3,2)])
@pytest.mark.parametrize("pad", [0,1])
@pytest.mark.parametrize("bn", [0,1])
def test_layer_output_shape_convolutional(size, stride, pad, bn):
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
        f"activation=linear",
    ])

    compare_layer_output_shape(cfg_text)


def test_layer_output_shape_route():
    cfg_text = '\n'.join([
        "[net]",
        "height=100",
        "width=200",
        "channels=3",
        "",
        "[convolutional]",
        "pad=1",
        "activation=linear",
        "",
        "[convolutional]",
        "pad=1",
        "activation=linear",
        "",
        "[convolutional]",
        "pad=1",
        "activation=linear",
        "",
        "[route]",
        "layers=-1, -3"
    ])

    compare_layer_output_shape(cfg_text)


def test_layer_output_shape_shortcut():
    cfg_text = '\n'.join([
        "[net]",
        "height=100",
        "width=200",
        "channels=3",
        "",
        "[convolutional]",
        "pad=0",
        "activation=linear",
        "",
        "[convolutional]",
        "pad=0",
        "activation=linear",
        "",
        "[convolutional]",
        "pad=0",
        "activation=linear",
        "",
        "[shortcut]",
        "from=-3"
    ])

    compare_layer_output_shape(cfg_text)


@pytest.mark.parametrize("stride", [1, 2, 3])
def test_layer_output_shape_upsample(stride):
    cfg_text = '\n'.join([
        "[net]",
        "height=100",
        "width=200",
        "channels=3",
        "",
        "[upsample]",
        f"stride={stride}",
    ])

    compare_layer_output_shape(cfg_text)


@pytest.mark.parametrize("stride", [1, 2, 3, 5, 7])
@pytest.mark.parametrize("size", [2, 3, 4, 5, 7])
@pytest.mark.parametrize("padding", [None, 0, 2, 3])
def test_layer_output_shape_maxpool(stride, size, padding):
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

    compare_layer_output_shape(cfg_text)


@pytest.mark.parametrize("size", [2, 10, 20])
@pytest.mark.parametrize("classes", [3, 20])
@pytest.mark.parametrize("mask", [range(0,3), range(2,7), range(0,9)])
def test_layer_output_shape_yolo(size, classes, mask):
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

    compare_layer_output_shape(cfg_text)
