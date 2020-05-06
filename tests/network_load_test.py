import pytest
import d2k.network
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
