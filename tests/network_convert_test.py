import pytest
from d2k.network import Network, ConversionError
from pathlib import Path

yolov3_cfg = Path('darknet-files/yolov3.cfg')
yolov4_cfg = Path('darknet-files/yolov4.cfg')

# assume YOLOv4 requirements supported if this file exists
darknet_has_yolov4 = Path('darknet/cfg/yolov4.cfg').exists()

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

    net = Network.load(cfg).convert()
    assert net == [
        "layer_in = keras.Input(shape=(100, 200, 3))",
        "layer_0 = keras.layers.ZeroPadding2D(((1,1),(1,1)))(layer_in)",
        "layer_0 = keras.layers.Conv2D(32, 3, strides=2, use_bias=True, name='conv_0')(layer_0)",
        "layer_0 = layer_0 * K.tanh(K.switch(layer_0 > 20, layer_0, " +
                                   "K.switch(layer_0 < -20, K.exp(layer_0), K.log(K.exp(layer_0)+1))))",
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


def test_convert_maxpool():
    cfg = '\n'.join([
        "[net]",
        "height=100",
        "width=200",
        "channels=3",
        "",
        "[maxpool]",
    ])

    net = Network.load(cfg).convert()
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

    net = Network.load(cfg).convert()
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

    net = Network.load(cfg).convert()
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
        'layer_0_objectness = layer_0[...,4:5]',
        'layer_0_classes = layer_0[...,5:] * layer_0_objectness',
        'layer_0 = K.concatenate((layer_0_x, layer_0_y, layer_0_w, layer_0_h, layer_0_objectness, layer_0_classes))',
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
        'layer_1_objectness = layer_1[...,4:5]',
        'layer_1_classes = layer_1[...,5:] * layer_1_objectness',
        'layer_1 = K.concatenate((layer_1_x, layer_1_y, layer_1_w, layer_1_h, layer_1_objectness, layer_1_classes))',
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
        'layer_3_objectness = layer_3[...,4:5]',
        'layer_3_classes = layer_3[...,5:] * layer_3_objectness',
        'layer_3 = K.concatenate((layer_3_x, layer_3_y, layer_3_w, layer_3_h, layer_3_objectness, layer_3_classes))',

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

    net = Network.load(cfg).convert(just_activate_yolo=True)
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
        Network.load(cfg).convert()


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


def test_convert_yolov4_doesnt_throw():
    with open(yolov4_cfg, 'r') as f:
        cfg = f.read()

    Network.load(cfg).convert()
